"""Tax-lot analysis: what selling would actually cost, and what a loss is worth.

The through-line of this module is that a trade's true cost is fees PLUS the
tax it triggers, and a loss is an asset: harvested against gains or income it
is worth its size times your marginal rate. Both only work at lot level --
one symbol can simultaneously hold a lot worth harvesting and a lot that
should not be touched -- which is why holdings carry an ``acquired`` date.

Everything here is an ESTIMATE built on the two-rate model in the config.
Real tax law has thresholds, phase-outs, netting order and state wrinkles this
does not model. Treat the output as "worth a conversation with a professional",
never as the conversation itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from .config import Config
from .models import Holding

#: Days a lot must be held for its gain or loss to be long-term (US rule:
#: strictly more than one year).
LONG_TERM_DAYS = 365


@dataclass
class LotView:
    """One holding lot, valued and classified as of a given day."""

    lot: Holding
    price: float
    today: date

    @property
    def value(self) -> float:
        return self.lot.quantity * self.price

    @property
    def has_basis(self) -> bool:
        return bool(self.lot.acquired) and self.lot.cost_basis > 0

    @property
    def acquired_date(self) -> date | None:
        return date.fromisoformat(self.lot.acquired) if self.lot.acquired else None

    @property
    def gain(self) -> float:
        """Unrealised gain (negative = loss). Zero when basis is unknown."""
        return self.value - self.lot.cost_basis if self.has_basis else 0.0

    @property
    def days_held(self) -> int:
        acquired = self.acquired_date
        return (self.today - acquired).days if acquired else LONG_TERM_DAYS + 1

    @property
    def long_term(self) -> bool:
        return self.days_held > LONG_TERM_DAYS

    @property
    def days_to_long_term(self) -> int:
        return max(0, LONG_TERM_DAYS + 1 - self.days_held)


@dataclass
class HarvestCandidate:
    view: LotView
    tax_saving: float
    replacement: str
    wash_risk: str = ""
    wash_applies: bool = True

    @property
    def symbol(self) -> str:
        return self.view.lot.symbol


@dataclass
class SellAdvice:
    """What selling a lot right now would trigger, and any reason to wait."""

    view: LotView
    tax_cost: float
    rate_applied: float
    wait: str = ""


@dataclass
class TaxReport:
    today: date
    harvest: list[HarvestCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unknown_basis: list[str] = field(default_factory=list)

    @property
    def total_harvestable_loss(self) -> float:
        return sum(-c.view.gain for c in self.harvest)

    @property
    def total_tax_saving(self) -> float:
        return sum(c.tax_saving for c in self.harvest)


def lot_views(
    cfg: Config, prices: dict[str, float], today: date | None = None
) -> list[LotView]:
    today = today or date.today()
    views = []
    for lot in cfg.holdings:
        price = prices.get(lot.symbol)
        if price is None or price <= 0 or lot.quantity <= 0:
            continue
        views.append(LotView(lot=lot, price=price, today=today))
    return views


def _wash_sale_applies(cfg: Config, symbol: str) -> bool:
    if cfg.asset(symbol).asset_class != "crypto":
        return True
    return cfg.tax.crypto_wash_sale


def _recent_buy_conflicts(
    cfg: Config, views: list[LotView], candidate: LotView
) -> list[LotView]:
    """Other lots of the same symbol bought inside the wash window.

    Selling a lot at a loss while a purchase of the same asset sits within
    ``wash_sale_days`` before the sale disallows (part of) the loss. Lots are
    the only purchase record this tool has, so that is what it checks; buys
    made outside the config are invisible to it.
    """
    window = timedelta(days=cfg.tax.wash_sale_days)
    conflicts = []
    for other in views:
        if other is candidate or other.lot.symbol != candidate.lot.symbol:
            continue
        acquired = other.acquired_date
        if acquired is not None and candidate.today - acquired <= window:
            conflicts.append(other)
    return conflicts


def rate_for(cfg: Config, view: LotView) -> float:
    return cfg.tax.long_term_rate if view.long_term else cfg.tax.short_term_rate


def sell_advice(cfg: Config, view: LotView) -> SellAdvice:
    """The tax consequence of selling this lot today, plus timing advice."""
    gain = view.gain
    rate = rate_for(cfg, view)
    tax = max(0.0, gain) * rate
    wait = ""
    if (
        gain > 0
        and not view.long_term
        and view.days_to_long_term <= cfg.tax.long_term_boundary_days
        and cfg.tax.short_term_rate > cfg.tax.long_term_rate
    ):
        saved = gain * (cfg.tax.short_term_rate - cfg.tax.long_term_rate)
        wait = (
            f"waiting {view.days_to_long_term} days makes this gain long-term "
            f"and saves about ${saved:,.2f} in tax"
        )
    return SellAdvice(view=view, tax_cost=tax, rate_applied=rate, wait=wait)


def review(
    cfg: Config, prices: dict[str, float], today: date | None = None
) -> TaxReport:
    """Scan every lot for harvestable losses and tax timing hazards."""
    today = today or date.today()
    report = TaxReport(today=today)
    if cfg.tax.short_term_rate <= 0 and cfg.tax.long_term_rate <= 0:
        report.notes.append(
            "both tax rates are zero (tax-sheltered account?); harvesting and "
            "gain-timing advice are moot"
        )
        return report

    views = lot_views(cfg, prices, today)
    for view in views:
        if not view.has_basis:
            report.unknown_basis.append(view.lot.symbol)
            continue

        gain = view.gain
        if gain < 0:
            loss = -gain
            saving = loss * rate_for(cfg, view)
            if loss < cfg.tax.harvest_min_loss or saving < cfg.tax.harvest_min_saving:
                continue
            asset = cfg.asset(view.lot.symbol)
            replacement = asset.tlh_replacement
            wash_risk = ""
            if _wash_sale_applies(cfg, view.lot.symbol):
                conflicts = _recent_buy_conflicts(cfg, views, view)
                if conflicts:
                    days = min(
                        cfg.tax.wash_sale_days - (today - c.acquired_date).days
                        for c in conflicts
                        if c.acquired_date is not None
                    )
                    wash_risk = (
                        f"a lot of {view.lot.symbol} was bought within the last "
                        f"{cfg.tax.wash_sale_days} days; selling now would wash out "
                        f"the loss. Clear in {days} day(s)."
                    )
            report.harvest.append(
                HarvestCandidate(
                    view=view, tax_saving=saving,
                    replacement=replacement, wash_risk=wash_risk,
                    wash_applies=_wash_sale_applies(cfg, view.lot.symbol),
                )
            )
        elif gain > 0:
            advice = sell_advice(cfg, view)
            if advice.wait:
                report.warnings.append(f"{view.lot.symbol}: {advice.wait}")

    if report.harvest:
        washable = [c for c in report.harvest if _wash_sale_applies(cfg, c.symbol)]
        if washable:
            report.notes.append(
                f"after harvesting, do not rebuy the same asset for "
                f"{cfg.tax.wash_sale_days} days or the loss is disallowed "
                "(this includes automatic reinvestment and buys in other "
                "accounts, including IRAs, which this tool cannot see)"
            )
        crypto_harvest = [
            c for c in report.harvest
            if cfg.asset(c.symbol).asset_class == "crypto" and not cfg.tax.crypto_wash_sale
        ]
        if crypto_harvest:
            report.notes.append(
                "crypto losses are treated here as exempt from the wash-sale rule, "
                "matching US law as of this tool's writing -- but that treatment "
                "is a recurring target of legislation. Verify before relying on it."
            )
        report.notes.append(
            "harvested losses offset capital gains first, then up to $3,000 of "
            "ordinary income per year; the rest carries forward"
        )
    report.notes.append(
        "estimates use your configured marginal rates and ignore netting order, "
        "thresholds and state rules -- confirm with a tax professional"
    )
    return report


def cost_of_selling(
    cfg: Config, symbol: str, notional: float,
    prices: dict[str, float], today: date | None = None,
) -> tuple[float, list[str]]:
    """Estimated tax triggered by selling ``notional`` dollars of a symbol.

    Assumes highest-cost-first lot selection (HIFO), which minimises the
    realised gain -- most brokers let you pick lots, and if you are selling
    anyway this is usually the order to pick them in.
    """
    views = [v for v in lot_views(cfg, prices, today) if v.lot.symbol == symbol]
    views.sort(key=lambda v: -(v.lot.cost_basis / max(v.lot.quantity, 1e-12)))
    remaining = notional
    tax = 0.0
    notes: list[str] = []
    for view in views:
        if remaining <= 0:
            break
        take = min(remaining, view.value)
        if view.value <= 0:
            continue
        fraction = take / view.value
        if view.has_basis:
            gain = view.gain * fraction
            tax += max(0.0, gain) * rate_for(cfg, view)
        else:
            notes.append(
                f"a {symbol} lot has no cost basis recorded; its tax cost is unknown"
            )
        remaining -= take
    return tax, notes
