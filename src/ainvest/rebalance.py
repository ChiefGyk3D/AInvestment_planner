"""Rebalancing that respects the two things small accounts actually feel:
transaction costs and taxes.

The default is cash-flow rebalancing -- steer the portfolio by *where new
money goes*, never by selling. It is slower than selling winners to buy
losers, but it costs nothing in fees on the sell side and triggers no taxable
gains, which for a few hundred dollars is almost always the better trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .lots import cost_of_selling
from .models import Allocation, Holding, Plan, Trade
from .plan import notional_for_budget


@dataclass
class Drift:
    symbol: str
    current_value: float
    current_weight: float
    target_weight: float

    @property
    def drift(self) -> float:
        return self.current_weight - self.target_weight


@dataclass
class RebalanceReport:
    total_value: float
    new_cash: float
    drifts: list[Drift] = field(default_factory=list)
    plan: Plan | None = None
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def max_drift(self) -> float:
        return max((abs(d.drift) for d in self.drifts), default=0.0)


def holdings_value(
    holdings: list[Holding], prices: dict[str, float]
) -> tuple[float, dict[str, float]]:
    values: dict[str, float] = {}
    for h in holdings:
        price = prices.get(h.symbol)
        if price is None or price <= 0:
            continue
        values[h.symbol] = values.get(h.symbol, 0.0) + h.quantity * price
    return sum(values.values()), values


def plan_rebalance(
    cfg: Config,
    allocation: Allocation,
    prices: dict[str, float],
    *,
    new_cash: float | None = None,
    allow_sells: bool | None = None,
) -> RebalanceReport:
    """Work out what to trade to move current holdings toward the target."""
    new_cash = cfg.investable if new_cash is None else new_cash
    allow_sells = cfg.allow_sells if allow_sells is None else allow_sells

    held_value, values = holdings_value(cfg.holdings, prices)
    total = held_value + new_cash
    report = RebalanceReport(total_value=total, new_cash=new_cash)
    if total <= 0:
        report.notes.append("portfolio is empty and there is no new cash to invest")
        return report

    targets = allocation.as_dict()
    symbols = sorted(set(targets) | set(values))
    for symbol in symbols:
        report.drifts.append(
            Drift(
                symbol=symbol,
                current_value=values.get(symbol, 0.0),
                current_weight=values.get(symbol, 0.0) / total if total else 0.0,
                target_weight=targets.get(symbol, 0.0),
            )
        )

    plan = Plan(allocation=allocation, budget=new_cash)
    band = cfg.rebalance_band

    # The fee-payback gate: a trade must earn its fee back within
    # cfg.fee_payback_months at the portfolio's expected return, or it is not
    # worth placing. The rate is floored so a bearish return estimate does not
    # veto every trade -- the gate is about proportion, not forecasting.
    benefit_rate = max(allocation.expected_return, 0.03)
    max_fee_fraction = benefit_rate * cfg.fee_payback_months / 12.0

    if allow_sells:
        _two_sided(cfg, report, plan, prices, total, targets, values, band,
                   max_fee_fraction)
    else:
        _cash_flow(cfg, report, plan, prices, total, targets, values, new_cash, band,
                   max_fee_fraction)

    buys = sum(t.notional for t in plan.trades if t.side == "buy")
    plan.deployed = buys
    plan.total_fees = sum(t.fee for t in plan.trades)
    sells = sum(t.notional for t in plan.trades if t.side == "sell")
    plan.cash_left = new_cash + sells - buys - plan.total_fees
    report.plan = plan

    if report.max_drift <= band and not plan.trades:
        report.notes.append(
            f"every position is within the {band:.1%} no-trade band; doing nothing "
            "is the correct move"
        )
    return report


def _cash_flow(cfg, report, plan, prices, total, targets, values, new_cash, band,
               max_fee_fraction) -> None:
    """Direct new money at whatever is furthest below target. No selling."""
    shortfalls = []
    for symbol, weight in targets.items():
        gap = weight * total - values.get(symbol, 0.0)
        if gap > 0:
            shortfalls.append((symbol, gap))
    if not shortfalls:
        report.notes.append(
            "nothing is below target, so new cash has no obviously underweight home; "
            "either raise the target weights or hold the cash"
        )
        return

    overweight = [
        d.symbol for d in report.drifts if d.drift > band and d.target_weight > 0
    ]
    if overweight:
        report.notes.append(
            "held above target and left alone (selling would cost fees and possibly "
            f"tax): {', '.join(overweight)}"
        )

    total_gap = sum(gap for _, gap in shortfalls)
    remaining = new_cash
    for symbol, gap in sorted(shortfalls, key=lambda x: -x[1]):
        if remaining <= 0:
            break
        asset = cfg.asset(symbol)
        share = min(remaining, new_cash * gap / total_gap)
        share = min(share, gap)
        venue = cfg.cheapest_venue(asset, share)
        notional = notional_for_budget(venue, share)
        if notional < max(venue.min_notional, cfg.min_trade):
            report.skipped.append(
                (symbol, f"${notional:,.2f} is under the ${max(venue.min_notional, cfg.min_trade):,.2f} minimum")
            )
            continue
        price = prices[symbol]
        quantity = venue.quantize(notional / price)
        if quantity <= 0:
            report.skipped.append((symbol, "rounds to zero units"))
            continue
        notional = quantity * price
        fee = venue.cost(notional)
        if fee > notional * max_fee_fraction:
            months = 12.0 * fee / (notional * max(max_fee_fraction, 1e-9)) \
                * (cfg.fee_payback_months / 12.0)
            report.skipped.append(
                (symbol,
                 f"the ${fee:,.2f} fee would take ~{months:.0f} months of expected "
                 f"return to earn back (limit: {cfg.fee_payback_months:.0f}); "
                 "skipped as not worth it at this size")
            )
            continue
        plan.trades.append(
            Trade(symbol=symbol, side="buy", venue=venue.name, notional=notional,
                  quantity=quantity, price=price, fee=fee)
        )
        remaining -= notional + fee


def _two_sided(cfg, report, plan, prices, total, targets, values, band,
               max_fee_fraction) -> None:
    """Sell overweights and buy underweights. Only when explicitly enabled."""
    report.notes.append(
        "selling is enabled: these trades can realise capital gains and, in a "
        "taxable account, may trigger wash-sale rules if you rebuy within 30 days"
    )
    for drift in report.drifts:
        gap_value = (drift.target_weight - drift.current_weight) * total
        if abs(drift.drift) <= band:
            continue
        symbol = drift.symbol
        if symbol not in targets and drift.current_value <= 0:
            continue
        asset = cfg.asset(symbol)
        side = "buy" if gap_value > 0 else "sell"
        magnitude = abs(gap_value)
        venue = cfg.cheapest_venue(asset, magnitude)
        if magnitude < max(venue.min_notional, cfg.min_trade):
            report.skipped.append(
                (symbol, f"${magnitude:,.2f} drift is not worth a trade")
            )
            continue
        price = prices[symbol]
        notional = notional_for_budget(venue, magnitude) if side == "buy" else magnitude
        if side == "sell":
            notional = min(notional, drift.current_value)
        quantity = venue.quantize(notional / price)
        if quantity <= 0:
            report.skipped.append((symbol, "rounds to zero units"))
            continue
        notional = quantity * price
        fee = venue.cost(notional)
        friction = fee
        if side == "sell":
            # Selling can trigger tax; that is part of the trade's real cost.
            tax, tax_notes = cost_of_selling(cfg, symbol, notional, prices)
            friction += tax
            for note in tax_notes:
                if note not in report.notes:
                    report.notes.append(note)
            if tax > 0:
                report.notes.append(
                    f"selling ${notional:,.2f} of {symbol} realises an estimated "
                    f"${tax:,.2f} of tax (HIFO lot order assumed)"
                )
        if friction > notional * max(max_fee_fraction, 0.0) and friction > fee:
            report.skipped.append(
                (symbol,
                 f"fees plus an estimated ${friction - fee:,.2f} of tax exceed what "
                 f"this ${notional:,.2f} correction is worth; drift can also be "
                 "fixed for free by pointing new contributions here")
            )
            continue
        if fee > notional * max_fee_fraction:
            report.skipped.append(
                (symbol,
                 f"the ${fee:,.2f} fee fails the {cfg.fee_payback_months:.0f}-month "
                 "payback gate for this trade size")
            )
            continue
        plan.trades.append(
            Trade(symbol=symbol, side=side, venue=venue.name, notional=notional,
                  quantity=quantity, price=price, fee=fee)
        )
