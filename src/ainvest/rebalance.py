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
    approve_sells: bool = False,
) -> RebalanceReport:
    """Work out what to trade to move current holdings toward the target.

    ``approve_sells`` is the explicit, per-run consent to sell beyond the
    programmed withdrawal allowance (``withdrawal.auto_limit``). Without it,
    the plan will never propose pulling out more than the allowance -- that is
    a hard policy, not a preference, and no signal or optimiser output
    overrides it.
    """
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
                   max_fee_fraction, approve_sells, new_cash)
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
               max_fee_fraction, approve_sells, new_cash) -> None:
    """Sell overweights, then buy underweights WITH THE MONEY THAT EXISTS.

    Two passes on purpose. Sells run first, subject to the withdrawal
    allowance, the tax-aware friction gate and the fee-payback gate; whatever
    they actually raise -- not what the drift arithmetic wished for -- joins
    the new cash to fund the buys. An earlier version sized both sides off
    drift independently, and the moment the withdrawal policy withheld a sell
    the plan quietly proposed buys with money it did not have.
    """
    report.notes.append(
        "selling is enabled: these trades can realise capital gains and, in a "
        "taxable account, may trigger wash-sale rules if you rebuy within 30 days"
    )
    # The withdrawal policy: sells beyond the programmed allowance need
    # explicit per-run approval. Drifts are processed largest-first, so the
    # allowance goes to the corrections that matter most.
    sell_allowance = float("inf") if approve_sells else cfg.withdrawal.auto_limit
    withheld = 0.0
    proceeds = 0.0

    ranked = sorted(report.drifts, key=lambda d: -abs(d.drift))

    # ---- Pass 1: sells raise cash ----------------------------------------
    for drift in ranked:
        if drift.drift <= band:
            continue
        symbol = drift.symbol
        magnitude = min((drift.current_weight - drift.target_weight) * total,
                        drift.current_value)
        asset = cfg.asset(symbol)
        venue = cfg.cheapest_venue(asset, magnitude)
        floor = max(cfg.min_trade, venue.min_notional)
        if magnitude < floor:
            report.skipped.append(
                (symbol, f"${magnitude:,.2f} drift is not worth a trade")
            )
            continue
        if magnitude > sell_allowance:
            if sell_allowance >= floor:
                withheld += magnitude - sell_allowance
                magnitude = sell_allowance
            else:
                withheld += magnitude
                report.skipped.append(
                    (symbol,
                     f"selling ${magnitude:,.2f} exceeds the programmed withdrawal "
                     f"allowance (${cfg.withdrawal.auto_limit:,.2f} per run); "
                     "withheld pending your approval")
                )
                continue

        price = prices[symbol]
        quantity = venue.quantize(magnitude / price)
        if quantity <= 0:
            report.skipped.append((symbol, "rounds to zero units"))
            continue
        notional = quantity * price
        fee = venue.cost(notional)
        tax, tax_notes = cost_of_selling(cfg, symbol, notional, prices)
        for note in tax_notes:
            if note not in report.notes:
                report.notes.append(note)
        if tax > 0:
            report.notes.append(
                f"selling ${notional:,.2f} of {symbol} realises an estimated "
                f"${tax:,.2f} of tax (HIFO lot order assumed)"
            )
        friction = fee + tax
        if friction > notional * max(max_fee_fraction, 0.0) and friction > fee:
            report.skipped.append(
                (symbol,
                 f"fees plus an estimated ${tax:,.2f} of tax exceed what this "
                 f"${notional:,.2f} correction is worth; drift can also be fixed "
                 "for free by pointing new contributions here")
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
            Trade(symbol=symbol, side="sell", venue=venue.name, notional=notional,
                  quantity=quantity, price=price, fee=fee)
        )
        sell_allowance -= notional
        proceeds += notional - fee

    # ---- Pass 2: buys spend what is actually available -------------------
    available = new_cash + proceeds
    for drift in ranked:
        if -drift.drift <= band:
            continue
        if available <= 0:
            break
        symbol = drift.symbol
        if symbol not in targets:
            continue
        gap = (drift.target_weight - drift.current_weight) * total
        spend = min(gap, available)
        asset = cfg.asset(symbol)
        venue = cfg.cheapest_venue(asset, spend)
        floor = max(cfg.min_trade, venue.min_notional)
        notional = notional_for_budget(venue, spend)
        if notional < floor:
            report.skipped.append(
                (symbol, f"${notional:,.2f} is under the ${floor:,.2f} minimum")
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
            report.skipped.append(
                (symbol,
                 f"the ${fee:,.2f} fee fails the {cfg.fee_payback_months:.0f}-month "
                 "payback gate for this trade size")
            )
            continue
        plan.trades.append(
            Trade(symbol=symbol, side="buy", venue=venue.name, notional=notional,
                  quantity=quantity, price=price, fee=fee)
        )
        available -= notional + fee

    if withheld > 0:
        report.notes.append(
            f"${withheld:,.2f} of selling was withheld by the withdrawal policy "
            f"(programmed allowance: ${cfg.withdrawal.auto_limit:,.2f} per run). "
            "Nothing leaves this portfolio beyond that without you saying so: "
            "re-run with --approve-sells to approve this specific plan."
        )
