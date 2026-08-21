"""Turn target weights into orders you can actually place.

This is where a portfolio optimiser usually stops and a small account starts
losing money. A 2% weight of a $300 budget is a $6 order; if the venue charges
$1 to place it, the position is down 17% before the market opens. So the rules
here are about size, not risk: drop what is too small to be worth buying,
redistribute it into what is left, and quote the total cost honestly.
"""

from __future__ import annotations

from .config import Config
from .models import Allocation, Plan, Trade, Venue


def notional_for_budget(venue: Venue, budget: float) -> float:
    """Largest order whose notional *plus fees* fits inside ``budget``.

    Inverts ``notional + venue.cost(notional) = budget``. The closed form
    handles the fixed and percentage components; the check afterwards catches
    the case where the minimum fee binds.
    """
    if budget <= 0:
        return 0.0
    rate = (venue.fee_bps + venue.spread_bps) / 10_000.0
    guess = (budget - venue.fixed_fee) / (1.0 + rate)
    if guess <= 0:
        return 0.0
    if venue.cost(guess) > budget - guess + 1e-12:
        # The fee floor binds: back the notional out of it directly.
        guess = budget - venue.min_fee
    return max(0.0, guess)


def _prices_for(allocation: Allocation, prices: dict[str, float]) -> dict[str, float]:
    missing = [s for s, w in allocation.nonzero() if s not in prices or prices[s] <= 0]
    if missing:
        raise ValueError(f"no usable price for: {', '.join(missing)}")
    return prices


def build_plan(
    cfg: Config,
    allocation: Allocation,
    prices: dict[str, float],
    *,
    budget: float | None = None,
) -> Plan:
    """Produce the buy list for a fresh deployment of cash."""
    budget = cfg.investable if budget is None else budget
    _prices_for(allocation, prices)

    weights = dict(allocation.nonzero())
    plan = Plan(allocation=allocation, budget=budget)
    if not weights or budget <= 0:
        plan.cash_left = budget
        plan.warnings.append("nothing to buy: no positive weights or no budget")
        return plan

    # Drop anything too small for its venue, then re-spread its weight over
    # whatever survives. Repeat, because redistribution can push a *different*
    # position below its own minimum.
    live = dict(weights)
    for _ in range(len(weights) + 1):
        total_w = sum(live.values())
        if total_w <= 0:
            break
        too_small = {}
        for symbol, w in live.items():
            asset = cfg.asset(symbol)
            share = budget * w / total_w
            venue = cfg.cheapest_venue(asset, share)
            if notional_for_budget(venue, share) < venue.min_notional:
                too_small[symbol] = (venue.min_notional, share)
        if not too_small:
            break
        if len(too_small) == len(live):
            # Everything is below its minimum individually. Concentrating the
            # whole budget into the largest weight may still clear one venue's
            # minimum -- but only keep it if it actually does, otherwise the
            # plan proposes an order the venue would reject.
            keep = max(live, key=lambda s: live[s])
            keep_venue = cfg.cheapest_venue(cfg.asset(keep), budget)
            if notional_for_budget(keep_venue, budget) >= keep_venue.min_notional:
                too_small.pop(keep, None)
        for symbol, (minimum, share) in too_small.items():
            live.pop(symbol, None)
            plan.dropped.append(
                (symbol, f"${share:,.2f} is below the ${minimum:,.2f} venue minimum")
            )

    if not live:
        plan.cash_left = budget
        plan.warnings.append(
            f"every position came out below its venue minimum at a ${budget:,.2f} "
            "budget. Raise the budget, cut the universe, or lower venue min_notional."
        )
        return plan

    total_w = sum(live.values())
    for symbol in sorted(live, key=lambda s: -live[s]):
        asset = cfg.asset(symbol)
        price = prices[symbol]
        share = budget * live[symbol] / total_w
        venue = cfg.cheapest_venue(asset, share)

        notional = notional_for_budget(venue, share)
        quantity = venue.quantize(notional / price)
        if quantity <= 0:
            plan.dropped.append(
                (symbol, f"${share:,.2f} buys less than one tradable unit at ${price:,.2f}")
            )
            continue
        notional = quantity * price
        fee = venue.cost(notional)
        plan.trades.append(
            Trade(
                symbol=symbol, side="buy", venue=venue.name, notional=notional,
                quantity=quantity, price=price, fee=fee,
            )
        )

    plan.deployed = sum(t.notional for t in plan.trades)
    plan.total_fees = sum(t.fee for t in plan.trades)
    plan.cash_left = budget - plan.deployed - plan.total_fees
    _add_warnings(cfg, plan)
    return plan


def _add_warnings(cfg: Config, plan: Plan) -> None:
    if plan.fee_drag_bps > 100:
        plan.warnings.append(
            f"fees are {plan.fee_drag_bps:,.0f} bps ({plan.fee_drag_bps / 100:.2f}%) of "
            "the amount invested. Above ~1% you are handing over a year of a bond "
            "fund's return just to open the position; consider fewer, larger orders "
            "or a cheaper venue."
        )
    if plan.cash_left < -0.005:  # anything smaller is float noise, not real money
        plan.warnings.append(
            f"the plan overspends by ${abs(plan.cash_left):,.2f}; check venue fee settings"
        )
    if len(plan.trades) > 8 and plan.budget < 1000:
        plan.warnings.append(
            f"{len(plan.trades)} orders for a ${plan.budget:,.0f} budget is a lot of "
            "positions to hold. Fewer, broader holdings usually beat many tiny ones."
        )
    if plan.deployed > 0:
        smallest = min(plan.trades, key=lambda t: t.notional)
        if smallest.notional < 10:
            plan.warnings.append(
                f"smallest order is ${smallest.notional:,.2f} ({smallest.symbol}); "
                "positions this small rarely move the needle on the portfolio"
            )


def realised_weights(plan: Plan) -> dict[str, float]:
    """What you actually end up holding, after minimums and rounding."""
    if plan.deployed <= 0:
        return {}
    return {t.symbol: t.notional / plan.deployed for t in plan.trades}


def tracking_error(plan: Plan) -> float:
    """Total absolute drift between target and realised weights."""
    realised = realised_weights(plan)
    target = plan.allocation.as_dict()
    keys = set(realised) | set(target)
    return float(sum(abs(realised.get(k, 0.0) - target.get(k, 0.0)) for k in keys))
