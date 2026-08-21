"""Planning is where a small account actually loses money, so these tests are
mostly about money: never spend more than the budget, never place an order too
small to be worth its fee, and always say what the fees cost."""

from __future__ import annotations

import pytest

from ainvest.allocator import optimize_weights
from ainvest.config import from_dict
from ainvest.models import Allocation, Venue
from ainvest.plan import build_plan, notional_for_budget, realised_weights, tracking_error
from conftest import base_config


@pytest.fixture
def allocation(cfg, moments):
    return optimize_weights(cfg, moments)


class TestNotionalForBudget:
    """`notional + fees == budget` is the invariant that stops overspending."""

    @pytest.mark.parametrize(
        "venue",
        [
            Venue("free"),
            Venue("bps", fee_bps=60),
            Venue("spread", spread_bps=150),
            Venue("fixed", fixed_fee=0.99),
            Venue("both", fee_bps=25, spread_bps=25, fixed_fee=0.10),
        ],
    )
    @pytest.mark.parametrize("budget", [5.0, 25.0, 100.0, 1000.0])
    def test_notional_plus_fee_equals_budget(self, venue, budget):
        notional = notional_for_budget(venue, budget)
        assert notional + venue.cost(notional) == pytest.approx(budget, abs=1e-9)

    def test_a_budget_smaller_than_the_fixed_fee_buys_nothing(self):
        assert notional_for_budget(Venue("v", fixed_fee=5.0), 3.0) == 0.0

    def test_zero_budget_buys_nothing(self):
        assert notional_for_budget(Venue("v"), 0.0) == 0.0


def test_plan_never_spends_more_than_the_budget(cfg, allocation, latest):
    for budget in (25, 50, 100, 300, 1000, 5000):
        plan = build_plan(cfg, allocation, latest, budget=budget)
        assert plan.deployed + plan.total_fees <= budget + 1e-6, budget
        assert plan.cash_left >= -1e-6


def test_plan_deploys_nearly_all_of_it(cfg, allocation, latest):
    plan = build_plan(cfg, allocation, latest, budget=300)
    assert plan.deployed > 300 * 0.95


def test_fees_are_charged_at_the_chosen_venue(cfg, allocation, latest):
    plan = build_plan(cfg, allocation, latest, budget=1000)
    for trade in plan.trades:
        assert trade.fee == pytest.approx(cfg.venues[trade.venue].cost(trade.notional))
    assert plan.total_fees == pytest.approx(sum(t.fee for t in plan.trades))


def test_quantities_reconcile_with_notionals(cfg, allocation, latest):
    plan = build_plan(cfg, allocation, latest, budget=800)
    for trade in plan.trades:
        assert trade.quantity * trade.price == pytest.approx(trade.notional, rel=1e-9)
        assert trade.quantity > 0


def test_orders_below_the_venue_minimum_are_dropped_and_redistributed(latest, moments):
    """The money from a dropped order must go somewhere, not vanish."""
    raw = base_config()
    raw["venues"]["public"]["min_notional"] = 60.0
    raw["venues"]["coinbase_advanced"]["min_notional"] = 60.0
    cfg = from_dict(raw)
    allocation = optimize_weights(cfg, moments)
    plan = build_plan(cfg, allocation, latest, budget=150)
    assert plan.dropped, "a $150 budget cannot fund four $60-minimum orders"
    assert all(t.notional >= 60.0 - 1e-9 for t in plan.trades)
    assert plan.deployed + plan.total_fees <= 150 + 1e-6
    assert plan.deployed > 100, "dropped weight should have moved to the survivors"


def test_a_budget_too_small_for_anything_is_reported_not_crashed(latest, moments):
    raw = base_config()
    raw["venues"]["public"]["min_notional"] = 500.0
    raw["venues"]["coinbase_advanced"]["min_notional"] = 500.0
    cfg = from_dict(raw)
    allocation = optimize_weights(cfg, moments)
    plan = build_plan(cfg, allocation, latest, budget=20)
    assert plan.trades == []
    assert plan.cash_left == pytest.approx(20)
    assert any("below its venue minimum" in w for w in plan.warnings)


def test_whole_share_venue_rounds_down_and_leaves_change(moments):
    raw = base_config()
    raw["venues"]["public"]["fractional"] = False
    cfg = from_dict(raw)
    allocation = optimize_weights(cfg, moments)
    prices = {"VTI": 250.0, "GLD": 190.0, "BTC": 60000.0, "ETH": 3000.0}
    plan = build_plan(cfg, allocation, prices, budget=2000)
    for trade in plan.trades:
        if trade.venue == "public":
            assert trade.quantity == int(trade.quantity)
    assert plan.cash_left >= 0


def test_expensive_venues_raise_a_fee_warning(moments, latest):
    raw = base_config()
    raw["venues"]["public"]["spread_bps"] = 400
    raw["venues"]["coinbase_advanced"]["spread_bps"] = 400
    cfg = from_dict(raw)
    allocation = optimize_weights(cfg, moments)
    plan = build_plan(cfg, allocation, latest, budget=300)
    assert plan.fee_drag_bps > 100
    assert any("fees are" in w for w in plan.warnings)


def test_free_venue_reports_no_fee_drag(moments, latest):
    raw = base_config()
    raw["venues"]["coinbase_advanced"].update({"fee_bps": 0, "spread_bps": 0})
    cfg = from_dict(raw)
    allocation = optimize_weights(cfg, moments)
    plan = build_plan(cfg, allocation, latest, budget=300)
    assert plan.total_fees == pytest.approx(0.0)
    assert plan.fee_drag_bps == pytest.approx(0.0)


def test_realised_weights_track_the_target(cfg, allocation, latest):
    plan = build_plan(cfg, allocation, latest, budget=5000)
    realised = realised_weights(plan)
    assert sum(realised.values()) == pytest.approx(1.0)
    # With a comfortable budget, rounding should barely move anything.
    assert tracking_error(plan) < 0.02


def test_tracking_error_grows_when_minimums_start_to_bite(moments, latest):
    """Venue minimums are what pull a small account away from its target.

    With $1 minimums even a $60 budget tracks well, so this uses a venue whose
    minimum is large enough to actually force positions out -- which is the
    situation the tracking-error number exists to warn about.
    """
    raw = base_config()
    raw["venues"]["public"]["min_notional"] = 50.0
    raw["venues"]["coinbase_advanced"]["min_notional"] = 50.0
    cfg = from_dict(raw)
    allocation = optimize_weights(cfg, moments)

    roomy = build_plan(cfg, allocation, latest, budget=10000)
    cramped = build_plan(cfg, allocation, latest, budget=120)

    assert not roomy.dropped
    assert cramped.dropped, "a $120 budget cannot fund four $50 minimums"
    assert tracking_error(cramped) > tracking_error(roomy)


def test_empty_allocation_is_handled(cfg):
    empty = Allocation(symbols=["VTI"], weights=[0.0])
    plan = build_plan(cfg, empty, {"VTI": 100.0}, budget=300)
    assert plan.trades == []
    assert plan.cash_left == pytest.approx(300)
    assert plan.warnings


def test_missing_price_is_an_explicit_error(cfg, allocation):
    with pytest.raises(ValueError, match="no usable price"):
        build_plan(cfg, allocation, {"VTI": 100.0}, budget=300)


def test_trades_are_grouped_by_venue(cfg, allocation, latest):
    plan = build_plan(cfg, allocation, latest, budget=1000)
    grouped = plan.by_venue()
    assert sum(len(v) for v in grouped.values()) == len(plan.trades)
    for venue, trades in grouped.items():
        assert all(t.venue == venue for t in trades)
