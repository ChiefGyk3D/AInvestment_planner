"""Rebalancing must not quietly do the expensive thing: selling costs fees and,
in a taxable account, tax. The default path is buy-only."""

from __future__ import annotations

import pytest

from ainvest.allocator import optimize_weights
from ainvest.config import from_dict
from ainvest.models import Allocation
from ainvest.rebalance import holdings_value, plan_rebalance
from conftest import base_config

PRICES = {"VTI": 100.0, "GLD": 200.0, "BTC": 50000.0, "ETH": 2500.0}
EVEN = Allocation(
    symbols=["VTI", "GLD", "BTC", "ETH"], weights=[0.4, 0.35, 0.15, 0.10]
)


def config_with(holdings, **overrides):
    raw = base_config(**overrides)
    raw["holdings"] = holdings
    return from_dict(raw)


def test_holdings_are_valued_at_the_latest_price():
    cfg = config_with([{"symbol": "VTI", "quantity": 2.5, "venue": "public"}])
    total, values = holdings_value(cfg.holdings, PRICES)
    assert total == pytest.approx(250.0)
    assert values == {"VTI": 250.0}


def test_duplicate_holdings_of_one_symbol_are_summed():
    cfg = config_with([
        {"symbol": "VTI", "quantity": 1.0, "venue": "public"},
        {"symbol": "VTI", "quantity": 0.5, "venue": "public"},
    ])
    total, _ = holdings_value(cfg.holdings, PRICES)
    assert total == pytest.approx(150.0)


def test_cash_flow_mode_never_sells():
    """The default must be buy-only, however far a position has drifted."""
    cfg = config_with([{"symbol": "VTI", "quantity": 10.0, "venue": "public"}])  # 100% VTI
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=500)
    assert report.plan is not None
    assert all(t.side == "buy" for t in report.plan.trades)
    assert not any(t.symbol == "VTI" for t in report.plan.trades)
    assert any("left alone" in note for note in report.notes)


def test_cash_goes_to_the_most_underweight_positions():
    cfg = config_with([{"symbol": "VTI", "quantity": 10.0, "venue": "public"}])
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=500)
    bought = {t.symbol for t in report.plan.trades}
    assert "GLD" in bought  # the largest shortfall
    assert "VTI" not in bought


def test_new_cash_is_not_overspent():
    cfg = config_with([{"symbol": "VTI", "quantity": 4.0, "venue": "public"}])
    for cash in (25, 100, 500, 2000):
        report = plan_rebalance(cfg, EVEN, PRICES, new_cash=cash)
        spent = sum(t.notional + t.fee for t in report.plan.trades)
        assert spent <= cash + 1e-6, cash


def test_drift_is_measured_against_the_post_contribution_total():
    cfg = config_with([{"symbol": "VTI", "quantity": 3.0, "venue": "public"}])  # $300
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=100)
    assert report.total_value == pytest.approx(400.0)
    vti = next(d for d in report.drifts if d.symbol == "VTI")
    assert vti.current_weight == pytest.approx(0.75)
    assert vti.drift == pytest.approx(0.35)


def test_an_on_target_portfolio_is_left_alone():
    cfg = config_with(
        [
            {"symbol": "VTI", "quantity": 4.0, "venue": "public"},     # $400 -> 40%
            {"symbol": "GLD", "quantity": 1.75, "venue": "public"},    # $350 -> 35%
            {"symbol": "BTC", "quantity": 0.003, "venue": "coinbase_advanced"},  # $150
            {"symbol": "ETH", "quantity": 0.04, "venue": "coinbase_advanced"},   # $100
        ]
    )
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=0)
    assert report.max_drift < 1e-6
    assert report.plan.trades == []
    assert any("no-trade band" in note for note in report.notes)


def test_trades_below_the_minimum_are_skipped_with_a_reason():
    cfg = config_with(
        [{"symbol": "VTI", "quantity": 10.0, "venue": "public"}],
        min_trade=50.0,
    )
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=60)
    assert report.skipped
    assert all("minimum" in reason for _, reason in report.skipped)


def test_selling_is_opt_in_and_two_sided():
    cfg = config_with(
        [{"symbol": "VTI", "quantity": 10.0, "venue": "public"}],
        allow_sells=True,
    )
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=0, allow_sells=True)
    sides = {t.side for t in report.plan.trades}
    assert "sell" in sides
    assert any("capital gains" in note for note in report.notes)


def test_sells_never_exceed_what_is_held():
    cfg = config_with(
        [{"symbol": "VTI", "quantity": 10.0, "venue": "public"}],
        allow_sells=True,
    )
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=0, allow_sells=True)
    for trade in report.plan.trades:
        if trade.side == "sell":
            assert trade.quantity <= 10.0 + 1e-9
            assert trade.notional <= 1000.0 + 1e-6


def test_a_trade_whose_fee_swamps_it_is_skipped():
    """A small correction is not worth paying a large flat fee to make.

    Holdings here sit close to target, so the only drift worth trading is a
    $30 slice of VTI -- against a $20 flat fee. Making that trade would cost
    two thirds of the amount being corrected.
    """
    raw = base_config(allow_sells=True, min_trade=0.01, rebalance_band=0.01)
    raw["venues"]["public"]["fixed_fee"] = 20.0
    raw["venues"]["public"]["min_notional"] = 0.01
    raw["holdings"] = [
        {"symbol": "VTI", "quantity": 4.5, "venue": "public"},              # $450
        {"symbol": "GLD", "quantity": 1.75, "venue": "public"},             # $350
        {"symbol": "BTC", "quantity": 0.003, "venue": "coinbase_advanced"}, # $150
        {"symbol": "ETH", "quantity": 0.04, "venue": "coinbase_advanced"},  # $100
    ]
    cfg = from_dict(raw)
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=0, allow_sells=True)
    assert any("quarter of" in reason for _, reason in report.skipped), report.skipped
    assert not any(t.symbol == "VTI" for t in report.plan.trades)


def test_an_empty_portfolio_with_no_cash_is_reported():
    cfg = config_with([])
    report = plan_rebalance(cfg, EVEN, PRICES, new_cash=0)
    assert report.plan is None
    assert any("empty" in note for note in report.notes)


def test_rebalance_uses_the_optimised_target(cfg, moments, latest):
    """End to end: optimise, then plan the trades that get you there."""
    raw = base_config()
    raw["holdings"] = [{"symbol": "VTI", "quantity": 1.0, "venue": "public"}]
    live = from_dict(raw)
    allocation = optimize_weights(live, moments)
    report = plan_rebalance(live, allocation, latest, new_cash=400)
    assert report.plan is not None
    assert report.plan.trades
    assert all(t.side == "buy" for t in report.plan.trades)
