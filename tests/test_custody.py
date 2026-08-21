from __future__ import annotations

import pytest

from ainvest.config import from_dict
from ainvest.custody import review
from ainvest.models import Allocation, Plan, Trade
from conftest import base_config


def plan_with(**values) -> Plan:
    """A plan holding the given dollar amounts, all on coinbase_advanced."""
    trades = [
        Trade(symbol=s, side="buy", venue="coinbase_advanced", notional=v,
              quantity=v / 100.0, price=100.0, fee=0.0)
        for s, v in values.items()
    ]
    allocation = Allocation(symbols=list(values), weights=[1.0 / len(values)] * len(values))
    return Plan(allocation=allocation, trades=trades, deployed=sum(values.values()))


def test_small_positions_are_told_to_wait():
    cfg = from_dict(base_config(custody={"sweep_threshold": 100.0}))
    report = review(cfg, plan_with(BTC=40.0))
    action = next(a for a in report.actions if a.symbol == "BTC")
    assert action.recommendation == "wait"
    assert "sweep threshold" in action.reason


def test_large_positions_are_told_to_sweep():
    cfg = from_dict(base_config(custody={"sweep_threshold": 100.0}))
    report = review(cfg, plan_with(BTC=500.0))
    action = next(a for a in report.actions if a.symbol == "BTC")
    assert action.recommendation == "sweep"
    assert action.fee_fraction == pytest.approx(1.5 / 500.0)
    assert report.sweep_total == pytest.approx(500.0)


def test_a_withdrawal_fee_that_is_too_large_a_share_blocks_the_sweep():
    """$150 above the threshold is still not worth a fee costing 20% of it."""
    raw = base_config(custody={"sweep_threshold": 100.0, "max_fee_fraction": 0.02})
    raw["venues"]["coinbase_advanced"]["withdrawal_fee"] = 30.0
    cfg = from_dict(raw)
    action = next(a for a in review(cfg, plan_with(BTC=150.0)).actions if a.symbol == "BTC")
    assert action.recommendation == "wait"
    assert "network fee" in action.reason


def test_equities_are_not_part_of_the_custody_review():
    cfg = from_dict(base_config())
    report = review(cfg, plan_with(VTI=1000.0, BTC=500.0))
    assert [a.symbol for a in report.actions] == ["BTC"]


def test_a_venue_without_withdrawals_cannot_self_custody():
    """Some retail venues sell price exposure, not the coin itself."""
    raw = base_config()
    raw["venues"]["coinbase_advanced"]["supports_withdrawal"] = False
    cfg = from_dict(raw)
    action = next(a for a in review(cfg, plan_with(BTC=5000.0)).actions if a.symbol == "BTC")
    assert action.recommendation == "leave"
    assert "not supporting withdrawals" in action.reason


def test_assets_not_marked_self_custody_are_left_alone():
    raw = base_config()
    for asset in raw["assets"]:
        if asset["symbol"] == "BTC":
            asset["self_custody"] = False
    cfg = from_dict(raw)
    action = next(a for a in review(cfg, plan_with(BTC=5000.0)).actions if a.symbol == "BTC")
    assert action.recommendation == "leave"


def test_a_sweep_recommends_verifying_the_address_on_the_device():
    cfg = from_dict(base_config(custody={"sweep_threshold": 10.0}))
    report = review(cfg, plan_with(BTC=500.0))
    joined = " ".join(report.notes)
    assert "test transaction" in joined
    assert "hardware wallet screen" in joined


def test_existing_holdings_count_toward_the_sweep_threshold():
    """A new $40 buy on top of $200 already held is worth withdrawing."""
    cfg = from_dict(base_config(custody={"sweep_threshold": 100.0}))
    report = review(cfg, plan_with(BTC=40.0), extra_values={"BTC": 200.0})
    action = next(a for a in report.actions if a.symbol == "BTC")
    assert action.value == pytest.approx(240.0)
    assert action.recommendation == "sweep"


def test_the_review_can_be_switched_off():
    cfg = from_dict(base_config(custody={"enabled": False}))
    report = review(cfg, plan_with(BTC=5000.0))
    assert report.actions == []
    assert any("disabled" in note for note in report.notes)


def test_exchange_exposure_is_called_out():
    cfg = from_dict(base_config(custody={"sweep_threshold": 1000.0}))
    report = review(cfg, plan_with(BTC=100.0, ETH=50.0))
    assert any("stays on an exchange" in note for note in report.notes)
