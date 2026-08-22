"""Tax-lot analysis: the numbers a real tax bill would care about."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from ainvest.config import from_dict
from ainvest.lots import LONG_TERM_DAYS, cost_of_selling, lot_views, review, sell_advice
from conftest import base_config

TODAY = date(2026, 8, 21)
PRICES = {"VTI": 100.0, "GLD": 200.0, "BTC": 50000.0, "ETH": 2500.0}


def config_with_lots(lots, **overrides):
    raw = base_config(**overrides)
    raw["holdings"] = lots
    return from_dict(raw)


def lot(symbol, quantity, cost_basis, days_ago, venue="public"):
    return {
        "symbol": symbol, "quantity": quantity, "venue": venue,
        "cost_basis": cost_basis,
        "acquired": (TODAY - timedelta(days=days_ago)).isoformat(),
    }


def test_a_loss_is_a_harvest_candidate_with_the_right_saving():
    # 2 VTI at $100 = $200 value, bought for $300: a $100 long-term loss.
    cfg = config_with_lots([lot("VTI", 2.0, 300.0, 400)])
    report = review(cfg, PRICES, TODAY)
    assert len(report.harvest) == 1
    c = report.harvest[0]
    assert c.symbol == "VTI"
    assert -c.view.gain == pytest.approx(100.0)
    assert c.tax_saving == pytest.approx(100.0 * cfg.tax.long_term_rate)


def test_short_term_losses_save_at_the_short_term_rate():
    cfg = config_with_lots([lot("VTI", 2.0, 300.0, 100)])
    c = review(cfg, PRICES, TODAY).harvest[0]
    assert c.tax_saving == pytest.approx(100.0 * cfg.tax.short_term_rate)


def test_small_losses_are_not_worth_harvesting():
    cfg = config_with_lots(
        [lot("VTI", 2.0, 210.0, 400)],  # a $10 loss
        tax={"harvest_min_loss": 25.0},
    )
    assert review(cfg, PRICES, TODAY).harvest == []


def test_a_recent_buy_of_the_same_symbol_flags_a_wash_sale():
    cfg = config_with_lots([
        lot("VTI", 2.0, 300.0, 400),   # the loss lot
        lot("VTI", 0.5, 55.0, 10),     # bought 10 days ago -> wash conflict
    ])
    c = next(c for c in review(cfg, PRICES, TODAY).harvest if c.view.gain < 0)
    assert "wash" in c.wash_risk
    assert "20 day(s)" in c.wash_risk  # 30-day window minus 10 elapsed


def test_a_buy_outside_the_window_is_no_conflict():
    cfg = config_with_lots([
        lot("VTI", 2.0, 300.0, 400),
        lot("VTI", 0.5, 55.0, 31),
    ])
    c = next(c for c in review(cfg, PRICES, TODAY).harvest if c.view.gain < 0)
    assert c.wash_risk == ""


def test_crypto_is_wash_exempt_by_default_and_flagged_honestly():
    cfg = config_with_lots([
        lot("BTC", 0.002, 150.0, 200, venue="coinbase_advanced"),  # $100 value, $50 loss
        lot("BTC", 0.001, 60.0, 5, venue="coinbase_advanced"),     # recent buy
    ])
    report = review(cfg, PRICES, TODAY)
    c = next(c for c in report.harvest if c.symbol == "BTC")
    assert not c.wash_applies
    assert c.wash_risk == ""  # the recent buy does not matter for exempt assets
    assert any("recurring target of legislation" in n for n in report.notes)


def test_crypto_wash_rule_can_be_turned_on():
    cfg = config_with_lots(
        [
            lot("BTC", 0.002, 150.0, 200, venue="coinbase_advanced"),
            lot("BTC", 0.001, 60.0, 5, venue="coinbase_advanced"),
        ],
        tax={"crypto_wash_sale": True},
    )
    c = next(c for c in review(cfg, PRICES, TODAY).harvest if c.symbol == "BTC")
    assert c.wash_applies
    assert "wash" in c.wash_risk


def test_gains_near_the_long_term_boundary_get_a_wait_warning():
    # Bought 350 days ago with a gain: 16 more days makes it long-term.
    cfg = config_with_lots([lot("VTI", 2.0, 150.0, 350)])
    report = review(cfg, PRICES, TODAY)
    assert any("makes this gain long-term" in w for w in report.warnings)
    view = lot_views(cfg, PRICES, TODAY)[0]
    advice = sell_advice(cfg, view)
    expected_saving = 50.0 * (cfg.tax.short_term_rate - cfg.tax.long_term_rate)
    assert f"${expected_saving:,.2f}" in advice.wait


def test_a_long_held_gain_gets_no_wait_warning():
    cfg = config_with_lots([lot("VTI", 2.0, 150.0, LONG_TERM_DAYS + 10)])
    assert review(cfg, PRICES, TODAY).warnings == []


def test_lots_without_basis_are_reported_not_guessed():
    raw = base_config()
    raw["holdings"] = [{"symbol": "GLD", "quantity": 1.0, "venue": "public"}]
    cfg = from_dict(raw)
    report = review(cfg, PRICES, TODAY)
    assert report.unknown_basis == ["GLD"]
    assert report.harvest == []


def test_zero_rates_short_circuit_the_whole_review():
    cfg = config_with_lots(
        [lot("VTI", 2.0, 300.0, 400)],
        tax={"short_term_rate": 0.0, "long_term_rate": 0.0},
    )
    report = review(cfg, PRICES, TODAY)
    assert report.harvest == []
    assert any("tax-sheltered" in n for n in report.notes)


def test_replacement_asset_is_surfaced():
    raw = base_config()
    raw["assets"][0]["tlh_replacement"] = "VOO"
    raw["holdings"] = [lot("VTI", 2.0, 300.0, 400)]
    cfg = from_dict(raw)
    c = review(cfg, PRICES, TODAY).harvest[0]
    assert c.replacement == "VOO"


class TestCostOfSelling:
    def test_selling_a_pure_loss_position_costs_no_tax(self):
        cfg = config_with_lots([lot("VTI", 2.0, 300.0, 400)])
        tax, notes = cost_of_selling(cfg, "VTI", 200.0, PRICES, TODAY)
        assert tax == 0.0
        assert notes == []

    def test_selling_a_gain_costs_its_rate(self):
        cfg = config_with_lots([lot("VTI", 2.0, 100.0, 400)])  # $100 LT gain
        tax, _ = cost_of_selling(cfg, "VTI", 200.0, PRICES, TODAY)
        assert tax == pytest.approx(100.0 * cfg.tax.long_term_rate)

    def test_partial_sales_realise_a_proportional_gain(self):
        cfg = config_with_lots([lot("VTI", 2.0, 100.0, 400)])
        tax, _ = cost_of_selling(cfg, "VTI", 100.0, PRICES, TODAY)  # half the lot
        assert tax == pytest.approx(50.0 * cfg.tax.long_term_rate)

    def test_hifo_sells_the_expensive_lot_first(self):
        """Highest cost first minimises the gain realised."""
        cfg = config_with_lots([
            lot("VTI", 1.0, 40.0, 400),   # $60 gain if sold
            lot("VTI", 1.0, 95.0, 400),   # $5 gain if sold  <- should go first
        ])
        tax, _ = cost_of_selling(cfg, "VTI", 100.0, PRICES, TODAY)
        assert tax == pytest.approx(5.0 * cfg.tax.long_term_rate)

    def test_unknown_basis_is_reported_as_a_note(self):
        raw = base_config()
        raw["holdings"] = [{"symbol": "VTI", "quantity": 1.0, "venue": "public"}]
        cfg = from_dict(raw)
        tax, notes = cost_of_selling(cfg, "VTI", 50.0, PRICES, TODAY)
        assert tax == 0.0
        assert any("no cost basis" in n for n in notes)


def test_bad_acquired_date_is_rejected_at_load():
    raw = base_config()
    raw["holdings"] = [{
        "symbol": "VTI", "quantity": 1.0, "venue": "public",
        "cost_basis": 100.0, "acquired": "sometime last year",
    }]
    with pytest.raises(Exception, match="ISO date"):
        from_dict(raw)
