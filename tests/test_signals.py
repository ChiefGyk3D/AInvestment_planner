"""Signals are alerts, and the exit-rule backtest must be brutally fair:
taxes and fees on every exit, no look-ahead, no free warm-up in cash."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ainvest.config import from_dict
from ainvest.signals import buy_and_hold, evaluate, portfolio_index, trend_overlay
from conftest import base_config


def series(values, freq="D"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def frame(**cols):
    n = len(next(iter(cols.values())))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=idx)


class TestPortfolioIndex:
    def test_starts_at_one_and_weights_the_returns(self):
        prices = frame(A=[100, 110, 121], B=[50, 50, 50])
        index = portfolio_index(prices, {"A": 0.5, "B": 0.5})
        assert index.iloc[0] == pytest.approx(1.0)
        # Half the money in a +10%/day asset, half flat: +5% a day.
        assert index.iloc[1] == pytest.approx(1.05)
        assert index.iloc[2] == pytest.approx(1.1025)

    def test_zero_weight_assets_are_ignored(self):
        prices = frame(A=[100, 200], B=[100, 1])
        index = portfolio_index(prices, {"A": 1.0, "B": 0.0})
        assert index.iloc[-1] == pytest.approx(2.0)

    def test_no_overlap_is_an_error(self):
        with pytest.raises(ValueError, match="no overlap"):
            portfolio_index(frame(A=[1, 2]), {"Z": 1.0})


class TestEvaluate:
    def test_drawdown_alert_fires_below_the_threshold(self):
        raw = base_config()
        raw["signals"] = {"trend_days": 0, "drawdown_alert": 0.20}
        cfg = from_dict(raw)
        path = list(np.linspace(100, 150, 50)) + list(np.linspace(150, 100, 50))
        state = evaluate(cfg, frame(VTI=path), {"VTI": 1.0})
        assert state.drawdown == pytest.approx(100 / 150 - 1, abs=1e-9)
        assert state.in_drawdown_alert
        assert any("contributions buy the most" in n for n in state.notes)

    def test_no_alert_in_a_rising_market(self):
        raw = base_config()
        raw["signals"] = {"trend_days": 0, "drawdown_alert": 0.20}
        cfg = from_dict(raw)
        state = evaluate(cfg, frame(VTI=np.linspace(100, 150, 80)), {"VTI": 1.0})
        assert not state.in_drawdown_alert
        assert state.drawdown == pytest.approx(0.0)

    def test_reserve_deployment_flag(self):
        raw = base_config()
        raw["signals"] = {"trend_days": 0}
        raw["contribution"] = {"reserve_fraction": 0.25, "deploy_on_drawdown": 0.10}
        cfg = from_dict(raw)
        path = list(np.linspace(100, 150, 50)) + list(np.linspace(150, 130, 50))
        state = evaluate(cfg, frame(VTI=path), {"VTI": 1.0})
        assert state.deploy_reserve
        assert any("reserve" in n for n in state.notes)

    def test_trend_state_above_and_below(self):
        raw = base_config()
        raw["signals"] = {"trend_days": 20, "drawdown_alert": 0.99}
        cfg = from_dict(raw)
        rising = frame(VTI=np.linspace(100, 200, 120))
        assert evaluate(cfg, rising, {"VTI": 1.0}).above_trend
        path = list(np.linspace(100, 200, 100)) + list(np.linspace(200, 120, 40))
        state = evaluate(cfg, frame(VTI=path), {"VTI": 1.0})
        assert not state.above_trend
        assert any("backtest --exit-rule trend" in n for n in state.notes)

    def test_too_little_history_says_so(self):
        raw = base_config()
        raw["signals"] = {"trend_days": 200}
        cfg = from_dict(raw)
        state = evaluate(cfg, frame(VTI=np.linspace(100, 110, 50)), {"VTI": 1.0})
        assert any("not enough history" in n for n in state.notes)


def crash_recovery_index(n=600, trend_days=50):
    """Up, hard down, then recovery -- the shape exit rules dream about."""
    rng = np.random.default_rng(3)
    up = np.linspace(100, 160, 250)
    down = np.linspace(160, 90, 120)
    recover = np.linspace(90, 170, n - 370)
    path = np.concatenate([up, down, recover]) * (1 + rng.normal(0, 0.002, n))
    return series(path)


class TestTrendOverlay:
    def cfg(self, **tax):
        raw = base_config()
        raw["tax"] = {"short_term_rate": 0.24, "long_term_rate": 0.15, **tax}
        raw["signals"] = {"trend_days": 50}
        return from_dict(raw)

    def test_warm_up_stays_invested(self):
        """Before the moving average exists there is no signal, and no signal
        must mean 'hold', not 'hide in cash' -- an early version of this code
        spent the whole warm-up out of the market via a NaN comparison."""
        cfg = self.cfg()
        index = series(np.linspace(100, 200, 300))  # relentless rally
        result = trend_overlay(cfg, index, trend_days=100, cost_bps=0.0)
        assert result.days_in_market == pytest.approx(1.0)
        assert result.exits == 0

    def test_no_lookahead_in_the_signal(self):
        """Truncating the future must not change the past of the equity curve."""
        cfg = self.cfg()
        index = crash_recovery_index()
        full = trend_overlay(cfg, index, trend_days=50).equity
        cut = trend_overlay(cfg, index.iloc[:400], trend_days=50).equity
        assert np.allclose(full.iloc[:400].to_numpy(), cut.to_numpy(), rtol=1e-12)

    def test_exits_pay_fees_and_taxes(self):
        cfg = self.cfg()
        index = crash_recovery_index()
        result = trend_overlay(cfg, index, trend_days=50, cost_bps=25.0)
        assert result.exits >= 1
        assert result.fees_paid > 0
        assert result.days_in_market < 1.0

    def test_taxes_only_reduce_the_outcome(self):
        taxed = self.cfg()
        untaxed = self.cfg(short_term_rate=0.0, long_term_rate=0.0)
        index = crash_recovery_index()
        with_tax = trend_overlay(taxed, index, trend_days=50)
        without = trend_overlay(untaxed, index, trend_days=50)
        assert with_tax.after_liquidation_return <= without.after_liquidation_return + 1e-12

    def test_buy_and_hold_defers_a_single_long_term_bill(self):
        cfg = self.cfg()
        index = series(np.linspace(100, 200, 500))
        result = buy_and_hold(cfg, index)
        assert result.total_return == pytest.approx(1.0)
        assert result.final_liquidation_tax == pytest.approx(1.0 * 0.15)
        assert result.after_liquidation_return == pytest.approx(1.0 - 0.15)

    def test_a_short_backtest_is_refused(self):
        cfg = self.cfg()
        with pytest.raises(ValueError, match="not enough history"):
            trend_overlay(cfg, series(np.linspace(100, 110, 30)), trend_days=50)

    def test_comparison_is_fair_after_liquidation(self):
        """In a straight-up market the exit rule can only tie or lose: same
        exposure, plus any whipsaw costs. This is the honesty property the
        whole feature exists to demonstrate."""
        cfg = self.cfg()
        index = series(np.linspace(100, 220, 500))
        hold = buy_and_hold(cfg, index)
        trend = trend_overlay(cfg, index, trend_days=50)
        assert trend.after_liquidation_return <= hold.after_liquidation_return + 1e-9
