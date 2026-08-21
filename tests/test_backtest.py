"""A backtest that peeks at the future is worse than no backtest, because it
looks like evidence. The look-ahead test below is the important one here."""

from __future__ import annotations

import numpy as np
import pytest

from ainvest.backtest import run_backtest
from ainvest.config import from_dict
from conftest import base_config


def test_backtest_runs_and_reports_benchmarks(cfg, prices):
    report = run_backtest(cfg, prices, lookback_days=180, rebalance_days=60)
    assert report.strategy.rebalances > 1
    labels = [r.label for r in report.benchmarks]
    assert "equal weight" in labels
    assert any("buy & hold" in label for label in labels)
    assert report.warnings


def test_no_lookahead(cfg, prices):
    """Rewrite every price after a cut-off date and the equity curve before
    that date must not move by a single basis point. If it does, the strategy
    is trading on information it could not have had."""
    cut = prices.index[int(len(prices) * 0.6)]
    tampered = prices.copy()
    future = tampered.index > cut
    # A violent, obviously-fake future: one asset moons, another collapses.
    tampered.loc[future, tampered.columns[0]] *= 4.0
    tampered.loc[future, tampered.columns[-1]] *= 0.1

    original = run_backtest(cfg, prices, lookback_days=150, rebalance_days=45).strategy.equity
    altered = run_backtest(cfg, tampered, lookback_days=150, rebalance_days=45).strategy.equity

    before = original.index <= cut
    assert before.sum() > 20, "not enough pre-cutoff history to make this meaningful"
    assert np.allclose(
        original[before].to_numpy(), altered[: before.sum()].to_numpy(), rtol=1e-9
    ), "the equity curve before the cut-off changed when only the future changed"


def test_results_are_deterministic(cfg, prices):
    a = run_backtest(cfg, prices, lookback_days=180, rebalance_days=60).strategy
    b = run_backtest(cfg, prices, lookback_days=180, rebalance_days=60).strategy
    assert a.total_return == pytest.approx(b.total_return)
    assert np.allclose(a.equity.to_numpy(), b.equity.to_numpy())


def test_equity_curve_starts_at_one_and_stays_positive(cfg, prices):
    equity = run_backtest(cfg, prices, lookback_days=180, rebalance_days=60).strategy.equity
    assert equity.iloc[0] == pytest.approx(1.0, rel=0.02)
    assert (equity > 0).all()
    assert equity.index.is_monotonic_increasing


def test_reported_metrics_agree_with_the_equity_curve(cfg, prices):
    result = run_backtest(cfg, prices, lookback_days=180, rebalance_days=60).strategy
    expected = result.equity.iloc[-1] / result.equity.iloc[0] - 1.0
    assert result.total_return == pytest.approx(expected)
    assert result.max_drawdown <= 0.0
    assert result.volatility > 0


def test_trading_costs_reduce_returns(prices):
    """A backtest that ignores fees flatters every strategy that trades."""
    free = base_config()
    free["venues"]["public"].update({"fee_bps": 0, "spread_bps": 0})
    free["venues"]["coinbase_advanced"].update({"fee_bps": 0, "spread_bps": 0})

    costly = base_config()
    costly["venues"]["public"].update({"fee_bps": 300})
    costly["venues"]["coinbase_advanced"].update({"fee_bps": 300})

    kwargs = {"lookback_days": 150, "rebalance_days": 30}
    a = run_backtest(from_dict(free), prices, **kwargs).strategy
    b = run_backtest(from_dict(costly), prices, **kwargs).strategy
    assert b.fees_paid > a.fees_paid
    assert b.total_return < a.total_return


def test_equal_weight_turns_over_less_than_a_fitted_strategy(cfg, prices):
    report = run_backtest(cfg, prices, lookback_days=150, rebalance_days=30)
    equal = next(r for r in report.benchmarks if r.label == "equal weight")
    assert equal.turnover > 0
    assert report.strategy.turnover > 0


def test_too_little_history_is_refused_rather_than_faked(cfg, prices):
    with pytest.raises(ValueError, match="aligned observations"):
        run_backtest(cfg, prices.iloc[:100], lookback_days=90, rebalance_days=20)


def test_lookback_is_shortened_with_a_warning_when_history_is_thin(cfg, prices):
    report = run_backtest(cfg, prices.iloc[:250], lookback_days=252, rebalance_days=30)
    assert any("shortened" in w for w in report.warnings)
