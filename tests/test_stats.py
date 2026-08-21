from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ainvest.stats import (
    align_prices,
    effective_positions,
    estimate_moments,
    ledoit_wolf,
    max_drawdown,
    periods_per_year,
    shrink_means,
)


def test_weekday_data_annualises_at_about_252():
    """The bug this guards: a weekday series has a median gap of one day, so
    median-based annualisation would claim 365 periods a year and overstate
    every return and volatility by 45%."""
    index = pd.bdate_range("2024-01-01", periods=504)
    assert periods_per_year(index) == pytest.approx(261, abs=8)


def test_daily_crypto_data_annualises_at_365():
    index = pd.date_range("2024-01-01", periods=730, freq="D")
    assert periods_per_year(index) == pytest.approx(365, abs=2)


def test_align_intersects_the_two_calendars():
    """Equities do not trade at the weekend; crypto does. Rows must be spans
    both assets actually experienced."""
    weekdays = pd.bdate_range("2024-01-01", periods=60)
    every_day = pd.date_range("2024-01-01", periods=84, freq="D")
    frame = pd.DataFrame(index=every_day)
    frame["VTI"] = pd.Series(np.linspace(100, 110, 60), index=weekdays)
    frame["BTC"] = pd.Series(np.linspace(50, 60, 84), index=every_day)
    aligned = align_prices(frame, min_observations=40)
    assert list(aligned.columns) == ["VTI", "BTC"]
    assert aligned.index.dayofweek.max() < 5
    assert not aligned.isna().any().any()


def test_align_drops_assets_without_enough_history():
    index = pd.bdate_range("2024-01-01", periods=100)
    frame = pd.DataFrame({"GOOD": np.linspace(1, 2, 100)}, index=index)
    frame["NEW"] = np.nan
    frame.loc[frame.index[-5:], "NEW"] = 1.0
    aligned = align_prices(frame, min_observations=60)
    assert list(aligned.columns) == ["GOOD"]
    assert aligned.attrs["dropped"] == ["NEW"]


def test_align_rejects_nonpositive_prices():
    index = pd.bdate_range("2024-01-01", periods=80)
    frame = pd.DataFrame({"A": np.linspace(1, 2, 80), "B": np.linspace(1, 2, 80)}, index=index)
    frame.loc[frame.index[10], "B"] = 0.0
    aligned = align_prices(frame, min_observations=60)
    assert len(aligned) == 79  # the zero-price row is gone, not carried through


def test_ledoit_wolf_conditions_a_singular_sample():
    """With fewer observations than assets the sample covariance is singular;
    shrinkage is what stops the optimiser dividing by ~zero."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(20, 30))
    sample = np.cov(returns, rowvar=False, ddof=1)
    shrunk, intensity = ledoit_wolf(returns)
    assert 0.0 < intensity <= 1.0
    assert np.linalg.matrix_rank(sample) < 30
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)  # positive definite
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)


def test_ledoit_wolf_barely_shrinks_a_long_sample():
    """Shrinkage should fade as evidence accumulates -- but only when the
    target is actually wrong. The target here is a scaled identity, so the
    data needs unequal variances and real correlation for that to be true;
    with iid returns the target *is* the truth and shrinking hard is correct.
    """
    rng = np.random.default_rng(1)
    loadings = np.array([[0.30, 0.02], [0.10, 0.25], [0.02, 0.05]])
    factors = rng.normal(0, 1, size=(8000, 2))
    returns = factors @ loadings.T + rng.normal(0, 0.01, size=(8000, 3))
    _, intensity = ledoit_wolf(returns)
    assert intensity < 0.05


def test_ledoit_wolf_shrinks_harder_on_less_data():
    """The whole point: less evidence, more shrinkage."""
    rng = np.random.default_rng(4)
    loadings = np.array([[0.30, 0.02], [0.10, 0.25], [0.02, 0.05]])

    def intensity_for(n_obs: int) -> float:
        factors = rng.normal(0, 1, size=(n_obs, 2))
        returns = factors @ loadings.T + rng.normal(0, 0.01, size=(n_obs, 3))
        return ledoit_wolf(returns)[1]

    assert intensity_for(40) > intensity_for(4000)


def test_ledoit_wolf_is_symmetric_and_matches_scale():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.02, size=(400, 5))
    shrunk, _ = ledoit_wolf(returns)
    assert np.allclose(shrunk, shrunk.T)
    sample = (lambda x: (x.T @ x) / len(x))(returns - returns.mean(axis=0))
    # Shrinking toward a scaled identity preserves total variance.
    assert np.trace(shrunk) == pytest.approx(np.trace(sample), rel=1e-9)


@pytest.mark.parametrize("intensity, expected", [(0.0, [0.1, 0.2, 0.3]), (1.0, [0.2, 0.2, 0.2])])
def test_mean_shrinkage_endpoints(intensity, expected):
    assert np.allclose(shrink_means(np.array([0.1, 0.2, 0.3]), intensity), expected)


def test_mean_shrinkage_preserves_the_average():
    mu = np.array([0.05, 0.30, -0.10])
    for intensity in (0.0, 0.25, 0.5, 1.0):
        assert shrink_means(mu, intensity).mean() == pytest.approx(mu.mean())


def test_risk_contributions_sum_to_one(moments):
    w = np.full(len(moments.symbols), 1.0 / len(moments.symbols))
    assert moments.risk_contributions(w).sum() == pytest.approx(1.0)


def test_correlation_matrix_is_well_formed(moments):
    corr = moments.correlation()
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert corr.min() >= -1.0 and corr.max() <= 1.0


def test_moments_recover_the_synthetic_profiles(prices):
    """The synthetic generator standardises its shocks, so realised moments
    should land close to the profiles it was given."""
    m = estimate_moments(prices, mean_shrinkage=0.0)
    vols = dict(zip(m.symbols, m.vol, strict=True))
    assert vols["VTI"] == pytest.approx(0.16, abs=0.03)
    assert vols["BTC"] == pytest.approx(0.65, abs=0.10)
    assert vols["BTC"] > vols["VTI"] * 2


def test_max_drawdown():
    assert max_drawdown(np.array([1.0, 1.2, 0.6, 0.9])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == pytest.approx(0.0)


def test_effective_positions_counts_real_diversification():
    assert effective_positions(np.array([0.25] * 4)) == pytest.approx(4.0)
    assert effective_positions(np.array([1.0, 0.0])) == pytest.approx(1.0)
    concentrated = effective_positions(np.array([0.9, 0.05, 0.05]))
    assert concentrated < 1.3  # three tickers, effectively one position
