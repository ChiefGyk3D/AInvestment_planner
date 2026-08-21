"""Return, risk and covariance estimation.

Everything downstream depends on two numbers per asset (expected return and
volatility) and one matrix (covariance). With two years of daily data and a
handful of assets those estimates are noisy, so the helpers here lean
deliberately toward shrinkage over precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
CALENDAR_DAYS = 365.0


def align_prices(frame: pd.DataFrame, min_observations: int = 60) -> pd.DataFrame:
    """Restrict a price panel to dates where every asset has a price.

    Crypto trades every day and equities do not. Intersecting the calendars
    keeps each row a like-for-like span for every asset: a Monday row spans
    Friday-to-Monday for the stock *and* the coin, so a weekend crypto move is
    attributed to the same interval on both sides rather than inflating one
    series' apparent volatility.
    """
    if frame.empty:
        raise ValueError("no price data at all")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.where(frame > 0)

    usable = [c for c in frame.columns if frame[c].notna().sum() >= min_observations]
    dropped = [c for c in frame.columns if c not in usable]
    if not usable:
        raise ValueError(
            f"no asset has at least {min_observations} price observations"
        )
    aligned = frame[usable].dropna(how="any")
    if len(aligned) < min_observations:
        raise ValueError(
            f"only {len(aligned)} dates are common to all of "
            f"{', '.join(usable)}; need {min_observations}. Shorten the "
            "universe or lengthen data.lookback_days."
        )
    aligned.attrs["dropped"] = dropped
    return aligned


def periods_per_year(index: pd.Index) -> float:
    """Infer the annualisation factor from the observed sampling density.

    Counting observations across the elapsed span, rather than taking the
    median gap between them, is what makes this correct for both calendars.
    Weekday-only equity data has a median gap of one day but only ~252
    observations a year; median-based annualisation would overstate every
    return and volatility by a factor of 365/252.
    """
    if len(index) < 3:
        return TRADING_DAYS
    days = np.asarray(index, dtype="datetime64[D]").astype(float)
    span = float(days[-1] - days[0])
    if span <= 0:
        return TRADING_DAYS
    per_day = (len(index) - 1) / span
    return float(np.clip(per_day * CALENDAR_DAYS, 12.0, CALENDAR_DAYS))


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna(how="any")


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward a scaled identity matrix.

    This is the Ledoit-Wolf (2004) well-conditioned estimator. It trades a
    little bias for a large drop in variance, which matters enormously here:
    an unshrunk sample covariance from ~500 daily observations across a dozen
    assets produces near-singular matrices, and an optimiser will happily pour
    the whole budget into whichever asset the noise happened to favour.

    Returns the shrunk covariance and the shrinkage intensity actually used.
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    if t < 2:
        raise ValueError("need at least 2 observations to estimate covariance")
    x = x - x.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t

    mu = float(np.trace(sample) / n)
    target = mu * np.eye(n)

    d2 = float(np.sum((sample - target) ** 2) / n)
    if d2 <= 0:
        return sample, 0.0

    # Average squared deviation of the per-observation outer products from the
    # sample covariance: the estimator's own noise level.
    b_bar2 = 0.0
    for row in x:
        outer = np.outer(row, row)
        b_bar2 += float(np.sum((outer - sample) ** 2))
    b_bar2 /= (t**2) * n

    b2 = min(b_bar2, d2)
    intensity = float(np.clip(b2 / d2, 0.0, 1.0))
    shrunk = intensity * target + (1.0 - intensity) * sample
    return shrunk, intensity


def shrink_means(mu: np.ndarray, intensity: float) -> np.ndarray:
    """Pull expected returns toward their cross-sectional average.

    Sample means are the least reliable input to mean-variance optimisation.
    At ``intensity=1`` every asset gets the same expected return, which makes
    the optimiser purely risk-driven.
    """
    mu = np.asarray(mu, dtype=float)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    grand = float(np.mean(mu))
    return (1.0 - intensity) * mu + intensity * grand


class Moments:
    """Annualised expected returns and covariance for one universe."""

    def __init__(
        self,
        symbols: list[str],
        mu: np.ndarray,
        cov: np.ndarray,
        periods: float,
        observations: int,
        shrinkage: float = 0.0,
    ) -> None:
        self.symbols = symbols
        self.mu = np.asarray(mu, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.periods = periods
        self.observations = observations
        self.shrinkage = shrinkage

    @property
    def vol(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    def correlation(self) -> np.ndarray:
        sd = self.vol
        safe = np.where(sd > 0, sd, 1.0)
        corr = self.cov / np.outer(safe, safe)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def portfolio_vol(self, w: np.ndarray) -> float:
        return float(np.sqrt(max(w @ self.cov @ w, 0.0)))

    def portfolio_return(self, w: np.ndarray) -> float:
        return float(w @ self.mu)

    def risk_contributions(self, w: np.ndarray) -> np.ndarray:
        """Each asset's share of total portfolio variance-based risk."""
        w = np.asarray(w, dtype=float)
        total = self.portfolio_vol(w)
        if total <= 0:
            return np.zeros_like(w)
        marginal = self.cov @ w
        return w * marginal / (total**2)


def estimate_moments(
    prices: pd.DataFrame,
    *,
    covariance_shrinkage: str = "ledoit_wolf",
    mean_shrinkage: float = 0.5,
) -> Moments:
    """Turn a clean price panel into annualised moments."""
    rets = simple_returns(prices)
    if rets.empty:
        raise ValueError("price panel produced no returns")
    ppy = periods_per_year(rets.index)
    values = rets.to_numpy(dtype=float)

    if covariance_shrinkage == "ledoit_wolf":
        cov, intensity = ledoit_wolf(values)
    elif covariance_shrinkage == "none":
        cov, intensity = np.cov(values, rowvar=False, ddof=1), 0.0
        cov = np.atleast_2d(cov)
    else:
        raise ValueError(
            f"unknown covariance_shrinkage {covariance_shrinkage!r}; "
            "expected 'ledoit_wolf' or 'none'"
        )

    mu = shrink_means(values.mean(axis=0) * ppy, mean_shrinkage)
    cov = cov * ppy
    cov = 0.5 * (cov + cov.T)  # kill any asymmetry from floating point
    return Moments(
        symbols=list(rets.columns),
        mu=mu,
        cov=cov,
        periods=ppy,
        observations=len(rets),
        shrinkage=intensity,
    )


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline of an equity curve, as a fraction."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks) - 1.0)


def effective_positions(weights: np.ndarray) -> float:
    """Inverse Herfindahl index: how many positions you *effectively* hold.

    Five names at 20% each scores 5.0; five names where one is 90% scores
    close to 1.2. A useful sanity check on a concentrated "optimal" answer.
    """
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w**2))
    return 0.0 if denom <= 0 else 1.0 / denom
