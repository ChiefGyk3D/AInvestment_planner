"""Walk-forward test of an objective.

The point is not to find the strategy with the best past. It is to check that
the *procedure* -- fit on a trailing window, hold, refit -- would not have
produced something wild, and to see how it compares to equal weight, which is
the benchmark worth beating. Every weight is fitted only on data available
before the period it is applied to, and turnover is charged at the configured
venue costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .allocator import optimize_weights
from .config import Config
from .stats import align_prices, estimate_moments, max_drawdown, periods_per_year


@dataclass
class BacktestResult:
    label: str
    equity: pd.Series
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    fees_paid: float = 0.0
    rebalances: int = 0
    weight_history: list[tuple[pd.Timestamp, dict[str, float]]] = field(default_factory=list)


@dataclass
class BacktestReport:
    strategy: BacktestResult
    benchmarks: list[BacktestResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def all_results(self) -> list[BacktestResult]:
        return [self.strategy, *self.benchmarks]


def _summarise(
    label: str,
    equity: pd.Series,
    risk_free: float,
    turnover: float = 0.0,
    fees: float = 0.0,
    rebalances: int = 0,
    weights: list | None = None,
) -> BacktestResult:
    ppy = periods_per_year(equity.index)
    rets = equity.pct_change().dropna()
    years = max(len(rets) / ppy, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    vol = float(rets.std(ddof=1) * np.sqrt(ppy)) if len(rets) > 1 else 0.0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    return BacktestResult(
        label=label,
        equity=equity,
        total_return=total,
        cagr=cagr,
        volatility=vol,
        sharpe=(cagr - risk_free) / vol if vol > 0 else 0.0,
        max_drawdown=max_drawdown(equity.to_numpy()),
        turnover=turnover,
        fees_paid=fees,
        rebalances=rebalances,
        weight_history=weights or [],
    )


def _average_cost_bps(cfg: Config, symbols: list[str]) -> float:
    """Blended one-way trading cost across the universe, in basis points."""
    costs = []
    for symbol in symbols:
        try:
            venue = cfg.cheapest_venue(cfg.asset(symbol), 100.0)
        except Exception:  # noqa: BLE001 - unknown symbol falls back to zero cost
            continue
        costs.append(venue.fee_bps + venue.spread_bps)
    return float(np.mean(costs)) if costs else 0.0


def _run_weights(
    prices: pd.DataFrame,
    weight_fn,
    step: int,
    lookback: int,
    cost_bps: float,
) -> tuple[pd.Series, float, float, int, list]:
    """Chain returns through a rebalanced portfolio, charging turnover costs."""
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index
    equity = [1.0]
    equity_dates = [dates[lookback]]
    current = np.zeros(prices.shape[1])
    turnover_total = 0.0
    fees_total = 0.0
    rebalances = 0
    history: list = []

    i = lookback
    while i < len(dates):
        window = prices.iloc[i - lookback : i]
        target = weight_fn(window)
        turnover = float(np.sum(np.abs(target - current)))
        cost = turnover * cost_bps / 10_000.0
        turnover_total += turnover
        rebalances += 1
        history.append((dates[i], dict(zip(prices.columns, target, strict=True))))
        equity[-1] *= 1.0 - cost
        fees_total += cost
        current = target.copy()

        end = min(i + step, len(dates))
        for j in range(i, end):
            period = float(np.dot(current, rets.iloc[j].to_numpy()))
            equity.append(equity[-1] * (1.0 + period))
            equity_dates.append(dates[j])
            # Positions drift with prices between rebalances.
            grown = current * (1.0 + rets.iloc[j].to_numpy())
            total = grown.sum()
            current = grown / total if total > 0 else current
        i = end

    series = pd.Series(equity, index=pd.DatetimeIndex(equity_dates))
    series = series[~series.index.duplicated(keep="last")]
    return series, turnover_total, fees_total, rebalances, history


def run_backtest(
    cfg: Config,
    prices: pd.DataFrame,
    *,
    lookback_days: int = 252,
    rebalance_days: int = 63,
    include_benchmarks: bool = True,
) -> BacktestReport:
    """Walk the configured objective forward through the price history."""
    aligned = align_prices(prices, cfg.data.min_observations)
    n_dates = len(aligned)
    warnings: list[str] = []

    # Half the history fits the weights, half tests them. Any less out-of-sample
    # and the "walk forward" is mostly just the fitting window looked at twice.
    required = 2 * cfg.data.min_observations
    if n_dates < required:
        raise ValueError(
            f"only {n_dates} aligned observations; a walk-forward test needs at "
            f"least {required} so that half the history can be held out. Lengthen "
            "data.lookback_days, or lower data.min_observations if you accept a "
            "noisier estimate."
        )
    lookback = min(lookback_days, n_dates // 2)
    if lookback != lookback_days:
        warnings.append(
            f"lookback shortened from {lookback_days} to {lookback} rows so that "
            f"half of the {n_dates} available observations stay out of sample"
        )

    periods = min(rebalance_days, max(1, (n_dates - lookback) // 2))
    cost_bps = _average_cost_bps(cfg, list(aligned.columns))
    opt = cfg.optimizer

    def optimised(window: pd.DataFrame) -> np.ndarray:
        moments = estimate_moments(
            window,
            covariance_shrinkage=opt.covariance_shrinkage,
            mean_shrinkage=opt.mean_shrinkage,
        )
        allocation = optimize_weights(cfg, moments)
        return np.array([allocation.as_dict().get(c, 0.0) for c in window.columns])

    equity, turnover, fees, count, history = _run_weights(
        aligned, optimised, periods, lookback, cost_bps
    )
    strategy = _summarise(
        f"{opt.objective} (walk-forward)", equity, opt.risk_free_rate,
        turnover, fees, count, history,
    )

    benchmarks: list[BacktestResult] = []
    if include_benchmarks:
        n = aligned.shape[1]

        def equal(_window: pd.DataFrame) -> np.ndarray:
            return np.full(n, 1.0 / n)

        eq_equity, eq_turn, eq_fees, eq_count, _ = _run_weights(
            aligned, equal, periods, lookback, cost_bps
        )
        benchmarks.append(
            _summarise("equal weight", eq_equity, opt.risk_free_rate, eq_turn, eq_fees, eq_count)
        )

        for column in aligned.columns:
            if cfg.asset(column).asset_class == "equity":
                held = aligned[column].iloc[lookback:]
                benchmarks.append(
                    _summarise(f"buy & hold {column}", held / held.iloc[0], opt.risk_free_rate)
                )
                break

    warnings.append(
        "past performance says nothing about future returns, and a backtest over "
        "one particular stretch of history says even less. Read this as a sanity "
        "check on the method, not as a forecast."
    )
    return BacktestReport(strategy=strategy, benchmarks=benchmarks, warnings=warnings)
