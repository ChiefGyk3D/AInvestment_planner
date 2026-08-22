"""Long-horizon projection by block bootstrap of historical returns.

The point of a projection is expectation-setting, not prediction: what does a
plan of "start with $X, add $Y a month" plausibly look like in five or ten
years, and how wide is 'plausibly'? A range forces the honest framing that a
single compound-growth number hides.

Method: resample the portfolio's own historical returns in multi-week blocks
(blocks preserve some of the volatility clustering that daily i.i.d.
resampling destroys), simulate thousands of paths with the contribution
schedule applied, and report percentiles. Every caveat about the past not
predicting the future applies twice here, because the resampling cannot
produce anything history did not contain -- a regime worse than the sample
window simply is not in the deck being shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .signals import portfolio_index
from .stats import periods_per_year

PERCENTILES = (10, 25, 50, 75, 90)


@dataclass
class ProjectionResult:
    years: float
    start: float
    contribution_per_year: float
    total_contributed: float
    #: percentile -> final wealth
    wealth: dict[int, float] = field(default_factory=dict)
    probability_of_loss: float = 0.0
    #: median max drawdown experienced along the way
    median_max_drawdown: float = 0.0
    paths: int = 0
    label: str = ""


@dataclass
class ProjectionReport:
    results: list[ProjectionResult] = field(default_factory=list)
    dca_comparison: list[tuple[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _block_bootstrap_paths(
    returns: np.ndarray,
    n_paths: int,
    n_periods: int,
    block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_paths, n_periods) resampled return paths."""
    n = len(returns)
    if n <= block:
        raise ValueError("not enough history to bootstrap from")
    blocks_needed = int(np.ceil(n_periods / block))
    starts = rng.integers(0, n - block, size=(n_paths, blocks_needed))
    # Gather blocks: for each start, take `block` consecutive returns.
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
    return returns[idx[:, :n_periods]]


def _simulate(
    paths: np.ndarray,
    start: float,
    contribution: float,
    contribute_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wealth paths with periodic contributions; returns (final, max_dd)."""
    n_paths, n_periods = paths.shape
    wealth = np.full(n_paths, float(start))
    peak = wealth.copy()
    worst = np.zeros(n_paths)
    for t in range(n_periods):
        if contribution > 0 and contribute_every > 0 and t % contribute_every == 0 and t > 0:
            wealth += contribution
        wealth *= 1.0 + paths[:, t]
        peak = np.maximum(peak, wealth)
        worst = np.minimum(worst, wealth / peak - 1.0)
    return wealth, worst


def run_projection(
    cfg: Config,
    prices: pd.DataFrame,
    weights: dict[str, float],
    *,
    years: float = 10.0,
    n_paths: int = 3000,
    seed: int = 7,
    dca_months: int = 0,
) -> ProjectionReport:
    """Project the configured plan (budget now + contributions) forward."""
    index = portfolio_index(prices, weights)
    rets = index.pct_change().dropna().to_numpy()
    ppy = periods_per_year(index.index)
    n_periods = int(round(years * ppy))
    block = max(5, int(ppy / 12))  # ~ month-long blocks

    per_year = {"monthly": 12, "biweekly": 26, "weekly": 52}[cfg.contribution.frequency]
    contribution = cfg.contribution.amount
    contribute_every = max(1, int(round(ppy / per_year))) if contribution > 0 else 0

    rng = np.random.default_rng(seed)
    paths = _block_bootstrap_paths(rets, n_paths, n_periods, block, rng)

    report = ProjectionReport()
    start = cfg.investable

    for horizon in sorted({1.0, min(5.0, years), years}):
        n_h = int(round(horizon * ppy))
        if n_h < 2 or n_h > n_periods:
            continue
        final, worst = _simulate(
            paths[:, :n_h], start, contribution, contribute_every
        )
        contributed = start + contribution * per_year * horizon
        report.results.append(
            ProjectionResult(
                years=horizon,
                start=start,
                contribution_per_year=contribution * per_year,
                total_contributed=contributed,
                wealth={p: float(np.percentile(final, p)) for p in PERCENTILES},
                probability_of_loss=float(np.mean(final < contributed)),
                median_max_drawdown=float(np.median(worst)),
                paths=n_paths,
                label=f"{horizon:g}y",
            )
        )

    if dca_months > 0:
        # Same total money: all now, versus spread over the first N months.
        n_h = n_periods
        lump, _ = _simulate(paths[:, :n_h], start, 0.0, 0)
        tranche = start / dca_months
        step = max(1, int(round(ppy / 12)))
        # The cash schedule is deterministic (it never depends on returns),
        # so it is a scalar; only the invested side varies by path.
        cash = float(start)
        invested = np.zeros(paths.shape[0])
        for t in range(n_h):
            if t % step == 0 and cash > 1e-9:
                move = min(cash, tranche)
                invested += move
                cash -= move
            invested *= 1.0 + paths[:, t]
        dca_final = invested + cash
        report.dca_comparison = [
            ("lump sum now (median)", float(np.median(lump))),
            (f"spread over {dca_months} months (median)", float(np.median(dca_final))),
            ("lump sum wins in", float(np.mean(lump > dca_final))),
        ]

    sample_years = len(rets) / ppy
    report.warnings.append(
        f"paths are reshuffled from only {sample_years:.1f} years of history; "
        "outcomes outside what that window contained (a deeper crash, a longer "
        "stagnation) are invisible to this method. The bands are a plausible "
        "range, not the full range."
    )
    return report
