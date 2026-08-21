"""Turn a config plus estimated moments into a target allocation."""

from __future__ import annotations

import numpy as np

from .config import RISK_PROFILES, Config
from .models import Allocation
from .optimize.core import InfeasibleError, OptimizeResult, project, solve
from .optimize.objectives import build_objective
from .stats import Moments, effective_positions


def _diagnose(cfg: Config, symbols: list[str], hi: np.ndarray) -> str:
    """Explain *why* the caps cannot add up, and which knob fixes it.

    "bounds allow a total of 0.600" is true but useless. A small universe plus
    a tight per-asset cap is the usual cause, and the arithmetic that makes it
    infeasible is worth spelling out rather than leaving the reader to redo.
    """
    equity_reach = sum(
        h for s, h in zip(symbols, hi, strict=True) if cfg.asset(s).asset_class != "crypto"
    )
    crypto_reach = sum(
        h for s, h in zip(symbols, hi, strict=True) if cfg.asset(s).asset_class == "crypto"
    )
    equity_room = min(equity_reach, cfg.constraints.equity_max)
    crypto_room = min(crypto_reach, cfg.constraints.crypto_max)
    n_equity = sum(1 for s in symbols if cfg.asset(s).asset_class != "crypto")
    n_crypto = len(symbols) - n_equity

    lines = [
        f"the {len(symbols)} available assets cannot add up to a fully invested "
        f"portfolio: at most {equity_room:.0%} in equities and {crypto_room:.0%} in "
        f"crypto, or {equity_room + crypto_room:.0%} in total.",
        f"  {n_equity} equities capped at {cfg.constraints.single_max:.0%} each "
        f"reach {equity_reach:.0%}"
        + (f", and constraints.equity_max limits that to {equity_room:.0%}"
           if equity_room < equity_reach - 1e-9 else ""),
        f"  {n_crypto} crypto assets reach {crypto_reach:.0%}"
        + (f", and constraints.crypto_max limits that to {crypto_room:.0%}"
           if crypto_room < crypto_reach - 1e-9 else ""),
        "  fixes: raise constraints.single_max, raise the ceiling on whichever "
        "group is short, or add more assets to the universe.",
    ]
    if cfg.risk_profile in RISK_PROFILES:
        lines.append(
            f"  note: risk_profile '{cfg.risk_profile}' sets single_max="
            f"{RISK_PROFILES[cfg.risk_profile]['single_max']:.0%} and crypto_max="
            f"{RISK_PROFILES[cfg.risk_profile]['crypto_max']:.0%}, which override "
            "the defaults; a tighter profile needs a wider universe."
        )
    return "\n".join(lines)


def _bounds(cfg: Config, symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for s in symbols:
        asset = cfg.asset(s)
        lo.append(max(asset.min_weight, cfg.constraints.single_min))
        hi.append(min(asset.max_weight, cfg.constraints.single_max))
    lo_arr, hi_arr = np.array(lo, dtype=float), np.array(hi, dtype=float)
    if np.any(lo_arr > hi_arr):
        bad = [
            s for s, low, high in zip(symbols, lo_arr, hi_arr, strict=True) if low > high
        ]
        raise InfeasibleError(
            "these assets have a min_weight above the effective max "
            f"(constraints.single_max={cfg.constraints.single_max}): {', '.join(bad)}"
        )
    equity_room = min(
        sum(
            h for s, h in zip(symbols, hi_arr, strict=True)
            if cfg.asset(s).asset_class != "crypto"
        ),
        cfg.constraints.equity_max,
    )
    crypto_room = min(
        sum(
            h for s, h in zip(symbols, hi_arr, strict=True)
            if cfg.asset(s).asset_class == "crypto"
        ),
        cfg.constraints.crypto_max,
    )
    if equity_room + crypto_room < 1.0 - 1e-9:
        raise InfeasibleError(_diagnose(cfg, symbols, hi_arr))
    return lo_arr, hi_arr


def _buckets(cfg: Config, symbols: list[str]):
    equity_idx = [i for i, s in enumerate(symbols) if cfg.asset(s).asset_class != "crypto"]
    crypto_idx = [i for i, s in enumerate(symbols) if cfg.asset(s).asset_class == "crypto"]
    buckets, blo, bhi = [], [], []
    if equity_idx:
        buckets.append(tuple(equity_idx))
        blo.append(cfg.constraints.equity_min)
        bhi.append(cfg.constraints.equity_max)
    if crypto_idx:
        buckets.append(tuple(crypto_idx))
        blo.append(cfg.constraints.crypto_min)
        bhi.append(cfg.constraints.crypto_max)
    # With only one bucket present its bounds must still admit a total of 1.
    if len(buckets) == 1:
        blo[0], bhi[0] = 0.0, 1.0
    return buckets, blo, bhi


def _calibrate_target_vol(
    cfg: Config, moments: Moments, n: int, lo, hi, buckets, blo, bhi, finalize
) -> OptimizeResult:
    """Find the risk aversion whose *finished* portfolio sits at the target vol.

    Portfolio volatility falls monotonically as risk aversion rises, so a
    bisection on ``log(k)`` lands on the target. Volatility is measured on
    ``finalize(w)`` -- the weights after blending and pruning -- because those
    are the weights the caller actually gets; calibrating on the raw solver
    output would miss the target by however much the blend added. If even the
    most risk-averse portfolio is more volatile than the target, we return
    that one: the target is not reachable with this universe and these caps.
    """
    target = cfg.optimizer.target_vol
    opt = cfg.optimizer

    def run(k: float, restarts: int, warm=None) -> OptimizeResult:
        objective = build_objective("mean_variance", moments, risk_aversion=k)
        return solve(
            objective, n, lo, hi, buckets, blo, bhi,
            max_iter=opt.max_iter, tolerance=opt.tolerance,
            restarts=restarts, warm_start=warm,
        )

    def final_vol(result: OptimizeResult) -> float:
        return moments.portfolio_vol(finalize(result.weights))

    low_k, high_k = 1e-3, 1e4
    lowest = run(high_k, opt.restarts)
    if final_vol(lowest) >= target:
        return lowest  # target unreachable; hand back the least risky portfolio
    highest = run(low_k, opt.restarts)
    if final_vol(highest) <= target:
        return highest  # even the greediest portfolio is calm enough

    # Volatility falls monotonically as risk aversion rises, so bisect on
    # log(k). Each solve warm-starts from the last, which is close by, so the
    # inner solves converge in a handful of steps rather than from scratch.
    best = lowest
    warm = lowest.weights
    for _ in range(24):
        mid = float(np.sqrt(low_k * high_k))
        best = run(mid, 1, warm)
        warm = best.weights
        vol = final_vol(best)
        if abs(vol - target) < 1e-4:
            break
        if vol > target:
            low_k = mid
        else:
            high_k = mid
    return best


def optimize_weights(cfg: Config, moments: Moments) -> Allocation:
    """Run the configured objective and post-process the result."""
    symbols = list(moments.symbols)
    n = len(symbols)
    lo, hi = _bounds(cfg, symbols)
    buckets, blo, bhi = _buckets(cfg, symbols)
    opt = cfg.optimizer
    name = opt.objective

    def finalize(raw: np.ndarray) -> np.ndarray:
        """Everything that happens between the solver and the answer."""
        blended = _blend_equal(raw, opt.blend_equal_weight, lo, hi, buckets, blo, bhi)
        return _prune(cfg, blended, lo, hi, buckets, blo, bhi)

    if name == "equal_weight":
        raw = project(np.full(n, 1.0 / n), lo, hi, buckets, blo, bhi)
    elif name == "target_vol":
        raw = _calibrate_target_vol(
            cfg, moments, n, lo, hi, buckets, blo, bhi, finalize
        ).weights
    else:
        objective = build_objective(name, moments, risk_free=opt.risk_free_rate)
        raw = solve(
            objective, n, lo, hi, buckets, blo, bhi,
            max_iter=opt.max_iter, tolerance=opt.tolerance, restarts=opt.restarts,
        ).weights

    weights = finalize(raw)

    return Allocation(
        symbols=symbols,
        weights=[float(w) for w in weights],
        expected_return=moments.portfolio_return(weights),
        expected_vol=moments.portfolio_vol(weights),
        sharpe=_sharpe(moments, weights, opt.risk_free_rate),
        risk_contributions=[float(x) for x in moments.risk_contributions(weights)],
        diversification=effective_positions(weights),
        objective=name,
    )


def _sharpe(moments: Moments, w: np.ndarray, risk_free: float) -> float:
    vol = moments.portfolio_vol(w)
    if vol <= 0:
        return 0.0
    return (moments.portfolio_return(w) - risk_free) / vol


def _blend_equal(w, intensity, lo, hi, buckets, blo, bhi) -> np.ndarray:
    """Pull the optimised weights partway toward equal weight.

    Cheap insurance. The optimiser's answer is fitted to one particular
    sample; equal weight is fitted to nothing. Splitting the difference gives
    up a little in-sample optimality for a lot of out-of-sample robustness.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0:
        return w
    n = w.size
    blended = (1.0 - intensity) * w + intensity * np.full(n, 1.0 / n)
    return project(blended, lo, hi, buckets, blo, bhi)


def _prune(cfg: Config, w, lo, hi, buckets, blo, bhi) -> np.ndarray:
    """Remove dust positions and enforce ``max_positions``.

    A 0.4% weight of a $300 budget is a $1.20 order: pure fee drag and clutter.
    Assets with a genuine ``min_weight`` are never pruned, since dropping them
    would violate a constraint the user asked for.
    """
    c = cfg.constraints
    droppable = lo <= 1e-12
    hi = hi.copy()

    for _ in range(w.size):
        candidates = np.where(droppable & (hi > 0) & (w < c.include_threshold) & (w > 0))[0]
        if c.max_positions > 0:
            live = np.where(hi > 0)[0]
            if live.size > c.max_positions:
                ranked = live[np.argsort(-w[live])]
                extra = [i for i in ranked[c.max_positions:] if droppable[i]]
                candidates = np.unique(np.concatenate([candidates, np.array(extra, dtype=int)])) if extra else candidates
        if candidates.size == 0:
            break
        trial = hi.copy()
        trial[candidates] = 0.0
        try:
            w = project(w, lo, trial, buckets, blo, bhi)
        except InfeasibleError:
            break  # dropping these would break a bucket minimum; keep them
        hi = trial
    return w
