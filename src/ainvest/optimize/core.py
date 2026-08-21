"""Projection onto the feasible weight set, and a projected-gradient solver.

The feasible set is

    { w : sum(w) = 1, lo_i <= w_i <= hi_i, bl_k <= sum_{i in bucket k} w_i <= bh_k }

with disjoint buckets (an asset is equity *or* crypto, never both). That set is
convex and its Euclidean projection has a closed form, which ``project``
computes exactly -- no iteration, no tolerance. Having an exact projection
means the optimiser never trades off "satisfy the constraints" against
"improve the objective": every iterate it ever sees is already feasible.

The closed form comes from the KKT conditions. For a plain box-and-sum
constraint the solution is ``w_i = clip(y_i - theta, lo_i, hi_i)`` for the
scalar ``theta`` that makes the weights sum to the target. Adding a bucket
constraint adds one multiplier per bucket, which is zero unless that bucket's
own bound binds -- and when it does bind, the bucket's weights are simply its
projection at that bound, independent of ``theta``. So the whole problem
reduces to finding one scalar, and both ``sum(clip(y - theta))`` and the
bucket-clamped total are piecewise-linear in ``theta``, so that scalar is
found by locating the right linear segment and solving on it directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

Bucket = Sequence[int]


class InfeasibleError(ValueError):
    """The constraints as written cannot all hold at once."""


def _solve_theta(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, target: float
) -> float:
    """Exact ``theta`` with ``sum(clip(y - theta, lo, hi)) == target``.

    ``s(theta)`` is non-increasing and piecewise linear, bending only where
    some element enters or leaves its bounds -- that is, at ``y_i - lo_i`` and
    ``y_i - hi_i``. Evaluating ``s`` at those breakpoints identifies the
    segment containing the target; the answer follows by linear interpolation
    along it.
    """
    knots = np.unique(np.concatenate([y - hi, y - lo]))
    # s at every knot at once: rows are knots, columns are assets.
    values = np.clip(y[None, :] - knots[:, None], lo, hi).sum(axis=1)

    if target >= values[0]:
        return float(knots[0])  # saturated at the upper bounds
    if target <= values[-1]:
        return float(knots[-1])  # saturated at the lower bounds

    # values is non-increasing; searchsorted wants increasing, so flip it.
    j = len(knots) - 1 - int(np.searchsorted(values[::-1], target, side="left"))
    j = int(np.clip(j, 0, len(knots) - 2))
    lo_v, hi_v = values[j], values[j + 1]
    if lo_v == hi_v:
        return float(knots[j])
    slope = (hi_v - lo_v) / (knots[j + 1] - knots[j])
    return float(knots[j] + (target - lo_v) / slope)


def project(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    buckets: Sequence[Bucket] = (),
    bucket_lo: Sequence[float] = (),
    bucket_hi: Sequence[float] = (),
    total: float = 1.0,
) -> np.ndarray:
    """Euclidean projection of ``y`` onto the feasible weight set."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = y.size
    if lo.shape != y.shape or hi.shape != y.shape:
        raise ValueError("lo and hi must match the shape of y")
    if np.any(lo > hi + 1e-12):
        raise InfeasibleError("an asset has min_weight above max_weight")

    explicit_buckets = bool(buckets)
    if not buckets:
        # One implicit bucket spanning everything, so the code below has a
        # single path. Error messages still distinguish the two cases, because
        # "your bucket bounds are wrong" is nonsense to someone who never
        # configured a bucket.
        buckets = [tuple(range(n))]
        bucket_lo = [0.0]
        bucket_hi = [float(total)]
    if len(buckets) != len(bucket_lo) or len(buckets) != len(bucket_hi):
        raise ValueError("buckets, bucket_lo and bucket_hi must be the same length")

    covered = sorted(i for b in buckets for i in b)
    if covered != list(range(n)):
        raise ValueError("buckets must be disjoint and cover every asset exactly once")

    idx_arrays = [np.asarray(b, dtype=int) for b in buckets]

    # Tighten each bucket's declared bounds by what its own assets can reach.
    eff_lo: list[float] = []
    eff_hi: list[float] = []
    for idx, bl, bh in zip(idx_arrays, bucket_lo, bucket_hi, strict=True):
        reach_lo, reach_hi = float(lo[idx].sum()), float(hi[idx].sum())
        low = max(float(bl), reach_lo)
        high = min(float(bh), reach_hi)
        if low > high + 1e-9:
            if explicit_buckets:
                raise InfeasibleError(
                    f"a group's bounds of {bl:.3f}..{bh:.3f} cannot be met by its "
                    f"assets, whose own limits only allow {reach_lo:.3f}..{reach_hi:.3f}"
                )
            raise InfeasibleError(
                f"the per-asset bounds allow a total of {reach_lo:.3f}..{reach_hi:.3f}, "
                f"which does not contain the required total of {total:.3f}"
            )
        eff_lo.append(low)
        eff_hi.append(high)

    if sum(eff_lo) > total + 1e-9 or sum(eff_hi) < total - 1e-9:
        source = "the group and per-asset bounds" if explicit_buckets else "the per-asset bounds"
        raise InfeasibleError(
            f"{source} allow a total of {sum(eff_lo):.3f}..{sum(eff_hi):.3f}, "
            f"which does not contain the required total of {total:.3f}"
        )

    # The theta at which each bucket hits its own ceiling and its own floor.
    # Outside that range the bucket is pinned and stops responding to theta.
    theta_hi = [
        _solve_theta(y[idx], lo[idx], hi[idx], h)
        for idx, h in zip(idx_arrays, eff_hi, strict=True)
    ]
    theta_lo = [
        _solve_theta(y[idx], lo[idx], hi[idx], low)
        for idx, low in zip(idx_arrays, eff_lo, strict=True)
    ]

    def bucket_weights(k: int, theta: float) -> np.ndarray:
        idx = idx_arrays[k]
        if theta <= theta_hi[k]:
            return np.clip(y[idx] - theta_hi[k], lo[idx], hi[idx])
        if theta >= theta_lo[k]:
            return np.clip(y[idx] - theta_lo[k], lo[idx], hi[idx])
        return np.clip(y[idx] - theta, lo[idx], hi[idx])

    def total_at(theta: float) -> float:
        return sum(float(bucket_weights(k, theta).sum()) for k in range(len(idx_arrays)))

    # The clamped total is piecewise linear too, with the per-asset knots plus
    # the thetas at which a bucket pins. Find the segment, then interpolate.
    knots = np.unique(
        np.concatenate([y - hi, y - lo, np.array(theta_hi), np.array(theta_lo)])
    )
    values = np.array([total_at(k) for k in knots])

    if total >= values[0]:
        theta = float(knots[0])
    elif total <= values[-1]:
        theta = float(knots[-1])
    else:
        j = len(knots) - 1 - int(np.searchsorted(values[::-1], total, side="left"))
        j = int(np.clip(j, 0, len(knots) - 2))
        lo_v, hi_v = values[j], values[j + 1]
        if lo_v == hi_v:
            theta = float(knots[j])
        else:
            slope = (hi_v - lo_v) / (knots[j + 1] - knots[j])
            theta = float(knots[j] + (total - lo_v) / slope)

    w = np.empty(n, dtype=float)
    for k, idx in enumerate(idx_arrays):
        w[idx] = bucket_weights(k, theta)

    # Absorb any last floating-point residual into whichever slot has room.
    residual = total - float(w.sum())
    if abs(residual) > 1e-12:
        slack = (hi - w) if residual > 0 else (w - lo)
        for i in np.argsort(-slack):
            room = slack[i]
            if room <= 0:
                continue
            step = min(room, abs(residual))
            w[i] += step if residual > 0 else -step
            residual += -step if residual > 0 else step
            if abs(residual) <= 1e-12:
                break
    return np.clip(w, lo, hi)


@dataclass
class OptimizeResult:
    weights: np.ndarray
    value: float
    iterations: int
    converged: bool
    restarts: int = 1
    history: list[float] = field(default_factory=list)


def solve(
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
    n: int,
    lo: np.ndarray,
    hi: np.ndarray,
    buckets: Sequence[Bucket] = (),
    bucket_lo: Sequence[float] = (),
    bucket_hi: Sequence[float] = (),
    *,
    max_iter: int = 1500,
    tolerance: float = 1e-10,
    restarts: int = 6,
    seed: int = 0,
    warm_start: np.ndarray | None = None,
) -> OptimizeResult:
    """Maximise ``objective`` over the feasible set by projected gradient ascent.

    ``objective`` returns ``(value, gradient)``. Each step moves along the
    gradient and projects back onto the feasible set; the step size doubles
    down on success and halves on failure, which keeps the method stable on the
    Sharpe ratio's non-concave surface. Several restarts guard against
    stopping at a local optimum.
    """
    rng = np.random.default_rng(seed)
    kwargs = {"buckets": buckets, "bucket_lo": bucket_lo, "bucket_hi": bucket_hi}

    starts = []
    if warm_start is not None:
        starts.append(project(warm_start, lo, hi, **kwargs))
    starts.append(project(np.full(n, 1.0 / n), lo, hi, **kwargs))
    while len(starts) < max(1, restarts):
        starts.append(project(rng.dirichlet(np.ones(n)), lo, hi, **kwargs))

    best: OptimizeResult | None = None
    for start in starts:
        w = start.copy()
        value, grad = objective(w)
        step = 1.0
        converged = False
        history = [value]
        used = 0
        for it in range(max_iter):
            used = it + 1
            improved = False
            for _ in range(40):
                candidate = project(w + step * grad, lo, hi, **kwargs)
                cand_value, cand_grad = objective(candidate)
                if cand_value > value + 1e-16:
                    moved = float(np.max(np.abs(candidate - w)))
                    w, value, grad = candidate, cand_value, cand_grad
                    improved = True
                    step *= 1.6
                    if moved < tolerance:
                        converged = True
                    break
                step *= 0.5
                if step < 1e-18:
                    break
            history.append(value)
            if not improved or converged:
                converged = True
                break
        result = OptimizeResult(
            weights=w, value=value, iterations=used, converged=converged,
            restarts=len(starts), history=history,
        )
        if best is None or result.value > best.value:
            best = result
    assert best is not None
    return best
