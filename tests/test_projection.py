"""The projection is the load-bearing piece of the optimiser: if it returns an
infeasible point, every constraint in the config is a suggestion."""

from __future__ import annotations

import numpy as np
import pytest

from ainvest.optimize.core import InfeasibleError, project


def random_problem(rng):
    """A random but always-feasible constraint set."""
    n = int(rng.integers(3, 9))
    # Floors must leave room to reach a total of 1, and ceilings must be able
    # to get there, or there is no feasible point to project onto at all.
    lo = np.round(rng.uniform(0.0, 0.5 / n, n), 4)
    hi = np.clip(lo + np.round(rng.uniform(0.3, 0.9, n), 4), 0.0, 1.0)
    if hi.sum() < 1.2:
        hi = np.minimum(hi * 2.0, 1.0)
    k = int(rng.integers(1, min(3, n) + 1))
    parts = np.array_split(rng.permutation(n), k)
    buckets = [tuple(sorted(int(i) for i in part)) for part in parts]
    blo = [0.0] * k
    # A group cap is only as useful as its members' own caps allow, so check
    # feasibility against the reachable total rather than the nominal one.
    reach = [float(hi[list(b)].sum()) for b in buckets]
    bhi = [float(rng.uniform(0.5, 1.0)) for _ in range(k)]
    for _ in range(10):
        if sum(min(cap, r) for cap, r in zip(bhi, reach, strict=True)) >= 1.0:
            break
        bhi = [min(1.0, cap * 1.5) for cap in bhi]
    else:
        bhi = [1.0] * k
    return n, lo, hi, buckets, blo, bhi


def test_output_is_always_feasible():
    rng = np.random.default_rng(3)
    for _ in range(400):
        n, lo, hi, buckets, blo, bhi = random_problem(rng)
        w = project(rng.normal(0.2, 0.6, n), lo, hi, buckets, blo, bhi)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(w >= lo - 1e-9)
        assert np.all(w <= hi + 1e-9)
        for b, low, high in zip(buckets, blo, bhi, strict=True):
            assert low - 1e-8 <= w[list(b)].sum() <= high + 1e-8


def test_satisfies_the_variational_inequality():
    """The defining property: <y - P(y), z - P(y)> <= 0 for every feasible z.

    This pins down *the* projection rather than merely a feasible point, and
    it needs no reference solver to check against.
    """
    rng = np.random.default_rng(11)
    worst = -np.inf
    for _ in range(120):
        n, lo, hi, buckets, blo, bhi = random_problem(rng)
        y = rng.normal(0.2, 0.7, n)
        w = project(y, lo, hi, buckets, blo, bhi)
        for _ in range(30):
            z = project(rng.normal(0.2, 0.7, n), lo, hi, buckets, blo, bhi)
            worst = max(worst, float((y - w) @ (z - w)))
    assert worst <= 1e-9, f"projection is not optimal; worst inner product {worst}"


def test_feasible_points_are_returned_unchanged():
    lo, hi = np.zeros(4), np.full(4, 0.5)
    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(project(w, lo, hi), w)


def test_projection_is_idempotent():
    rng = np.random.default_rng(5)
    for _ in range(80):
        n, lo, hi, buckets, blo, bhi = random_problem(rng)
        once = project(rng.normal(0.2, 0.6, n), lo, hi, buckets, blo, bhi)
        twice = project(once, lo, hi, buckets, blo, bhi)
        assert np.allclose(once, twice, atol=1e-9)


def test_bucket_ceiling_binds():
    """A crypto cap must hold even when the unconstrained answer wants more."""
    n = 4
    lo, hi = np.zeros(n), np.ones(n)
    buckets, blo, bhi = [(0, 1), (2, 3)], [0.0, 0.0], [1.0, 0.25]
    w = project(np.array([0.05, 0.05, 0.6, 0.6]), lo, hi, buckets, blo, bhi)
    assert w[2] + w[3] == pytest.approx(0.25, abs=1e-9)
    assert w[0] + w[1] == pytest.approx(0.75, abs=1e-9)


def test_bucket_floor_binds():
    n = 4
    lo, hi = np.zeros(n), np.ones(n)
    buckets, blo, bhi = [(0, 1), (2, 3)], [0.0, 0.30], [1.0, 1.0]
    w = project(np.array([0.9, 0.9, 0.0, 0.0]), lo, hi, buckets, blo, bhi)
    assert w[2] + w[3] == pytest.approx(0.30, abs=1e-9)


def test_per_asset_floor_is_respected():
    lo = np.array([0.10, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    w = project(np.array([-5.0, 1.0, 1.0]), lo, hi)
    assert w[0] == pytest.approx(0.10, abs=1e-9)


@pytest.mark.parametrize(
    "lo, hi, buckets, blo, bhi, fragment",
    [
        # caps too low to reach 100% invested
        (np.zeros(3), np.full(3, 0.2), (), (), (), "per-asset bounds allow a total"),
        # floors that add up to more than the whole portfolio
        (np.full(3, 0.4), np.ones(3), (), (), (), "per-asset bounds allow a total"),
        # two group caps that together cannot reach 100%
        (np.zeros(4), np.ones(4), [(0, 1), (2, 3)], [0.0, 0.0], [0.3, 0.3],
         "group and per-asset bounds allow a total"),
        (np.array([0.5, 0.0]), np.array([0.2, 1.0]), (), (), (), "min_weight above"),
    ],
)
def test_impossible_constraints_are_reported(lo, hi, buckets, blo, bhi, fragment):
    with pytest.raises(InfeasibleError, match=fragment):
        project(np.full(len(lo), 1.0 / len(lo)), lo, hi, buckets, blo, bhi)


def test_buckets_must_partition_the_assets():
    lo, hi = np.zeros(3), np.ones(3)
    with pytest.raises(ValueError, match="disjoint"):
        project(np.full(3, 1 / 3), lo, hi, [(0, 1)], [0.0], [1.0])
    with pytest.raises(ValueError, match="disjoint"):
        project(np.full(3, 1 / 3), lo, hi, [(0, 1), (1, 2)], [0.0, 0.0], [1.0, 1.0])
