"""Each objective must actually optimise the thing it claims to optimise."""

from __future__ import annotations

import numpy as np
import pytest

from ainvest.allocator import optimize_weights
from ainvest.config import from_dict
from ainvest.optimize.core import project, solve
from ainvest.optimize.objectives import (
    OBJECTIVES,
    build_objective,
    max_sharpe,
    min_vol,
    risk_parity,
)
from ainvest.stats import Moments
from conftest import base_config


def toy_moments(n=5, seed=0) -> Moments:
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n, 2))
    cov = loadings @ loadings.T * 0.02 + np.diag(rng.uniform(0.01, 0.4, n))
    mu = rng.uniform(0.02, 0.30, n)
    return Moments([f"A{i}" for i in range(n)], mu, cov, 252.0, 500)


def beats_random(objective, m, n, lo, hi, value_fn, samples=30000, seed=0):
    """Optimiser output vs brute force. A local optimum would lose this."""
    result = solve(objective, n, lo, hi, restarts=5)
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(samples):
        w = project(rng.dirichlet(np.ones(n) * rng.uniform(0.2, 4.0)), lo, hi)
        best = max(best, value_fn(w))
    return value_fn(result.weights), best, result.weights


def test_risk_parity_equalises_risk_contributions():
    """Unconstrained, every asset must carry exactly 1/n of the risk. This is
    the property the scale-invariant log formulation guarantees; the more
    common `0.5 w'Sw - c*sum(log w)` form does not, on the simplex."""
    for seed in range(6):
        m = toy_moments(seed=seed)
        n = len(m.symbols)
        result = solve(risk_parity(m), n, np.zeros(n), np.ones(n), restarts=4)
        rc = m.risk_contributions(result.weights)
        assert np.allclose(rc, 1.0 / n, atol=1e-5), f"seed {seed}: {rc}"


def test_risk_parity_is_insensitive_to_covariance_scale():
    """Annualising the covariance must not change the answer."""
    m = toy_moments(seed=1)
    n = len(m.symbols)
    scaled = Moments(m.symbols, m.mu, m.cov * 252.0, 252.0, 500)
    a = solve(risk_parity(m), n, np.zeros(n), np.ones(n), restarts=4).weights
    b = solve(risk_parity(scaled), n, np.zeros(n), np.ones(n), restarts=4).weights
    assert np.allclose(a, b, atol=1e-4)


def test_risk_parity_underweights_the_volatile_asset():
    cov = np.diag([0.01, 0.04, 0.36])  # vols of 10%, 20%, 60%
    m = Moments(["calm", "mid", "wild"], np.zeros(3), cov, 252.0, 500)
    w = solve(risk_parity(m), 3, np.zeros(3), np.ones(3), restarts=3).weights
    assert w[0] > w[1] > w[2]
    # For uncorrelated assets ERC weights are proportional to 1/vol.
    assert np.allclose(w / w.sum(), np.array([1 / 0.1, 1 / 0.2, 1 / 0.6]) / (1 / 0.1 + 1 / 0.2 + 1 / 0.6), atol=1e-4)


def test_min_vol_beats_brute_force():
    m = toy_moments(seed=2)
    n = len(m.symbols)
    got, best, _ = beats_random(
        min_vol(m), m, n, np.zeros(n), np.ones(n), lambda w: -m.portfolio_vol(w)
    )
    assert got >= best - 1e-6, f"optimiser {-got:.6f} lost to random {-best:.6f}"


def test_max_sharpe_beats_brute_force():
    m = toy_moments(seed=3)
    n = len(m.symbols)
    rf = 0.02

    def sharpe(w):
        vol = m.portfolio_vol(w)
        return (m.portfolio_return(w) - rf) / vol if vol > 0 else -np.inf

    got, best, _ = beats_random(max_sharpe(m, rf), m, n, np.zeros(n), np.ones(n), sharpe)
    assert got >= best - 1e-6, f"optimiser {got:.6f} lost to random {best:.6f}"


def test_max_sharpe_and_min_vol_differ_when_returns_differ():
    """Two objectives that always agree would mean one of them is not running.
    Give one asset a much better return at slightly higher risk and they must
    part ways."""
    cov = np.array([[0.04, 0.0], [0.0, 0.05]])
    m = Moments(["safe", "better"], np.array([0.03, 0.40]), cov, 252.0, 500)
    lo, hi = np.zeros(2), np.ones(2)
    mv = solve(min_vol(m), 2, lo, hi, restarts=3).weights
    ms = solve(max_sharpe(m, 0.0), 2, lo, hi, restarts=3).weights
    assert ms[1] > mv[1] + 0.2


def test_objective_gradients_match_finite_differences():
    """A wrong gradient still 'converges' -- to the wrong place."""
    m = toy_moments(seed=5)
    n = len(m.symbols)
    rng = np.random.default_rng(9)
    for name in ("max_sharpe", "min_vol", "risk_parity", "max_return", "mean_variance"):
        objective = build_objective(name, m, risk_free=0.02, risk_aversion=3.0)
        w = project(rng.dirichlet(np.ones(n)), np.zeros(n), np.ones(n))
        _, grad = objective(w)
        eps = 1e-7
        numeric = np.empty(n)
        for i in range(n):
            step = np.zeros(n)
            step[i] = eps
            numeric[i] = (objective(w + step)[0] - objective(w - step)[0]) / (2 * eps)
        assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-6), name


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective"):
        build_objective("moon", toy_moments())


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_objective_produces_a_valid_allocation(objective, moments):
    cfg = from_dict(base_config(optimizer={"objective": objective, "restarts": 2}))
    allocation = optimize_weights(cfg, moments)
    assert sum(allocation.weights) == pytest.approx(1.0, abs=1e-6)
    assert all(w >= -1e-9 for w in allocation.weights)
    crypto = sum(
        w for s, w in zip(allocation.symbols, allocation.weights, strict=True)
        if cfg.asset(s).asset_class == "crypto"
    )
    assert crypto <= cfg.constraints.crypto_max + 1e-6
    for symbol, weight in zip(allocation.symbols, allocation.weights, strict=True):
        assert weight <= min(cfg.asset(symbol).max_weight, cfg.constraints.single_max) + 1e-6


def test_target_vol_hits_its_target(moments):
    """The target must be met by the weights actually returned -- including the
    equal-weight blend, which is applied after the solve and pushes volatility
    back up. Calibrating on the raw solver output silently missed by that gap.
    """
    settings = {"blend_equal_weight": 0.2, "restarts": 2}

    def vol_for(objective, **extra):
        cfg = from_dict(base_config(optimizer={
            "objective": objective, **settings, **extra
        }))
        return optimize_weights(cfg, moments).expected_vol

    floor = vol_for("min_vol")
    ceiling = vol_for("max_return")

    # A target inside the reachable range is met closely.
    inside = 0.5 * (floor + ceiling)
    assert vol_for("target_vol", target_vol=inside) == pytest.approx(inside, abs=0.005)

    # A target below the reachable floor saturates there rather than pretending.
    assert vol_for("target_vol", target_vol=floor * 0.5) == pytest.approx(floor, abs=0.005)

    # And a target above the ceiling gives the most aggressive feasible mix.
    assert vol_for("target_vol", target_vol=ceiling * 2) == pytest.approx(ceiling, abs=0.02)


def test_target_vol_is_monotone_in_its_target(moments):
    vols = []
    for target in (0.10, 0.15, 0.20, 0.30):
        cfg = from_dict(base_config(optimizer={
            "objective": "target_vol", "target_vol": target, "restarts": 2,
        }))
        vols.append(optimize_weights(cfg, moments).expected_vol)
    # Deliberately offset by one to compare consecutive pairs, so the two
    # sequences differ in length by design.
    assert all(b >= a - 1e-6 for a, b in zip(vols, vols[1:], strict=False)), vols


def test_equal_weight_respects_caps(moments):
    cfg = from_dict(base_config(
        optimizer={"objective": "equal_weight"},
        constraints={"crypto_max": 0.10},
    ))
    allocation = optimize_weights(cfg, moments)
    crypto = sum(
        w for s, w in zip(allocation.symbols, allocation.weights, strict=True)
        if cfg.asset(s).asset_class == "crypto"
    )
    assert crypto == pytest.approx(0.10, abs=1e-6)


def test_blending_toward_equal_weight_raises_diversification(moments):
    plain = optimize_weights(
        from_dict(base_config(optimizer={"objective": "max_sharpe", "blend_equal_weight": 0.0})),
        moments,
    )
    blended = optimize_weights(
        from_dict(base_config(optimizer={"objective": "max_sharpe", "blend_equal_weight": 0.5})),
        moments,
    )
    assert blended.diversification >= plain.diversification - 1e-9
