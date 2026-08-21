"""The objectives you can ask the optimiser to maximise.

Every objective is a callable ``w -> (value, gradient)`` that the solver
maximises. Minimisation problems are expressed as a negated value.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..stats import Moments

Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]

OBJECTIVES = (
    "risk_parity",
    "max_sharpe",
    "min_vol",
    "target_vol",
    "equal_weight",
    "max_return",
)

#: Why each objective exists, shown by ``ainvest explain``.
DESCRIPTIONS = {
    "risk_parity": (
        "Equalise each holding's contribution to portfolio risk. Uses only the "
        "covariance matrix, never the return estimates, so it is the most "
        "stable choice for a small sample. The default."
    ),
    "max_sharpe": (
        "Maximise return per unit of volatility. Depends on expected-return "
        "estimates, which are noisy, so it tends to concentrate. Pair it with "
        "mean_shrinkage and blend_equal_weight."
    ),
    "min_vol": "Smallest possible portfolio volatility. Ignores returns entirely.",
    "target_vol": (
        "Highest expected return whose volatility stays at or below "
        "optimizer.target_vol."
    ),
    "equal_weight": (
        "Same weight for everything, adjusted to respect the bounds. A "
        "surprisingly hard benchmark to beat out of sample."
    ),
    "max_return": (
        "Chase the highest estimated return subject only to the weight caps. "
        "Included for comparison; it is what over-fitting looks like."
    ),
}


def max_sharpe(moments: Moments, risk_free: float = 0.0) -> Objective:
    cov, mu = moments.cov, moments.mu

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        var = float(w @ cov @ w)
        sigma = np.sqrt(max(var, 1e-18))
        excess = float(w @ mu) - risk_free
        value = excess / sigma
        # d/dw [ (w'mu - rf) / sqrt(w'Sw) ]
        grad = mu / sigma - excess * (cov @ w) / (sigma**3)
        return value, grad

    return f


def min_vol(moments: Moments) -> Objective:
    cov = moments.cov

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        var = float(w @ cov @ w)
        return -var, -2.0 * (cov @ w)

    return f


def max_return(moments: Moments) -> Objective:
    mu = moments.mu

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        return float(w @ mu), mu.copy()

    return f


def mean_variance(moments: Moments, risk_aversion: float) -> Objective:
    """Classic utility ``w'mu - (k/2) w'Sw``; ``k`` sets the risk appetite."""
    cov, mu = moments.cov, moments.mu

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        value = float(w @ mu) - 0.5 * risk_aversion * float(w @ cov @ w)
        grad = mu - risk_aversion * (cov @ w)
        return value, grad

    return f


def risk_parity(moments: Moments, floor: float = 1e-9) -> Objective:
    """Equal risk contribution, in its scale-invariant logarithmic form.

    Minimises ``0.5*log(w'Sw) - (1/n) sum(log w_i)``. Both terms shift by the
    same constant when ``w`` is rescaled, so the objective does not care about
    the magnitude of the covariance matrix -- which matters, because the more
    familiar ``0.5 w'Sw - c*sum(log w)`` form is only equivalent to ERC when
    minimised over the unnormalised positive orthant. On the simplex that form
    silently degenerates toward equal weight for small annualised covariances.

    At a stationary point on the simplex this form gives *exactly* equal risk
    contributions: setting the gradient to ``lambda`` and multiplying through by
    ``w_i`` gives ``RC_i - 1/n = lambda * w_i``; summing over ``i`` forces
    ``lambda = 0``, hence ``RC_i = 1/n`` for every asset. Bounds and bucket
    caps can of course prevent the portfolio from reaching that point.

    The log barrier also keeps every weight strictly positive, so this never
    returns the all-in-one-asset answer a raw variance objective can.
    """
    cov = moments.cov
    n = cov.shape[0]

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        safe = np.clip(w, floor, None)
        var = max(float(w @ cov @ w), 1e-24)
        value = -(0.5 * np.log(var) - float(np.sum(np.log(safe))) / n)
        grad = -((cov @ w) / var - (1.0 / n) / safe)
        return value, grad

    return f


def build_objective(
    name: str,
    moments: Moments,
    *,
    risk_free: float = 0.0,
    risk_aversion: float = 1.0,
) -> Objective:
    if name == "max_sharpe":
        return max_sharpe(moments, risk_free)
    if name == "min_vol":
        return min_vol(moments)
    if name == "max_return":
        return max_return(moments)
    if name == "risk_parity":
        return risk_parity(moments)
    if name == "mean_variance":
        return mean_variance(moments, risk_aversion)
    raise ValueError(
        f"unknown objective {name!r}; expected one of {', '.join(OBJECTIVES)}"
    )
