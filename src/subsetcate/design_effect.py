"""Design-effect notation for aggregate error (adopted 2026-08; retires cbar/vbar).

For a task with per-unit errors e_i = tauhat_i - tau_i, let

    sigma^2 = average per-unit error variance      (E[e_i^2])
    rho     = intra-task error correlation         (avg_{i != i'} E[e_i e_i'] / sigma^2)

Then the aggregate error over a subset of size k obeys the standard
design-effect form

    Var(thetahat_S - theta_S) = (sigma^2 / k) * [1 + (k - 1) rho]                (1)

The bracket is the design effect. Independent aggregation is exactly the
assumption that it equals 1: it reports sigma^2 / k. Substituting both into
the coverage formula gives, for that estimator,

    coverage(k) = 2 Phi( z / sqrt(1 + (k - 1) rho) ) - 1                         (2)

which is monotonically decreasing in k for any rho > 0.

Legacy mapping (for reading old artifacts): cbar = sigma^2 * rho and
vbar = sigma^2, i.e. rho = cbar / vbar and sigma^2 = vbar.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def from_legacy(cbar: float, vbar: float) -> tuple[float, float]:
    """(cbar, vbar) -> (sigma2, rho)."""
    sigma2 = float(vbar)
    return sigma2, float(cbar) / max(sigma2, 1e-18)


def from_per_unit_errors(e: np.ndarray) -> tuple[float, float]:
    """Direct estimator from a [tasks, units] array of per-unit errors.

    sigma^2 = mean of e^2; rho = mean over tasks of the average product of two
    distinct units' errors, divided by sigma^2. Uses the equality
    sum_{i != i'} e_i e_i' = (sum e)^2 - sum e^2 to avoid the O(n^2) loop.
    """
    e = np.asarray(e, dtype=float)
    n = e.shape[1]
    sigma2 = float(np.mean(e**2))
    cbar = float(np.mean((e.sum(axis=1) ** 2 - (e**2).sum(axis=1)) / (n * (n - 1))))
    return sigma2, cbar / max(sigma2, 1e-18)


def design_effect(k, rho: float) -> np.ndarray:
    """1 + (k - 1) rho."""
    return 1.0 + (np.asarray(k, dtype=float) - 1.0) * rho


def aggregate_variance(k, sigma2: float, rho: float) -> np.ndarray:
    """Equation (1)."""
    k = np.asarray(k, dtype=float)
    return (sigma2 / k) * design_effect(k, rho)


def coverage_under_independent_aggregation(
    k, rho: float, z: float = 1.96, m: float = 1.0
) -> np.ndarray:
    """Equation (2): coverage of the independent-aggregation interval.

    `m` is the per-unit dispersion ratio, sigmahat_i^2 / sigma^2 -- how much
    wider the model's per-unit variance is than the true per-unit error
    variance. The reported aggregate variance is m*sigma^2/k, so

        coverage(k) = 2 Phi( z * sqrt( m / (1 + (k-1) rho) ) ) - 1.

    m = 1 (a per-unit-calibrated model) recovers the bare form, which then
    predicts coverage below nominal for every k > 1. Real models are
    over-dispersed per unit (m >> 1 for the shipped checkpoint on easy
    families), and that cushion is what delays the crossing; using m = 1
    against measured curves under-predicts coverage badly.
    """
    return 2.0 * norm.cdf(z * np.sqrt(m / design_effect(k, rho))) - 1.0


def breaking_k(rho: float, target: float = 0.95, z: float = 1.96, m: float = 1.0) -> float:
    """Smallest k at which equation (2) falls below `target`.

    Solving 2 Phi(z sqrt(m/D)) - 1 = target gives D* = m (z/z_target)^2, so
    k* = 1 + (D* - 1)/rho. With m = 1 and target equal to the nominal level
    this returns k* = 1: the bare form is below nominal for every k > 1, and
    the observed delay is entirely the m cushion.
    """
    z_target = norm.ppf(0.5 * (1.0 + target))
    d_star = m * (z / z_target) ** 2
    if rho <= 0:
        return float("inf")
    return 1.0 + (d_star - 1.0) / rho
