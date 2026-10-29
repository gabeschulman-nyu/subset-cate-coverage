"""Design-effect form of the subset-CATE error variance.

With per-unit error variance sigma^2 and error correlation rho within a task,
the error over a subset S of k units has

    Var(thetahat_S - theta_S) = (sigma^2 / k) * [1 + (k - 1) rho]                (1)

Unit-level aggregation reports sigma^2 / k, so its interval has coverage

    coverage(k) = 2 Phi( z / sqrt(1 + (k - 1) rho) ) - 1                         (2)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def from_legacy(cbar: float, vbar: float) -> tuple[float, float]:
    """(cbar, vbar) -> (sigma2, rho), where cbar = sigma2 * rho and vbar = sigma2."""
    sigma2 = float(vbar)
    return sigma2, float(cbar) / max(sigma2, 1e-18)


def from_per_unit_errors(e: np.ndarray) -> tuple[float, float]:
    """(sigma2, rho) from a [tasks, units] array of per-unit errors e.

    sigma^2 = mean of e^2; rho = mean over tasks of avg_{i != i'} e_i e_i',
    divided by sigma^2, via sum_{i != i'} e_i e_i' = (sum e)^2 - sum e^2.
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
    """Equation (2) with per-unit dispersion ratio m = sigmahat_i^2 / sigma^2:

        coverage(k) = 2 Phi( z * sqrt( m / (1 + (k-1) rho) ) ) - 1.
    """
    return 2.0 * norm.cdf(z * np.sqrt(m / design_effect(k, rho))) - 1.0


def breaking_k(rho: float, target: float = 0.95, z: float = 1.96, m: float = 1.0) -> float:
    """Smallest k at which equation (2) falls below `target`:
    k* = 1 + (D* - 1)/rho with D* = m (z/z_target)^2."""
    z_target = norm.ppf(0.5 * (1.0 + target))
    d_star = m * (z / z_target) ** 2
    if rho <= 0:
        return float("inf")
    return 1.0 + (d_star - 1.0) / rho
