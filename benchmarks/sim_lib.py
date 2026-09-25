"""Python port of the R simulation DGPs for CausalPFN appendix runs.

Mirrors sim_lib.R: dgp1 (assumptions fulfilled) and dgp2 (non-injective PS).
Returns numpy arrays and ground-truth ATE/ATT for PEHE/bias/RMSE scoring.

Validation-phase additions:
    hill_a: Hill (2011) Setting A — linear response surface, homogeneous tau.
    hill_b: Hill (2011) Setting B — non-linear (exponential) response, het. tau.
    kang_schafer: Kang & Schafer (2007) misspecified-PS stress test.
    friedman_confounded: non-linear Friedman outcome with smooth PS.
    hill_b_heavy: Hill B with Student-t residuals (non-Gaussian shape stress).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SimData:
    X: np.ndarray           # (n, k) covariates
    z: np.ndarray           # (n,)   treatment
    y: np.ndarray           # (n,)   observed outcome
    tau: np.ndarray         # (n,)   true unit-level effect
    y0: np.ndarray
    y1: np.ndarray
    p_treat: np.ndarray
    true_ate: float
    true_att: float
    dgp: str


def dgp1(seed: int = 1, n: int = 1000, p: int = 24,
         n_confounders: int = 4, temperature: float = 2.0) -> SimData:
    """Assumptions fulfilled: smooth near-linear PS in a few covariates."""
    rng = np.random.default_rng(seed)
    k = 2 * p + 1
    X = rng.standard_normal((n, k))

    beta_t = np.zeros(k)
    beta_t[:n_confounders] = rng.standard_normal(n_confounders)

    beta_y = rng.standard_normal(k)
    tau_const = 2.0

    p_treat = _sigmoid((X @ beta_t) / temperature)
    z = rng.binomial(1, p_treat).astype(np.float32)

    eps0 = rng.standard_normal(n)
    eps1 = rng.standard_normal(n)
    y0 = X @ beta_y + eps0
    tau = tau_const + 0.5 * X[:, 0]
    y1 = X @ beta_y + tau + eps1
    y = np.where(z == 1, y1, y0)

    return SimData(
        X=X.astype(np.float32), z=z, y=y.astype(np.float32),
        tau=tau.astype(np.float32), y0=y0.astype(np.float32), y1=y1.astype(np.float32),
        p_treat=p_treat.astype(np.float32),
        true_ate=float(np.mean(y1 - y0)),
        true_att=float(np.mean((y1 - y0)[z == 1])),
        dgp="DGP1",
    )


def dgp2(seed: int = 1, n: int = 1000, p: int = 24,
         temperature: float = 1.0, strength: float = 5.0,
         n_active: int = 24) -> SimData:
    """Non-injective PS: quadratic treatment model with heterogeneous tau in signed X."""
    rng = np.random.default_rng(seed)
    k = 2 * p + 1
    X = rng.standard_normal((n, k))

    w_t = np.zeros(k)
    w_t[:n_active] = rng.standard_normal(n_active) * strength
    lin = (X ** 2) @ w_t - w_t.sum()
    p_treat = _sigmoid(lin / (temperature * np.sqrt(n_active)))
    z = rng.binomial(1, p_treat).astype(np.float32)

    beta_y = rng.standard_normal(k)
    tau_w = np.zeros(k)
    tau_w[:n_active] = rng.standard_normal(n_active) * strength / np.sqrt(n_active)
    tau = 2.0 + X @ tau_w

    eps0 = rng.standard_normal(n)
    eps1 = rng.standard_normal(n)
    y0 = X @ beta_y + eps0
    y1 = y0 + tau + (eps1 - eps0)
    y = np.where(z == 1, y1, y0)

    return SimData(
        X=X.astype(np.float32), z=z, y=y.astype(np.float32),
        tau=tau.astype(np.float32), y0=y0.astype(np.float32), y1=y1.astype(np.float32),
        p_treat=p_treat.astype(np.float32),
        true_ate=float(np.mean(y1 - y0)),
        true_att=float(np.mean((y1 - y0)[z == 1])),
        dgp="DGP2",
    )


DGPS = {"DGP1": dgp1, "DGP2": dgp2}


# ---------------------------------------------------------------------------
# Validation-phase synthetic DGPs
# ---------------------------------------------------------------------------

def _finalize(X, z, y0, y1, p_treat, dgp_name, *, mu0, mu1) -> SimData:
    """The unit-level target is the CONDITIONAL effect mu1 - mu0.

    y1 - y0 would add eps1 - eps0, since the two potential outcomes carry
    independent noise; that is the individual effect, not tau(X_i). mu0 and
    mu1 are required so no generator can fall back to it.
    """
    y = np.where(z == 1, y1, y0)
    tau = mu1 - mu0
    return SimData(
        X=X.astype(np.float32), z=z.astype(np.float32), y=y.astype(np.float32),
        tau=tau.astype(np.float32), y0=y0.astype(np.float32), y1=y1.astype(np.float32),
        p_treat=p_treat.astype(np.float32),
        true_ate=float(np.mean(tau)),
        true_att=float(np.mean(tau[z == 1])) if (z == 1).any() else float("nan"),
        dgp=dgp_name,
    )


def hill_a(seed: int = 1, n: int = 1000, p: int = 25,
           sigma_y: float = 1.0) -> SimData:
    """Hill (2011) Setting A: linear response, homogeneous tau.

    Y_t = beta^T X + tau * t + eps,  tau = 4 (constant), eps ~ N(0, sigma_y^2).
    Propensity is a smooth logistic in a few confounders.
    Provides a baseline well-specified-linear DGP for comparison with Hill B.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))

    beta_t = np.zeros(p); beta_t[:5] = rng.standard_normal(5) * 0.5
    p_treat = 1.0 / (1.0 + np.exp(-X @ beta_t))
    z = rng.binomial(1, p_treat).astype(np.float32)

    beta_y = rng.standard_normal(p) * 0.5
    eps0 = rng.standard_normal(n) * sigma_y
    eps1 = rng.standard_normal(n) * sigma_y
    tau_const = 4.0
    mu0 = X @ beta_y
    mu1 = mu0 + tau_const
    y0 = mu0 + eps0
    y1 = mu1 + eps1
    return _finalize(X, z, y0, y1, p_treat, "HillA", mu0=mu0, mu1=mu1)


def hill_b(seed: int = 1, n: int = 1000, p: int = 25,
           sigma_y: float = 1.0, omega: float = 4.0) -> SimData:
    """Hill (2011) Setting B: non-linear (exponential) response, heterogeneous tau.

    Response surfaces follow the form used to generate IHDP-NPCI:
        Y0 = exp((X + W) beta_b)         beta_b coords ~ Cat({0,.1,.2,.3,.4}, p=[.6,.1,.1,.1,.1])
        Y1 = X gamma - omega             gamma coords ~ N(0, 1) scaled so E[tau]≈4
    where omega is chosen so the realized E[Y1 - Y0 | T=1] = 4 (matches Hill).
    Here we follow the Shalit/Johansson simplified variant.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))

    # Propensity: smooth in a subset of confounders.
    beta_t = np.zeros(p); beta_t[:5] = rng.standard_normal(5) * 0.5
    p_treat = 1.0 / (1.0 + np.exp(-X @ beta_t))
    z = rng.binomial(1, p_treat).astype(np.float32)

    W = 0.5 * np.ones((1, p))
    choices = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    probs = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    beta_b = rng.choice(choices, size=p, p=probs)

    mu0 = np.exp((X + W) @ beta_b)
    gamma = rng.standard_normal(p) * 0.3
    mu1 = X @ gamma
    # Center mu1 so realized E[mu1 - mu0 | t=1] = 4 (Hill's "ate=4" convention).
    treated_mask = z == 1
    if treated_mask.any():
        shift = 4.0 - float((mu1[treated_mask] - mu0[treated_mask]).mean())
        mu1 = mu1 + shift

    eps0 = rng.standard_normal(n) * sigma_y
    eps1 = rng.standard_normal(n) * sigma_y
    y0 = mu0 + eps0
    y1 = mu1 + eps1
    return _finalize(X, z, y0, y1, p_treat, "HillB", mu0=mu0, mu1=mu1)


def kang_schafer(seed: int = 1, n: int = 1000) -> SimData:
    """Kang & Schafer (2007): misspecified propensity stress test.

    Latent Z ~ N(0, I_4). Observed X = nonlinear(Z) so a linear PS in X is
    severely misspecified, but the analyst sees only X. Outcome is linear in Z:
        Y0 = 210 + 27.4*Z1 + 13.7*Z2 + 13.7*Z3 + 13.7*Z4 + eps
        Y1 = Y0 + tau,  tau = 0 (the classical KS setup has null effect), so the
    conditional effect is 0 for every unit; Y1 - Y0 is pure noise.
    We follow the canonical version with sigma=1, tau=0 (true ATE=0).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 4))
    z1, z2, z3, z4 = Z.T

    # KS observed covariates (nonlinear transforms of Z)
    x1 = np.exp(z1 / 2.0)
    x2 = z2 / (1.0 + np.exp(z1)) + 10.0
    x3 = (z1 * z3 / 25.0 + 0.6) ** 3
    x4 = (z2 + z4 + 20.0) ** 2
    X = np.stack([x1, x2, x3, x4], axis=1)

    # True propensity (in Z, unknown to analyst):
    logit = -z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4
    p_treat = 1.0 / (1.0 + np.exp(-logit))
    z = rng.binomial(1, p_treat).astype(np.float32)

    eps0 = rng.standard_normal(n)
    eps1 = rng.standard_normal(n)
    mu0 = 210 + 27.4 * z1 + 13.7 * z2 + 13.7 * z3 + 13.7 * z4
    tau = 0.0
    mu1 = mu0 + tau
    y0 = mu0 + eps0
    y1 = mu1 + eps1
    return _finalize(X, z, y0, y1, p_treat, "KangSchafer", mu0=mu0, mu1=mu1)


def friedman_confounded(seed: int = 1, n: int = 1000, p: int = 10,
                        sigma_y: float = 1.0) -> SimData:
    """Friedman-style non-linear outcome with smooth confounded propensity.

    Y0 = 10 sin(pi X1 X2) + 20 (X3 - 0.5)^2 + 10 X4 + 5 X5 + eps
    tau(X) = 2 + X1 - X2     (heterogeneous, sign-flips with X1 - X2)
    PS smooth in (X1, X3): logistic with moderate confounding.
    Breaks the "linear outcome" assumption that favored linear/matching baselines.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, p))
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    logit = 2.0 * (x1 - 0.5) + 1.5 * (x3 - 0.5)
    p_treat = 1.0 / (1.0 + np.exp(-logit))
    z = rng.binomial(1, p_treat).astype(np.float32)

    base = (10.0 * np.sin(np.pi * x1 * x2)
            + 20.0 * (x3 - 0.5) ** 2
            + 10.0 * x4 + 5.0 * x5)
    tau = 2.0 + x1 - x2
    eps0 = rng.standard_normal(n) * sigma_y
    eps1 = rng.standard_normal(n) * sigma_y
    y0 = base + eps0
    y1 = base + tau + eps1
    return _finalize(X, z, y0, y1, p_treat, "FriedmanConf", mu0=base, mu1=base + tau)


def hill_b_heavy(seed: int = 1, n: int = 1000, p: int = 25,
                 df: float = 3.0, sigma_y: float = 1.0) -> SimData:
    """Hill B with Student-t(df) residuals to stress the histogram form."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_t = np.zeros(p); beta_t[:5] = rng.standard_normal(5) * 0.5
    p_treat = 1.0 / (1.0 + np.exp(-X @ beta_t))
    z = rng.binomial(1, p_treat).astype(np.float32)

    W = 0.5 * np.ones((1, p))
    choices = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    probs = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    beta_b = rng.choice(choices, size=p, p=probs)
    mu0 = np.exp((X + W) @ beta_b)
    gamma = rng.standard_normal(p) * 0.3
    mu1 = X @ gamma
    if (z == 1).any():
        shift = 4.0 - float((mu1[z == 1] - mu0[z == 1]).mean())
        mu1 = mu1 + shift

    eps0 = rng.standard_t(df, size=n) * sigma_y
    eps1 = rng.standard_t(df, size=n) * sigma_y
    y0 = mu0 + eps0
    y1 = mu1 + eps1
    return _finalize(X, z, y0, y1, p_treat, "HillBHeavy", mu0=mu0, mu1=mu1)


DGPS.update({
    "HillA": hill_a,
    "HillB": hill_b,
    "KangSchafer": kang_schafer,
    "FriedmanConf": friedman_confounded,
    "HillBHeavy": hill_b_heavy,
})


try:
    from bench_lib import ihdp as _ihdp, acic2016 as _acic2016
    DGPS["IHDP"] = _ihdp
    DGPS["ACIC2016"] = _acic2016
except Exception as _e:
    import warnings as _w
    _w.warn(f"bench_lib unavailable (real benchmarks disabled): {_e}")
