"""Exact targets for the polynomial SCM family, in closed form (no Monte Carlo).

Interventional targets (PATE, CATE) use E[Y | do(t), x] = mu_t(x).
Counterfactual targets (ITE, SATE, subset-CATE) abduct u_y = y - f_y(parents)
from each row and apply both interventions with it; u_y cancels for this
additive family. The prior attaches the spec to the SCM as ``analytic_spec``.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from scipy import stats
from scipy.special import expit

from subsetcate.mechanisms import build_synthetic_causal_model

PROPENSITY_FLOOR = 0.05
PROPENSITY_SPAN = 0.90

# Lexicographic node ordering must equal numeric ordering for x_1..x_d
# (dowhy feeds parents to mechanisms in sorted order), so d stays single-digit.
MAX_COVARIATES = 9


@dataclass(frozen=True)
class PolynomialSCMSpec:
    """Closed-form mechanisms on the quadratic basis phi(X) = [X, X**2].

    propensity(x) = floor + span * expit(b_p + phi @ w_p)   (strict overlap)
    mu0(x)        = b_b + phi @ w_b
    tau(x)        = b_e + phi @ w_e
    y             = mu0(x) + t * tau(x) + u_y,  u_y ~ N(0, outcome_noise_scale**2)
    """

    baseline_weights: np.ndarray
    baseline_intercept: float
    effect_weights: np.ndarray
    effect_intercept: float
    propensity_weights: np.ndarray
    propensity_intercept: float
    outcome_noise_scale: float
    # Optional knobs; defaults give the base family. Hidden terms enter y
    # additively, so ITE = tau(X) for every setting.
    #   unobs_noise_coef (gamma_n): hidden non-confounder u_zn -> y.
    #   unobs_conf_coef (gamma): hidden confounder u_zc -> t logits and y.
    #   overlap_*: p = floor + span * expit(sharpness * logits).
    unobs_noise_coef: float = 0.0
    unobs_conf_coef: float = 0.0
    # Exchangeable covariate correlation r: X_j = sqrt(r) Z + sqrt(1-r) eps_j,
    # Z shared across j. Marginals stay N(0, 1), so the exact PATE is unchanged.
    covariate_corr: float = 0.0
    overlap_floor: float = PROPENSITY_FLOOR
    overlap_span: float = PROPENSITY_SPAN
    overlap_sharpness: float = 1.0

    @property
    def num_covariates(self) -> int:
        return len(self.baseline_weights) // 2

    def _basis(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        return np.concatenate([X, X**2], axis=1)

    def mu0(self, X: np.ndarray) -> np.ndarray:
        return self.baseline_intercept + self._basis(X) @ self.baseline_weights

    def tau(self, X: np.ndarray) -> np.ndarray:
        return self.effect_intercept + self._basis(X) @ self.effect_weights

    def mu1(self, X: np.ndarray) -> np.ndarray:
        return self.mu0(X) + self.tau(X)

    def propensity(self, X: np.ndarray, u_conf: np.ndarray | None = None) -> np.ndarray:
        logits = self.propensity_intercept + self._basis(X) @ self.propensity_weights
        if self.unobs_conf_coef != 0.0:
            if u_conf is None:
                raise ValueError("unobs_conf_coef != 0 requires u_conf draws")
            logits = logits + self.unobs_conf_coef * np.asarray(u_conf, dtype=np.float64)
        return self.overlap_floor + self.overlap_span * expit(
            self.overlap_sharpness * logits
        )

    def exact_pate(self) -> float:
        """E[tau(X)] for X_j ~iid N(0,1): E[X_j]=0, E[X_j^2]=1."""
        d = self.num_covariates
        return float(self.effect_intercept + self.effect_weights[d:].sum())

    def abduct_outcome_noise(
        self, X: np.ndarray, t: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Recover realized u_y from observed rows via the known mechanism."""
        return np.asarray(y, dtype=np.float64) - (
            self.mu0(X) + np.asarray(t, dtype=np.float64) * self.tau(X)
        )

    def ite_from_observed(
        self, X: np.ndarray, t: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Exact ITE by abduction; equals tau(X) for this additive family."""
        u = self.abduct_outcome_noise(X, t, y)
        y_treated = self.mu1(X) + u
        y_control = self.mu0(X) + u
        return y_treated - y_control


def sample_polynomial_spec(
    num_covariates: int,
    outcome_noise_scale: float = 0.25,
    effect_scale: float = 0.35,
    unobs_noise_coef: float = 0.0,
    unobs_conf_coef: float = 0.0,
    overlap_floor: float = PROPENSITY_FLOOR,
    overlap_sharpness: float = 1.0,
    covariate_corr: float = 0.0,
) -> PolynomialSCMSpec:
    """Draw one task's spec from the global np.random stream; p stays in
    [overlap_floor, 1 - overlap_floor]."""
    if not 1 <= num_covariates <= MAX_COVARIATES:
        raise ValueError(f"num_covariates must be in [1, {MAX_COVARIATES}]")
    two_d = 2 * num_covariates
    return PolynomialSCMSpec(
        baseline_weights=np.random.normal(scale=0.5, size=two_d),
        baseline_intercept=0.0,
        effect_weights=np.random.normal(scale=effect_scale, size=two_d),
        effect_intercept=0.25,
        propensity_weights=np.random.normal(scale=0.35, size=two_d),
        propensity_intercept=0.0,
        outcome_noise_scale=outcome_noise_scale,
        unobs_noise_coef=unobs_noise_coef,
        unobs_conf_coef=unobs_conf_coef,
        overlap_floor=overlap_floor,
        overlap_span=1.0 - 2.0 * overlap_floor,
        overlap_sharpness=overlap_sharpness,
        covariate_corr=covariate_corr,
    )


def covariate_names(num_covariates: int) -> list[str]:
    return [f"x_{i + 1}" for i in range(num_covariates)]


def build_polynomial_scm(spec: PolynomialSCMSpec) -> gcm.InvertibleStructuralCausalModel:
    """Build a dowhy SCM from the spec, attached as ``analytic_spec``. The
    knobs add unobserved roots u_zc (-> t, y), u_zn (-> y) and z_cov (-> x_j)."""
    d = spec.num_covariates
    x_names = covariate_names(d)
    has_conf = spec.unobs_conf_coef != 0.0
    has_noise = spec.unobs_noise_coef != 0.0

    r = float(spec.covariate_corr)
    if not 0.0 <= r < 1.0:
        raise ValueError(f"covariate_corr must be in [0, 1), got {r}")
    has_corr = r > 0.0

    edges = [(x, "t") for x in x_names] + [(x, "y") for x in x_names] + [("t", "y")]
    if has_corr:
        edges += [("z_cov", x) for x in x_names]
    if has_conf:
        edges += [("u_zc", "t"), ("u_zc", "y")]
    if has_noise:
        edges += [("u_zn", "y")]
    graph = nx.DiGraph(edges)

    def t_function(parents: np.ndarray) -> np.ndarray:
        # Parents sorted: [u_zc,] x_1..x_d ('u' < 'x').
        if has_conf:
            u_c, X = parents[:, 0], parents[:, 1:]
        else:
            u_c, X = None, parents
        p = spec.propensity(X, u_conf=u_c)
        return np.random.binomial(n=1, p=p).astype(np.float64)

    def y_function(parents: np.ndarray) -> np.ndarray:
        # Parents sorted: t, [u_zc,] [u_zn,] x_1..x_d.
        t = parents[:, 0]
        idx = 1
        extra = np.zeros_like(t)
        if has_conf:
            extra = extra + spec.unobs_conf_coef * parents[:, idx]
            idx += 1
        if has_noise:
            extra = extra + spec.unobs_noise_coef * parents[:, idx]
            idx += 1
        X = parents[:, idx:]
        return spec.mu0(X) + t * spec.tau(X) + extra

    if has_corr:
        # x_j = sqrt(r) * z_cov + sqrt(1-r) * eps_j  =>  Var(x_j) = 1, corr = r
        def _x_function(parents: np.ndarray, _a=np.sqrt(r)) -> np.ndarray:
            return _a * parents[:, 0]

        node_function_dict: dict = {
            name: (_x_function,
                   gcm.ScipyDistribution(stats.norm, loc=0, scale=np.sqrt(1.0 - r)))
            for name in x_names
        }
        node_function_dict["z_cov"] = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
    else:
        node_function_dict = {
            name: (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1)) for name in x_names
        }
    if has_conf:
        node_function_dict["u_zc"] = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
    if has_noise:
        node_function_dict["u_zn"] = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
    node_function_dict["t"] = (t_function, None)
    node_function_dict["y"] = (
        y_function,
        gcm.ScipyDistribution(stats.norm, loc=0, scale=spec.outcome_noise_scale),
    )

    scm = build_synthetic_causal_model(
        causal_graph=graph, node_function_dict=node_function_dict
    )
    scm.analytic_spec = spec
    return scm


def make_polynomial_prior_fn(
    num_covariates: int,
    outcome_noise_scale: float = 0.25,
    effect_scale: float = 0.35,
    unobs_noise_coef: float = 0.0,
    unobs_conf_coef: float = 0.0,
    overlap_floor: float = PROPENSITY_FLOOR,
    overlap_sharpness: float = 1.0,
    covariate_corr: float = 0.0,
):
    def prior_fn() -> gcm.InvertibleStructuralCausalModel:
        spec = sample_polynomial_spec(
            num_covariates, outcome_noise_scale, effect_scale,
            unobs_noise_coef, unobs_conf_coef, overlap_floor, overlap_sharpness,
            covariate_corr,
        )
        return build_polynomial_scm(spec)

    return prior_fn


def _get_spec(causal_model: gcm.ProbabilisticCausalModel) -> PolynomialSCMSpec:
    spec = getattr(causal_model, "analytic_spec", None)
    if spec is None:
        raise ValueError(
            "Exact query functions require an SCM with an attached analytic_spec "
            "(build the prior with make_polynomial_prior_fn)."
        )
    return spec


# ---- Exact interventional path ----
# Each factory tags its query_fn with `exact_path`; train.py requires the
# counterfactual path for route_a.

EXACT_PATH_INTERVENTIONAL = "interventional"
EXACT_PATH_COUNTERFACTUAL = "counterfactual_abduction"


def require_counterfactual_target(query_fn) -> None:
    """Raise unless query_fn returns abducted ITEs."""
    path = getattr(query_fn, "exact_path", None)
    if path != EXACT_PATH_COUNTERFACTUAL:
        raise ValueError(
            "Route A reconstruction targets must be exact abducted ITEs "
            f"(exact_path='{EXACT_PATH_COUNTERFACTUAL}'); got "
            f"exact_path={path!r}. CATE/interventional or untagged targets "
            "would silently estimate a different functional on non-additive "
            "families."
        )


def make_exact_pate_query_fn():
    def query_fn(causal_model, query_point=None) -> float:
        return _get_spec(causal_model).exact_pate()

    query_fn.exact_path = EXACT_PATH_INTERVENTIONAL
    return query_fn


def make_exact_cate_query_fn(query_feature_names: list[str]):
    x_cols = [c for c in query_feature_names if c not in ("t", "y")]

    def query_fn(causal_model, query_point: pd.DataFrame) -> np.ndarray:
        spec = _get_spec(causal_model)
        X = query_point[x_cols].values
        return spec.tau(X)[:, np.newaxis]

    query_fn.exact_path = EXACT_PATH_INTERVENTIONAL
    return query_fn


# ---- Exact counterfactual path (explicit abduction) ----


def make_exact_ite_query_fn(outcome_name: str = "y", treatment_name: str = "t"):
    def query_fn(causal_model, observed_data: pd.DataFrame) -> np.ndarray:
        spec = _get_spec(causal_model)
        x_cols = covariate_names(spec.num_covariates)
        X = observed_data[x_cols].values
        t_obs = observed_data[treatment_name].values
        y_obs = observed_data[outcome_name].values
        return spec.ite_from_observed(X, t_obs, y_obs)[:, np.newaxis]

    query_fn.exact_path = EXACT_PATH_COUNTERFACTUAL
    return query_fn


def make_exact_state_query_fn(outcome_name: str = "y", treatment_name: str = "t"):
    """Exact SATE: mean abducted ITE over the rows passed in. On a subset S
    this is the subset-CATE theta_S (query GATE)."""
    ite_fn = make_exact_ite_query_fn(outcome_name, treatment_name)

    def query_fn(causal_model, observed_data: pd.DataFrame) -> float:
        return float(ite_fn(causal_model, observed_data).mean())

    query_fn.exact_path = EXACT_PATH_COUNTERFACTUAL
    return query_fn
