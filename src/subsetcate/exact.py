"""Exact-target machinery for the polynomial SCM family (Phase M2).

Ground-truth functionals here are computed in closed form from the mechanism
coefficients, never by Monte Carlo: MC error in the target manifests as
spurious undercoverage downstream, which is exactly what this project measures.

Two distinct exact paths, deliberately kept separate:

- **Interventional** (PATE, CATE): analytic expectation over the outcome
  noise. For an additive-noise mechanism, E[Y | do(t), x] = mu_t(x), and the
  population functional integrates tau over the covariate distribution.
- **Counterfactual** (STATE/SATE, and GATE over a subset): exact abduction.
  The latent outcome noise is recovered from each observed row via the known
  mechanism (u_y = y - f_y(parents)), both interventions are applied with that
  realized noise, and the difference is averaged. SATE conditions on the
  specific units' realized noise; PATE integrates over the population. For
  this additive family u_y cancels in the difference, but the abduction path
  is kept explicit because it is the computation that generalizes to
  non-additive families (and it mirrors what `gcm.counterfactual_samples`
  does, which the tests assert exactly).

The mechanism coefficients cannot be recovered from a built dowhy SCM (they
live in closures), so the prior attaches its spec to the model instance as
``scm.analytic_spec``; the exact query functions read it from there. Query
function signatures match ``mixture._worker_generate_task`` expectations.
"""

from __future__ import annotations

import math
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
    # ---- Section 2 knobs (defaults reproduce the original family exactly).
    # Typed by identification consequence, not implementation:
    #   unobs_noise_coef  (gamma_n): hidden NON-confounder u_zn -> Y only.
    #       Identification untouched; raises irreducible noise. The control.
    #   unobs_conf_coef   (gamma):   hidden confounder u_zc -> T logits AND Y.
    #       Genuine partial identification; moves the across-function term c.
    #   overlap_floor/span/sharpness: p = floor + span*expit(sharpness*logits).
    #       floor -> 0 with sharpness up creates no-support regions (a support
    #       failure, deliberately distinct from confounding).
    # Effect heterogeneity is a sampler dial (effect_scale), not a spec field.
    # All knobs keep exact targets: hidden terms are additive in y, so they
    # cancel in the abducted counterfactual difference (ITE = tau(X) always
    # for this family).
    unobs_noise_coef: float = 0.0
    unobs_conf_coef: float = 0.0
    # Exchangeable covariate correlation r: X_j = sqrt(r) Z + sqrt(1-r) eps_j
    # with Z shared across j. Marginals stay N(0,1), so E[phi] and the exact
    # PATE formula are unchanged; only the Gram matrix conditioning changes.
    # In spec for CausalPFN: nothing in its prior constrains covariate
    # correlation, and no unobserved variable enters T or Y.
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
        """Exact counterfactual ITE by abduction. For this additive family
        the recovered noise cancels in the difference (ITE = tau(X)), but the
        computation stays explicitly counterfactual."""
        u = self.abduct_outcome_noise(X, t, y)
        y_treated = self.mu1(X) + u
        y_control = self.mu0(X) + u
        return y_treated - y_control


@dataclass(frozen=True)
class HeteroskedasticSCMSpec(PolynomialSCMSpec):
    """Non-additive family (M7): treatment x noise interaction.

        lam(x) = lambda_scale * expit(b_l + phi @ w_l)   (in (0, lambda_scale))
        y      = mu0(x) + t * tau(x) + (1 + t * lam(x)) * u,  u ~ N(0, sigma^2)

    ITE = tau(X) + lam(X) * u genuinely depends on the unit's realized noise:
    the within-function variance term is NONZERO and SATE != mean-CATE, so
    the abduction machinery is load-bearing. Abduction stays exact: u = (y - mu0 - t*tau) / (1 + t*lam),
    and 1 + t*lam >= 1 always (lam > 0, t in {0,1}).

    Closed forms survive: CATE(x) = E[ITE|x] = tau(x) (E[u]=0), PATE =
    E[tau(X)] — identical formulas to the additive family; only the
    counterfactual path changes.
    """

    lambda_weights: np.ndarray = None
    lambda_intercept: float = 0.0
    lambda_scale: float = 1.0

    def lam(self, X: np.ndarray) -> np.ndarray:
        logits = self.lambda_intercept + self._basis(X) @ self.lambda_weights
        return self.lambda_scale * expit(logits)

    def abduct_outcome_noise(
        self, X: np.ndarray, t: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        resid = np.asarray(y, dtype=np.float64) - (self.mu0(X) + t * self.tau(X))
        return resid / (1.0 + t * self.lam(X))

    def ite_from_observed(
        self, X: np.ndarray, t: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """y(1) - y(0) = tau(X) + lam(X) * u — noise does NOT cancel."""
        u = self.abduct_outcome_noise(X, t, y)
        return self.tau(X) + self.lam(X) * u


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
    """Per-task coefficient draw (uses global np.random like upstream priors,
    so mixture worker seeding applies). Knob kwargs default to the original
    family; effect_scale is the heterogeneity dial (difficulty, not
    identification). overlap_span shrinks with the floor so p stays in
    [floor, 1 - floor]."""
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


def sample_hetero_spec(
    num_covariates: int,
    outcome_noise_scale: float = 0.25,
    lambda_scale: float = 1.0,
) -> HeteroskedasticSCMSpec:
    if not 1 <= num_covariates <= MAX_COVARIATES:
        raise ValueError(f"num_covariates must be in [1, {MAX_COVARIATES}]")
    two_d = 2 * num_covariates
    return HeteroskedasticSCMSpec(
        baseline_weights=np.random.normal(scale=0.5, size=two_d),
        baseline_intercept=0.0,
        effect_weights=np.random.normal(scale=0.35, size=two_d),
        effect_intercept=0.25,
        propensity_weights=np.random.normal(scale=0.35, size=two_d),
        propensity_intercept=0.0,
        outcome_noise_scale=outcome_noise_scale,
        lambda_weights=np.random.normal(scale=0.35, size=two_d),
        lambda_intercept=0.0,
        lambda_scale=lambda_scale,
    )


def covariate_names(num_covariates: int) -> list[str]:
    return [f"x_{i + 1}" for i in range(num_covariates)]


def build_polynomial_scm(spec: PolynomialSCMSpec) -> gcm.InvertibleStructuralCausalModel:
    """Materialize the spec as a genuine dowhy SCM (the mixture sampler and
    batch builder require one), with the spec attached as ``analytic_spec``.

    Section 2 knobs add hidden ROOT nodes (never listed in
    observed_feature_names): u_zc (confounder: -> t and y) and u_zn
    (non-confounder: -> y only). Node names sort between 't' and 'x_1'
    (dowhy feeds parents lexicographically), and both enter y ADDITIVELY, so
    the abduction difference cancels them and exact targets are unchanged.
    With both knobs at 0 the graph is bit-identical to the original."""
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


def build_hetero_scm(spec: HeteroskedasticSCMSpec) -> gcm.InvertibleStructuralCausalModel:
    """Materialize the heteroskedastic spec as a gcm SCM.

    The multiplicative noise is expressed additively for gcm by promoting u
    to an explicit ROOT NODE ``u_y`` (N(0, sigma)) and making y a
    DETERMINISTIC function of (t, u_y, x_*). Everything downstream (mixture
    sampling, interventional_samples, even gcm counterfactuals — root-node
    abduction is trivial and y has zero own-noise) stays exact. ``u_y`` is
    simply never listed among observed_feature_names.
    """
    d = spec.num_covariates
    x_names = covariate_names(d)
    graph = nx.DiGraph(
        [(x, "t") for x in x_names]
        + [(x, "y") for x in x_names]
        + [("t", "y"), ("u_y", "y")]
    )

    def t_function(parents: np.ndarray) -> np.ndarray:
        # Parents sorted: x_1..x_d
        p = spec.propensity(parents)
        return np.random.binomial(n=1, p=p).astype(np.float64)

    def y_function(parents: np.ndarray) -> np.ndarray:
        # Parents sorted alphabetically: t, u_y, x_1..x_d
        t, u, X = parents[:, 0], parents[:, 1], parents[:, 2:]
        return spec.mu0(X) + t * spec.tau(X) + (1.0 + t * spec.lam(X)) * u

    node_function_dict: dict = {
        name: (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1)) for name in x_names
    }
    node_function_dict["u_y"] = (
        None,
        gcm.ScipyDistribution(stats.norm, loc=0, scale=spec.outcome_noise_scale),
    )
    node_function_dict["t"] = (t_function, None)
    node_function_dict["y"] = (y_function, None)  # deterministic given parents

    scm = build_synthetic_causal_model(
        causal_graph=graph, node_function_dict=node_function_dict
    )
    scm.analytic_spec = spec
    return scm


def make_hetero_prior_fn(
    num_covariates: int,
    outcome_noise_scale: float = 0.25,
    lambda_scale: float = 1.0,
):
    def prior_fn() -> gcm.InvertibleStructuralCausalModel:
        spec = sample_hetero_spec(num_covariates, outcome_noise_scale, lambda_scale)
        return build_hetero_scm(spec)

    return prior_fn


def _get_spec(causal_model: gcm.ProbabilisticCausalModel) -> PolynomialSCMSpec:
    spec = getattr(causal_model, "analytic_spec", None)
    if spec is None:
        raise ValueError(
            "Exact query functions require an SCM with an attached analytic_spec "
            "(build the prior with make_polynomial_prior_fn)."
        )
    return spec


# ---- Exact interventional path (analytic noise integration) ----
#
# Every exact factory tags its query_fn with `exact_path` so downstream
# wiring can enforce WHICH exact computation it consumes (Route A's ELBO
# targets must be counterfactual_abduction — a CATE target would train
# identically on the additive family and silently diverge on non-additive
# ones).

EXACT_PATH_INTERVENTIONAL = "interventional"
EXACT_PATH_COUNTERFACTUAL = "counterfactual_abduction"


def require_counterfactual_target(query_fn) -> None:
    """Structural guard: raise unless query_fn is the abduction primitive."""
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
        # Dispatches on the spec: additive family (noise cancels) or
        # heteroskedastic family (ITE = tau + lam*u, abduction load-bearing).
        return spec.ite_from_observed(X, t_obs, y_obs)[:, np.newaxis]

    query_fn.exact_path = EXACT_PATH_COUNTERFACTUAL
    return query_fn


def make_exact_state_query_fn(outcome_name: str = "y", treatment_name: str = "t"):
    """Exact SATE: mean abducted counterfactual ITE over whatever rows are
    passed in — applied to a subset, this is the exact GATE-over-subset
    primitive (M3)."""
    ite_fn = make_exact_ite_query_fn(outcome_name, treatment_name)

    def query_fn(causal_model, observed_data: pd.DataFrame) -> float:
        return float(ite_fn(causal_model, observed_data).mean())

    query_fn.exact_path = EXACT_PATH_COUNTERFACTUAL
    return query_fn


# ---- MC fallback budget (families without closed forms) ----


def required_mc_samples(
    target_sd: float, expected_half_width: float, budget_frac: float = 0.1
) -> int:
    """Smallest M with SE(M) = target_sd/sqrt(M) below budget_frac of the
    smallest expected interval half-width. Callers must log the chosen M and
    the implied SE alongside the run config."""
    if target_sd <= 0 or expected_half_width <= 0 or budget_frac <= 0:
        raise ValueError("target_sd, expected_half_width, budget_frac must be positive")
    return int(math.ceil((target_sd / (budget_frac * expected_half_width)) ** 2))


def implied_mc_se(target_sd: float, num_mc_samples: int) -> float:
    return target_sd / math.sqrt(num_mc_samples)
