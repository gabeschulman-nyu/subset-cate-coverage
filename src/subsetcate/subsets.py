"""Subset samplers for the subset-GATE curriculum (Phase M3).

A subset sampler draws a boolean membership mask over the context rows of one
task; the task's target is then the exact SATE over the masked rows
(GATE-over-subset, computed by the M2 abduction primitive).

Training/evaluation separation is a hard invariant, enforced twice:
`covariate_correlated` samplers carry ``allowed_in_training = False``, and the
``make_subset_sampler(..., training=True)`` factory — the only construction
path used by the training pipeline — raises for any backend that is not
``uniform_random``. Correlated subsets exist solely for the held-out
functional-transfer evaluation (M7).

Samplers select rows from covariates/propensity only; they never see
outcomes or effects (no target leakage). Randomness uses global ``np.random``
so mixture worker seeding applies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from subsetcate.exact import covariate_names

TRAINING_ALLOWED_BACKENDS = ("uniform_random",)
CORRELATED_STRATEGIES = ("propensity_stratum", "covariate_sign", "covariate_cell")


def sample_log_uniform_k(k_min: int, k_max: int) -> int:
    """Per-task subset size: round(exp(U[log k_min, log k_max])), clipped.

    k=1 recovers CATE-at-a-unit; k=n recovers the full-sample SATE; log
    spacing covers both scales densely (a distributional curriculum — online
    dataloader workers have no global-step signal for a ramp).
    """
    if not 1 <= k_min <= k_max:
        raise ValueError(f"Need 1 <= k_min <= k_max, got [{k_min}, {k_max}]")
    k = int(round(np.exp(np.random.uniform(np.log(k_min), np.log(k_max)))))
    return int(np.clip(k, k_min, k_max))


class UniformRandomSubsetSampler:
    """Training-allowed backend: k rows chosen independently of all columns."""

    allowed_in_training = True
    backend = "uniform_random"

    def __init__(self, k_min: int = 1, k_max: int | None = None) -> None:
        if k_min < 1:
            raise ValueError("k_min must be >= 1")
        self.k_min = k_min
        self.k_max = k_max  # None -> n_ctx at sample time

    def sample_mask(self, observed_df: pd.DataFrame, causal_model=None) -> np.ndarray:
        n = len(observed_df)
        k_max = n if self.k_max is None else min(self.k_max, n)
        k = sample_log_uniform_k(min(self.k_min, k_max), k_max)
        mask = np.zeros(n, dtype=bool)
        mask[np.random.choice(n, size=k, replace=False)] = True
        return mask


class CovariateCorrelatedSubsetSampler:
    """Evaluation-only backend: subsets aligned with covariate structure.

    Strategies:
    - ``propensity_stratum``: rows in a random quartile of the exact
      propensity (requires ``scm.analytic_spec``).
    - ``covariate_sign``: rows with a random covariate above/below zero.
    - ``covariate_cell``: conjunction of two independent sign splits.
    """

    allowed_in_training = False
    backend = "covariate_correlated"

    def __init__(self, strategy: str = "covariate_sign") -> None:
        if strategy not in CORRELATED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}; choose from {CORRELATED_STRATEGIES}"
            )
        self.strategy = strategy

    def _covariate_matrix(self, observed_df: pd.DataFrame, causal_model) -> np.ndarray:
        spec = getattr(causal_model, "analytic_spec", None)
        if spec is not None:
            return observed_df[covariate_names(spec.num_covariates)].values
        x_cols = [c for c in observed_df.columns if c.startswith("x_")]
        if not x_cols:
            raise ValueError("No covariate columns found for correlated subset sampling")
        return observed_df[x_cols].values

    def sample_mask(self, observed_df: pd.DataFrame, causal_model=None) -> np.ndarray:
        X = self._covariate_matrix(observed_df, causal_model)
        n, d = X.shape

        if self.strategy == "propensity_stratum":
            spec = getattr(causal_model, "analytic_spec", None)
            if spec is None:
                raise ValueError("propensity_stratum requires an SCM with analytic_spec")
            p = spec.propensity(X)
            lo = float(np.random.choice([0.0, 0.25, 0.5, 0.75]))
            q_lo, q_hi = np.quantile(p, [lo, lo + 0.25])
            mask = (p >= q_lo) & (p <= q_hi)
        elif self.strategy == "covariate_sign":
            j = np.random.randint(d)
            mask = X[:, j] > 0 if np.random.rand() < 0.5 else X[:, j] <= 0
        else:  # covariate_cell
            j1, j2 = np.random.randint(d), np.random.randint(d)
            s1 = X[:, j1] > 0 if np.random.rand() < 0.5 else X[:, j1] <= 0
            s2 = X[:, j2] > 0 if np.random.rand() < 0.5 else X[:, j2] <= 0
            mask = s1 & s2

        if not mask.any():
            mask = np.zeros(n, dtype=bool)
            mask[np.random.randint(n)] = True
        return np.asarray(mask, dtype=bool)


def make_subset_sampler(backend: str, *, training: bool, **kwargs):
    """Single construction path for pipelines. Hard-rejects any backend that
    is not training-allowed when ``training=True`` — this is the invariant
    that keeps correlated subsets out of every training config."""
    if training and backend not in TRAINING_ALLOWED_BACKENDS:
        raise ValueError(
            f"Subset backend {backend!r} is evaluation-only and cannot be used "
            f"in a training configuration (allowed: {TRAINING_ALLOWED_BACKENDS})"
        )
    if backend == "uniform_random":
        return UniformRandomSubsetSampler(**kwargs)
    if backend == "covariate_correlated":
        return CovariateCorrelatedSubsetSampler(**kwargs)
    raise ValueError(f"Unknown subset backend: {backend!r}")
