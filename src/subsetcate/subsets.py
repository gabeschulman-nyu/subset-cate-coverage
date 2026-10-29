"""Subset sampler for training on the subset-CATE.

A sampler draws a boolean mask over the context rows of one task; the task's
target is the exact SATE over the masked rows (theta_S, query GATE). Only the
``uniform_random`` backend exists, and it never sees outcomes or effects.
Randomness uses the global ``np.random`` stream, so worker seeding applies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAINING_ALLOWED_BACKENDS = ("uniform_random",)


def sample_log_uniform_k(k_min: int, k_max: int) -> int:
    """Per-task subset size: round(exp(U[log k_min, log k_max])), clipped.

    k = 1 is a single unit's CATE and k = n the full sample.
    """
    if not 1 <= k_min <= k_max:
        raise ValueError(f"Need 1 <= k_min <= k_max, got [{k_min}, {k_max}]")
    k = int(round(np.exp(np.random.uniform(np.log(k_min), np.log(k_max)))))
    return int(np.clip(k, k_min, k_max))


class UniformRandomSubsetSampler:
    """k rows chosen uniformly at random, independent of all columns."""

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


def make_subset_sampler(backend: str, *, training: bool, **kwargs):
    """Build a subset sampler. With ``training=True``, reject any backend not
    in TRAINING_ALLOWED_BACKENDS."""
    if training and backend not in TRAINING_ALLOWED_BACKENDS:
        raise ValueError(
            f"Subset backend {backend!r} is evaluation-only and cannot be used "
            f"in a training configuration (allowed: {TRAINING_ALLOWED_BACKENDS})"
        )
    if backend == "uniform_random":
        return UniformRandomSubsetSampler(**kwargs)
    raise ValueError(f"Unknown subset backend: {backend!r}")
