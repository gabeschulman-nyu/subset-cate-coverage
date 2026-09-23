"""Section 1 stage 1 (metadentify env): in-spec task generation.

All settings keep CausalPFN's assumptions: no unobserved confounding, positivity
holds (propensity bounded away from 0/1), exact targets. Dials are sample size n,
exchangeable covariate correlation r, and overlap strength.
"""
from __future__ import annotations

import argparse

import numpy as np

from metadentify.exact import make_polynomial_prior_fn
from metadentify.mixture import (
    CausalMixtureDistribution,
    set_worker_random_seed,
    set_worker_single_thread,
)

EVAL_SEED = 20260801
D = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--corr", type=float, default=0.0)
    ap.add_argument("--overlap_floor", type=float, default=0.05)
    ap.add_argument("--overlap_sharpness", type=float, default=1.0)
    ap.add_argument("--num_tasks", type=int, default=64)
    a = ap.parse_args()

    prior = make_polynomial_prior_fn(
        D, covariate_corr=a.corr,
        overlap_floor=a.overlap_floor, overlap_sharpness=a.overlap_sharpness,
    )
    x_cols = [f"x_{i+1}" for i in range(D)]
    T, N = a.num_tasks, a.n
    X = np.zeros((T, N, D), dtype=np.float32)
    t = np.zeros((T, N), dtype=np.float32)
    y = np.zeros((T, N), dtype=np.float32)
    ite = np.zeros((T, N), dtype=np.float32)
    sig = np.zeros(T)
    prop = np.zeros((T, N), dtype=np.float32)

    for i in range(T):
        set_worker_single_thread()
        set_worker_random_seed(EVAL_SEED + 7919 * i)
        scm = prior()
        mix = CausalMixtureDistribution(causal_model=scm)
        mix.add_observational(weight=1.0)
        df = mix.sample_batch(total_samples=N)
        spec = scm.analytic_spec
        Xi = df[x_cols].values
        X[i], t[i], y[i] = Xi, df["t"].values, df["y"].values
        ite[i] = spec.ite_from_observed(Xi, df["t"].values, df["y"].values)
        sy = float(df[["y"]].values.std()) or 1.0
        sig[i] = sy
        prop[i] = spec.propensity(Xi)

    np.savez_compressed(
        f"logs/inspec/tasks_{a.tag}.npz", X=X, t=t, y=y, ite=ite, sigma_y=sig,
        propensity=prop, n=N, corr=a.corr,
        overlap=(a.overlap_floor, a.overlap_sharpness),
    )
    print(f"{a.tag}: {T} tasks at n={N}, r={a.corr}, "
          f"overlap=({a.overlap_floor},{a.overlap_sharpness}), "
          f"propensity range [{prop.min():.3f}, {prop.max():.3f}]")


if __name__ == "__main__":
    main()
