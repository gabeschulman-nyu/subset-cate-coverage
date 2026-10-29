"""Stage 3 (subsetcate env): rho, m, sigma^2, coverage against k and breaking k.

Unit-level aggregation as in CausalPFN: each aggregate draw averages per-unit
draws at independent indices, and the interval is the empirical 2.5/97.5
quantiles. A moment-based Gaussian interval is reported alongside. Writes
res_<tag>.json (Tables 2, 3 and 6, Figure 4).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from subsetcate.design_effect import from_per_unit_errors

RNG = np.random.default_rng(20260801)
N_SUB = 24


def first_below(ks, cov, thr=0.95):
    for a in range(len(ks) - 1):
        if cov[a] >= thr > cov[a + 1]:
            f = (cov[a] - thr) / max(cov[a] - cov[a + 1], 1e-12)
            return float(np.exp(np.log(ks[a]) + f * (np.log(ks[a + 1]) - np.log(ks[a]))))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ks", type=str, default="",
                    help="explicit comma-separated k grid; default is 9-point "
                         "geomspace(2, n). The default wastes most of its "
                         "points where coverage is pinned at 1.0, so figures "
                         "pass a grid concentrated near the crossing.")
    ap.add_argument("--suffix", type=str, default="",
                    help="appended to the output filename so a denser grid "
                         "does not overwrite the canonical Section 1 result")
    ap.add_argument("--n_sub", type=int, default=N_SUB)
    a = ap.parse_args()
    tk = np.load(f"logs/inspec/tasks_{a.tag}.npz")
    dr = np.load(f"logs/inspec/draws_{a.tag}.npz")["draws"]
    sy = tk["sigma_y"][:, None]
    n = int(tk["n"])
    T, N, S = dr.shape

    tauhat = dr.mean(axis=2)
    delta = (tauhat - tk["ite"]) / sy                      # standardized per-unit errors
    sigma2, rho = from_per_unit_errors(delta)
    sig_unit = dr.std(axis=2) / sy
    m = float(np.mean(sig_unit**2) / sigma2)               # per-unit dispersion ratio

    if a.ks:
        ks = sorted({min(int(v), n) for v in a.ks.split(",")})
    else:
        ks = [int(k) for k in np.unique(np.geomspace(2, n, 9).astype(int))]
    # Coverage is also recorded per task, for standard errors clustered on
    # task: subsets within a task share its posterior draws.
    cov_draw, cov_mom, width, cov_by_task, rmse = [], [], [], [], []
    # (centre, half-width, target) of every percentile interval, per k, for
    # the rescaled and design-effect intervals of Appendix C. Draws no random
    # numbers.
    ivals = {k: [] for k in ks}
    for k in ks:
        hit_d, hit_m, w, per_task, err = [], [], [], [], []
        for i in range(T):
            task_hits = []
            for _ in range(a.n_sub):
                idx = RNG.choice(N, size=k, replace=False)
                sub = dr[i, idx]                                   # [k, S]
                sel = RNG.integers(0, S, size=(k, S))
                agg = np.take_along_axis(sub, sel, axis=1).mean(axis=0) / sy[i, 0]
                lo, hi = np.quantile(agg, [0.025, 0.975])
                truth = tk["ite"][i, idx].mean() / sy[i, 0]
                task_hits.append(lo <= truth <= hi)
                hit_d.append(lo <= truth <= hi)
                w.append(hi - lo)
                ivals[k].append((0.5 * (lo + hi), 0.5 * (hi - lo), truth))
                sd_m = np.sqrt((sig_unit[i, idx] ** 2).sum()) / k
                point = tauhat[i, idx].mean() / sy[i, 0]
                hit_m.append(abs(point - truth) <= 1.96 * sd_m)
                err.append(point - truth)
            per_task.append(float(np.mean(task_hits)))
        cov_by_task.append(per_task)
        cov_draw.append(float(np.mean(hit_d)))
        cov_mom.append(float(np.mean(hit_m)))
        width.append(float(np.mean(w)))
        # oracle_width = 2 * 1.96 * rmse, the width of a calibrated Gaussian
        # interval for the errors observed at this k.
        rmse.append(float(np.sqrt(np.mean(np.asarray(err) ** 2))))

    def _cov(fac_of_k):
        return [float(np.mean([abs(t - c) <= h * fac_of_k(k)
                               for c, h, t in ivals[k]])) for k in ks]

    # cov_recal: unit-level variances divided by m. cov_full: also multiplied
    # by 1 + (k-1) rho.
    cov_recal = _cov(lambda k: 1.0 / np.sqrt(max(m, 1e-12)))
    cov_full = _cov(lambda k: np.sqrt(max(1.0 + (k - 1) * rho, 1e-12) / max(m, 1e-12)))

    out = {"tag": a.tag, "n": n, "corr": float(tk["corr"]), "n_sub": a.n_sub,
           "overlap": [float(v) for v in tk["overlap"]],
           "tasks": T, "sigma2": sigma2, "rho": rho, "m": m,
           "ks": ks, "coverage_draw": cov_draw, "coverage_moment": cov_mom,
           "width": width, "rmse": rmse,
           "oracle_width": [2 * 1.96 * r for r in rmse],
           "cov_by_task": cov_by_task,
           "cov_se_clustered": [float(np.std(c, ddof=1) / np.sqrt(len(c)))
                                for c in cov_by_task],
           "cov_recal": cov_recal, "cov_full": cov_full,
           "breaking_k_draw": first_below(ks, cov_draw),
           "breaking_k_moment": first_below(ks, cov_mom),
           "propensity_min": float(tk["propensity"].min()),
           "propensity_max": float(tk["propensity"].max())}
    with open(f"logs/inspec/res_{a.tag}{a.suffix}.json", "w") as f:
        json.dump(out, f, indent=2)
    kb = out["breaking_k_draw"]
    print(f"{a.tag}: n={n} r={out['corr']} rho={rho:.4f} m={m:.1f} sigma2={sigma2:.4f} "
          f"k*={kb:.0f}" if kb == kb else f"{a.tag}: n={n} rho={rho:.4f} m={m:.1f} k*=none")
    print("   cov:", "  ".join(f"{k}:{c:.2f}" for k, c in zip(ks, cov_draw, strict=True)))


if __name__ == "__main__":
    main()
