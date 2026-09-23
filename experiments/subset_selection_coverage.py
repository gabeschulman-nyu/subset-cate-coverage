"""Coverage vs k for RANDOM subsets against TARGETED subgroups, in-spec DGP.

Same model, same tasks, same estimator, same nominal level. The only thing that
changes is how the subset is chosen -- which is the practitioner's choice, not a
property of the data-generating process. A practitioner asking for a subset-GATE
is almost never asking about a uniformly random subset; they are asking about a
subgroup defined by exactly the kind of variable (propensity-adjacent covariates)
that concentrates the shared error component.

Subgroup pools are fixed per task and independent of k, so sweeping k varies
only the subset size within a fixed subgroup. An earlier version sized the pool
at 2k, which made the "stratum" converge to the full sample as k grew and
produced spurious coverage recovery.

Errors are clustered on task: subsets drawn within a task share that task's
posterior draws and its shared error, so pooling all resamples into one binomial
understates the interval several-fold.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import norm

_Phi = norm.cdf

RNG = np.random.default_rng(20260802)
N_SUB = 48
MODES = ("random", "propensity stratum", "covariate cell", "x1_quartile")


def rho_of(E):
    k = E.shape[1]
    s1, s2 = E.sum(1), (E**2).sum(1)
    return float(np.mean((s1**2 - s2) / (k * (k - 1))) / max(np.mean(E**2), 1e-18))


def first_below(ks, cov, thr=0.95):
    for a in range(len(ks) - 1):
        if cov[a] >= thr > cov[a + 1]:
            f = (cov[a] - thr) / max(cov[a] - cov[a + 1], 1e-12)
            return float(np.exp(np.log(ks[a]) + f * (np.log(ks[a + 1]) - np.log(ks[a]))))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="hi_clean,hi_ovsoft")
    ap.add_argument("--out", default="logs/inspec/subset_selection.json")
    # Matched-n figures need the SAME k grid as the trained-arm cells, which
    # use geomspace(2, n, 9). Default keeps the old per-pool grid.
    ap.add_argument("--match_stage3_grid", action="store_true")
    # Ladder mode: one rule (bottom pool_frac of the true propensity) swept
    # over pool_frac, so rho_S moves continuously with the DGP, the model and
    # the estimator all untouched. Mirrors --pool_frac in two_path_ablation
    # so the trained-arm panel and this one select identically.
    ap.add_argument("--pool_fracs", default=None,
                    help="comma-separated pool fractions, e.g. 1.00,0.50,0.25,0.10")
    ap.add_argument("--rank_by", default="propensity",
                    choices=["propensity", "effect", "effect_oracle",
                             "neighbor_ball", "nested_cell"],
                    help="rank units by ascending true propensity, or by "
                         "descending predicted effect (feasible at test time)")
    ap.add_argument("--dense_grid", action="store_true",
                    help="use [1] + geomspace(2, N, 17), matching the ladder cells")
    a = ap.parse_args()
    out = {}
    for tag in a.tags.split(","):
        tk = np.load(f"logs/inspec/tasks_{tag}.npz")
        dr = np.load(f"logs/inspec/draws_{tag}.npz")["draws"]
        sy, prop, ite, X = tk["sigma_y"], tk["propensity"], tk["ite"], tk["X"]
        T, N, S = dr.shape

        fracs = ([float(v) for v in a.pool_fracs.split(",")]
                 if a.pool_fracs else None)
        # tau_hat is the shipped model's own posterior mean per unit, so
        # ranking on it is something an analyst can actually do at test time.
        tau_hat = dr.mean(axis=2)
        rank_src = {"propensity": prop, "effect": -tau_hat, "effect_oracle": -ite}
        pools = {}
        for i in range(T):
            if a.rank_by == "nested_cell":
                # Increasingly specific ordinary subgroups: "men", "men over
                # forty", "men over forty with condition Z". Each conjunct
                # roughly halves the pool, so the ladder needs no tuning and
                # no construction -- which is the point. The cells are the
                # first three covariates at their natural cut of zero; they
                # are NOT chosen to maximise rho_S.
                X_i = X[i]
                pools[i] = {"all units": np.arange(N)}
                m_run = np.ones(N, dtype=bool)
                for d in range(3):
                    m_run = m_run & (X_i[:, d] > 0)
                    key = ", ".join(f"$X_{e+1}$>0" for e in range(d + 1))
                    pools[i][key] = np.where(m_run)[0]
            elif fracs is not None:
                if a.rank_by == "neighbor_ball":
                    anchor = RNG.integers(N)
                    order = np.argsort(((X[i] - X[i][anchor]) ** 2).sum(axis=1))
                else:
                    order = np.argsort(rank_src[a.rank_by][i])
                pools[i] = {f"{a.rank_by} {f:.2f}":
                            order[: max(2, int(round(N * f)))] for f in fracs}
            else:
                q = np.argsort(prop[i])[: N // 4]          # bottom propensity quartile
                m = ((X[i][:, 0] > np.median(X[i][:, 0]))
                     & (X[i][:, 1] > np.median(X[i][:, 1])))
                # x1_quartile matches the rule the trained-arm correlated cells
                # use, so the two panels select identically.
                pools[i] = {"random": np.arange(N), "propensity stratum": q,
                            "covariate cell": np.where(m)[0],
                            "x1_quartile": np.argsort(-X[i][:, 0])[: N // 4]}

        modes = (list(pools[0]) if (fracs is not None
                                    or a.rank_by == "nested_cell") else MODES)
        res = {}
        for mode in modes:
            kmax = min(len(pools[i][mode]) for i in range(T))
            if a.dense_grid:
                ks = [1] + [int(k) for k in
                            np.unique(np.geomspace(2, N, 17).astype(int))
                            if k <= kmax]
            elif a.match_stage3_grid:
                ks = [int(k) for k in np.unique(np.geomspace(2, N, 9).astype(int))
                      if k <= kmax]
            else:
                ks = [int(k) for k in np.unique(np.geomspace(4, kmax, 12).astype(int))]
            # m is pool-specific, not a global constant: the same checkpoint
            # can be more or less conservative inside different subgroups, and
            # R = m/[1+(k-1)rho_S] depends on both. Reporting rho_S alone
            # makes a rule with high rho_S but high m look like a counter-
            # example when it is not.
            cov, se, rho_k, by_task, m_k = [], [], [], [], []
            # Intervals are kept rather than collapsed to hits, so the
            # post-hoc design-effect correction can be applied afterwards
            # using a rho_S pooled over k instead of the noisy per-k estimate.
            keep = []
            for k in ks:
                per_task, E, SD, ivals = [], [], [], []
                for i in range(T):
                    pool = pools[i][mode]
                    if len(pool) < k:
                        continue
                    hits, tri = [], []
                    for _ in range(N_SUB):
                        idx = RNG.choice(pool, size=k, replace=False)
                        sub = dr[i, idx]
                        sel = RNG.integers(0, S, size=(k, S))
                        agg = np.take_along_axis(sub, sel, axis=1).mean(axis=0) / sy[i]
                        lo, hi = np.quantile(agg, [0.025, 0.975])
                        truth = ite[i, idx].mean() / sy[i]
                        hits.append(lo <= truth <= hi)
                        tri.append((0.5 * (lo + hi), 0.5 * (hi - lo), truth))
                    per_task.append(float(np.mean(hits)))
                    ivals.append(np.array(tri))
                    E.append((dr[i, idx].mean(axis=1) - ite[i, idx]) / sy[i])
                    SD.append(dr[i, idx].std(axis=1) / sy[i])
                cov.append(float(np.mean(per_task)))
                se.append(float(np.std(per_task, ddof=1) / np.sqrt(len(per_task))))
                rho_k.append(rho_of(np.stack(E)) if len(E) > 1 else float("nan"))
                m_k.append(float(np.mean(np.stack(SD) ** 2)
                                 / max(np.mean(np.stack(E) ** 2), 1e-18))
                           if len(E) > 1 else float("nan"))
                by_task.append(per_task)
                keep.append((k, ivals))

            # Post-hoc correction: widen the half-width by the design-effect
            # factor sqrt(1 + (k-1) rho_S). That turns R = m_S/[1+(k-1)rho_S]
            # into exactly m_S, so PREDICTED corrected coverage is
            # 2*Phi(z*sqrt(m_S)) - 1 -- k-invariant but NOT nominal. On the
            # clean family m_S is large, so it should pin near 1.000; only
            # where m_S ~ 1 should it land near 0.95. Residual slope in k
            # would mean the design effect does not account for all of it.
            rho_bar = float(np.nanmedian(rho_k))
            m_bar = float(np.nanmedian(m_k))

            # Three constructions of the same interval, all rescalings of the
            # stored half-width, so no model is re-run:
            #   A  as reported                       factor 1
            #   B  per-unit recalibration only       factor 1/sqrt(m_S)
            #   C  recalibration + design effect     factor sqrt(Deff/m_S)
            # Their variance ratios are R_A = m_S/Deff, R_B = 1/Deff, R_C = 1.
            # So B falls FASTER than A whenever m_S > 1: recalibration removes
            # the very cushion that was carrying A to large k, while leaving
            # the aggregation untouched. C is nominal under a Gaussian
            # standardized error.
            def _cov(fac_of_k):
                out = []
                for k, ivals in keep:
                    f = fac_of_k(k)
                    pt = [float(np.mean(np.abs(a[:, 2] - a[:, 0]) <= a[:, 1] * f))
                          for a in ivals]
                    out.append(float(np.mean(pt)) if pt else float("nan"))
                return out

            def _deff(k):
                return max(1.0 + (k - 1) * rho_bar, 1e-12)

            rm = max(m_bar, 1e-12)
            cov_corr = _cov(lambda k: np.sqrt(_deff(k)))          # design effect only
            cov_recal = _cov(lambda k: 1.0 / np.sqrt(rm))          # arm B
            cov_full = _cov(lambda k: np.sqrt(_deff(k) / rm))      # arm C
            res[mode] = {"ks": ks, "coverage": cov, "cov_se_clustered": se,
                         "rho_by_k": rho_k, "m_by_k": m_k,
                         "cov_corrected": cov_corr,
                         "cov_recal": cov_recal, "cov_full": cov_full,
                         "rho_bar": rho_bar,
                         "m_bar": m_bar,
                         "cov_pred_corrected":
                             float(2 * _Phi(1.96 * np.sqrt(max(m_bar, 0))) - 1),
                         "cov_by_task": by_task,
                         "breaking_k": first_below(ks, cov)}
            kb = res[mode]["breaking_k"]
            kbs = f"{kb:.0f}" if kb == kb else "none"
            mid = len(rho_k) // 2
            print(f"{tag:10s} {mode:20s} rho_S={rho_k[mid]:.4f} m={m_k[mid]:6.2f}"
                  f"  k*={kbs}")
            print("           cov " + "  ".join(
                f"{k}:{c:.2f}" for k, c in zip(ks, cov, strict=True)))
            def _ends(v):
                w = [x for x in v if x == x]
                return (w[0], w[-1]) if w else (float("nan"),) * 2

            lk = np.log(np.array(ks, float))
            for lab, v in (("A as reported", cov), ("B recalibrated", cov_recal),
                           ("C recal+deff", cov_full)):
                a0, a1 = _ends(v)
                va = np.array(v, float)
                ok = ~np.isnan(va)
                sl = float(np.polyfit(lk[ok], va[ok], 1)[0]) if ok.sum() > 1 else float("nan")
                print(f"           {lab:15s} {a0:.3f} -> {a1:.3f}   slope {sl:+.4f}")
        out[tag] = res

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
