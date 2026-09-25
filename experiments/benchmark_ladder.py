"""Coverage against the fraction of the sample aggregated, per benchmark.

The subgroup ladders manufactured a range of rho_S by slicing one DGP. The
benchmark suite already spans a wider range on data chosen by other people for
other reasons, so it needs no selection rule defended in a caption. Random
subsets throughout.

The x-axis is k/n rather than k, because the benchmarks differ in sample size
and the comparable quantity is the fraction of the sample being aggregated --
which is also the axis the IHDP number is quoted on in the motivation.

Aggregation is the independent composition under test: per-unit posterior
draws averaged across units at independent draw indices.
"""

from __future__ import annotations

import argparse
import glob
import json

import os

import numpy as np

from subsetcate.design_effect import from_per_unit_errors

from benchmark_targets import target

# Per-unit CausalPFN draws per benchmark replication, as
# {DGP}_seed{NNNNN}.npz with keys samples0/samples1/tau_true.
# Produced by benchmarks/run_causalpfn_pit.py; ~9GB, so not
# shipped. The derived summary this writes IS shipped, so the
# figure and table rebuild without it.
BASE = os.environ.get("PIT_PARTS", "data/pit_calib/parts")
N_SUB = 48
RNG = np.random.default_rng(20260823)


def rho_of(E):
    k = E.shape[1]
    if k < 2:
        return float("nan")
    s1, s2 = E.sum(1), (E ** 2).sum(1)
    return float(np.mean((s1 ** 2 - s2) / (k * (k - 1))) / max(np.mean(E ** 2), 1e-18))


def kstar(ks, cov, thr=0.95):
    ks = np.asarray(ks, float)
    for i in range(len(ks) - 1):
        if cov[i] >= thr > cov[i + 1]:
            f = (cov[i] - thr) / max(cov[i] - cov[i + 1], 1e-12)
            return float(np.exp(np.log(ks[i]) + f * (np.log(ks[i + 1]) - np.log(ks[i]))))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="IHDP,HillA,HillB,KangSchafer,"
                                          "ACIC2016,FriedmanConf,HillBHeavy")
    ap.add_argument("--max_reps", type=int, default=40)
    ap.add_argument("--k_points", type=int, default=15)
    ap.add_argument("--out", default="logs/inspec/benchmark_ladder.json")
    a = ap.parse_args()

    out = {}
    for dgp in a.settings.split(","):
        files = sorted(glob.glob(f"{BASE}/{dgp}_seed*.npz"))[: a.max_reps]
        if not files:
            print(f"{dgp}: no parts, skipping")
            continue
        TAU, E, SD, TRUE, NRMSE_Y = [], [], [], [], []
        for f in files:
            z = np.load(f)
            d = z["samples1"] - z["samples0"]
            # The conditional effect tau(X_i), checked to carry no outcome
            # noise; see benchmark_targets.
            tau = target(dgp, z)
            TAU.append(d)
            E.append(d.mean(axis=1) - tau)
            SD.append(d.std(axis=1))
            TRUE.append(tau)
            # Per-unit RMSE over the spread of the observed outcome, within
            # this dataset: defined even when the effect is constant.
            NRMSE_Y.append(float(np.sqrt(np.mean(E[-1] ** 2)) / z["y"].std()))
        Ea, SDa, TRUEa = np.stack(E), np.stack(SD), np.stack(TRUE)
        T, n = Ea.shape
        sigma2, rho = from_per_unit_errors(Ea)
        m_glob = float(np.mean(SDa ** 2) / sigma2)
        # Scale context: these benchmarks live on different outcome scales, so
        # raw sigma^2 is not comparable across rows. Normalise by the spread of
        # the true effect, as the superseded legacy table did.
        tau_sd = float(TRUEa.std())
        # A constant effect (Hill-A, Kang--Schafer) has no spread to
        # normalise by; the tables use nrmse_y instead.
        nrmse = float(np.sqrt(sigma2) / tau_sd) if tau_sd > 0 else None
        # Is rho resolved or noise? Bootstrap over replications, the
        # independent units here.
        brng = np.random.default_rng(777)
        boot = [from_per_unit_errors(Ea[brng.integers(0, T, size=T)])[1]
                for _ in range(2000)]
        rho_lo, rho_hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

        ks = [1] + [int(k) for k in
                    np.unique(np.geomspace(2, n, a.k_points).astype(int))]
        cov, se, rho_k, m_k = [], [], [], []
        # (centre, half-width, target) per interval, so the rescaled and
        # design-effect intervals of Appendix C come from the SAME intervals.
        # Storing them consumes no randomness.
        ivals = {k: [] for k in ks}
        for k in ks:
            per_task, Ek, SDk = [], [], []
            for i in range(T):
                hits = []
                reps = 1 if k == n else N_SUB
                for _ in range(reps):
                    idx = (np.arange(n) if k == n
                           else RNG.choice(n, size=k, replace=False))
                    d = TAU[i][idx]
                    sel = RNG.integers(0, d.shape[1], size=d.shape)
                    agg = np.take_along_axis(d, sel, axis=1).mean(axis=0)
                    lo, hi = np.quantile(agg, [0.025, 0.975])
                    hits.append(lo <= TRUE[i][idx].mean() <= hi)
                    ivals[k].append((0.5 * (lo + hi), 0.5 * (hi - lo),
                                     TRUE[i][idx].mean()))
                per_task.append(float(np.mean(hits)))
                Ek.append(Ea[i, idx])
                SDk.append(SDa[i, idx])
            cov.append(float(np.mean(per_task)))
            se.append(float(np.std(per_task, ddof=1) / np.sqrt(len(per_task))))
            Ek, SDk = np.stack(Ek), np.stack(SDk)
            rho_k.append(rho_of(Ek))
            m_k.append(float(np.mean(SDk ** 2) / max(np.mean(Ek ** 2), 1e-18)))
        kx = kstar(ks, cov)
        # The m_S and rho_S this benchmark reports (Table legacy) are medians
        # over k; the corrections use those same two numbers at every k.
        m_rep = float(np.nanmedian(m_k))
        rho_rep = float(np.nanmedian(rho_k))

        def _cov(fac_of_k):
            return [float(np.mean([abs(t - c) <= h * fac_of_k(k)
                                   for c, h, t in ivals[k]])) for k in ks]

        cov_recal = _cov(lambda k: 1.0 / np.sqrt(max(m_rep, 1e-12)))
        cov_full = _cov(lambda k: np.sqrt(max(1.0 + (k - 1) * rho_rep, 1e-12)
                                          / max(m_rep, 1e-12)))
        out[dgp] = {"target": "conditional effect", "n": n, "reps": T, "ks": ks,
                    "coverage": cov,
                    "cov_se_clustered": se, "rho_by_k": rho_k, "m_by_k": m_k,
                    "rho_full": rho, "m_global": m_glob,
                    "tau_sd": tau_sd, "nrmse_unit": nrmse,
                    "rmse_unit": float(np.sqrt(sigma2)),
                    "nrmse_y": float(np.mean(NRMSE_Y)),
                    "rho_ci": [rho_lo, rho_hi],
                    "cov_recal": cov_recal, "cov_full": cov_full,
                    "breaking_k": kx}
        rb = float(np.nanmedian(rho_k))
        mb = float(np.nanmedian(m_k))
        ks_ = f"{kx:.0f}" if kx == kx else "none"
        kn_ = f"{kx / n:.3f}" if kx == kx else "  ---"
        print(f"{dgp:14s} n={n:5d} reps={T:3d} rho_S={rb:.4f} m_S={mb:7.2f} "
              f"k*={ks_:>5s} k*/n={kn_:>6s} cov@n={cov[-1]:.3f}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
