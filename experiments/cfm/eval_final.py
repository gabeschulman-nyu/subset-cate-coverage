"""One consolidated evaluation for the paper: both aggregations of every head, and
CausalPFN's own, on the same subsets, with per-dataset outcomes kept for
clustered standard errors.

For one method (a head checkpoint, or CausalPFN itself) on the paper's six
benchmarks, 40 datasets each, subsets on a k grid of 16 sizes from 1 to n (48
random subsets per size, one at k = n; one fixed generator seed and order, so
every method is scored on the same subsets):

    joint   CNP head: its direct subset-level Gaussian.
            Latent head: the SCM-marginalized interval (units share each draw of
            z; 500 draws, per-unit values sampled from the histograms).
    unit    the method's own per-unit predictive distributions aggregated as if
            independent.
            CNP: queried at |S| = 1 for every unit; Gaussian with mean = average
            of unit means, variance = sum of unit variances / k^2.
            Latent: each unit gets its own draw of z (the paper's
            functional_sampling independent baseline).
            CausalPFN: one draw per unit from its histogram, averaged; 1,000
            repetitions; at temperature 1 and at the temperature its
            calibration fits (or, with --paper_temperatures, the one the
            paper's Figure 1 run fitted and stored; the fit is stochastic).

Per dataset and size it records coverage and mean interval width (sd(y)
units). Per unit it records the method's point estimate and predictive sd
(CNP: the |S| = 1 query; latent: the marginal over z; CausalPFN: its histogram
at that temperature), from which m_S, rho_S and nRMSE (per-unit RMSE / sd(y),
within each dataset, averaged over datasets) follow. m_S and rho_S are computed
as benchmark_ladder.py computes the paper's: errors in raw outcome units pooled
over datasets, per-size values from ONE subset per dataset (the last one drawn
at that size; the ladder keeps only the last `idx` of its loop), median over
sizes. Kept alongside: the same from all subsets ("_all"; same expectation, far
less noise), and from all subsets with each dataset's errors in its own sd(y)
units ("_std"; this weights datasets as coverage does). Monte Carlo draws are seeded per
(benchmark, dataset), so reruns reproduce exactly.

    SIM_LIB_DIR=... python experiments/cfm/eval_final.py --method v2_lathist_s0=checkpoints/cfm/v2_lathist_s0.pt --out logs/cfm/final/v2_lathist_s0.json
    SIM_LIB_DIR=... python experiments/cfm/eval_final.py --method causalpfn --paper_temperatures $PIT_PARTS --out logs/cfm/final/causalpfn_paperT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SIM_LIB_DIR", "benchmarks"))
from causalpfn_embed import cate_embeddings, load_causalpfn  # noqa: E402
from sim_lib import DGPS  # noqa: E402
from subset_head import make_head, per_unit, unit_features, unit_level_interval  # noqa: E402

N_SUB = 48
BENCHMARKS = ("HillA", "HillB", "FriedmanConf", "KangSchafer", "IHDP", "ACIC2016")


def causalpfn_moments(e, bins, temperature, s_y):
    """CausalPFN's per-unit effect mean and sd (sd(y) units) at a temperature."""
    def mom(logits, scale, shift):
        p = torch.softmax(logits / temperature, -1)
        m = (p * bins).sum(-1)
        v = ((p * bins ** 2).sum(-1) - m ** 2).clamp(min=0)
        return m * scale + shift, v * scale ** 2
    m0, v0 = mom(e["logits0"], e["scale0"], e["shift0"])
    m1, v1 = mom(e["logits1"], e["scale1"], e["shift1"])
    return (m1 - m0) / s_y, torch.sqrt(v0 + v1) / s_y


class Accum:
    """Running sums for the paper's per-size m and rho estimators."""

    def __init__(self):
        self.pair, self.rows, self.e2, self.sd2, self.elems = 0.0, 0, 0.0, 0.0, 0

    def add(self, E, SD):                       # E, SD: [rows, k]
        k = E.shape[1]
        if k >= 2:
            s1, s2 = E.sum(1), (E ** 2).sum(1)
            self.pair += float(((s1 ** 2 - s2) / (k * (k - 1))).sum())
            self.rows += E.shape[0]
        self.e2 += float((E ** 2).sum())
        self.sd2 += float((SD ** 2).sum())
        self.elems += E.numel()

    def rho(self):
        return self.pair / self.rows / max(self.e2 / self.elems, 1e-18) if self.rows else float("nan")

    def m(self):
        return self.sd2 / max(self.e2, 1e-18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="name=checkpoint, or 'causalpfn'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--k_points", type=int, default=15)
    ap.add_argument("--paper_temperatures", default="",
                    help="CausalPFN only: directory of the paper's per-dataset parts; use their stored temperatures")
    a = ap.parse_args()
    dev = torch.device("cuda")
    icl = load_causalpfn("cuda")
    bins = icl.bin_centers
    if a.method == "causalpfn":
        name, head, kind = "causalpfn", None, "causalpfn"
        aggs = ("unit_T1", "unit_cal")
        from causalpfn import CATEEstimator
    else:
        name, path = a.method.split("=", 1)
        st = torch.load(path, map_location=dev, weights_only=False)
        head, kind = make_head(st["args"])
        head.load_state_dict(st["head"])
        head = head.to(dev).eval()
        aggs = ("joint", "unit")
    gen = torch.Generator().manual_seed(20260926)          # the subsets, shared by every method
    out = {"method": name, "kind": kind, "aggregations": list(aggs)}
    for bi, dgp in enumerate(BENCHMARKS):
        ks, acc, acc_all, acc_std = None, None, None, None
        cov = {g: [] for g in aggs}
        wid = {g: [] for g in aggs}
        nrmse, temps = [], []
        for seed in range(1, a.reps + 1):
            sim = DGPS[dgp](seed=seed)
            X, t, y = sim.X.astype(np.float32), sim.z.astype(np.float32), sim.y.astype(np.float32)
            n, s_y = len(X), float(y.std())
            if ks is None:
                ks = [1] + [int(k) for k in np.unique(np.geomspace(2, n, a.k_points).astype(int))]
                acc = {k: Accum() for k in ks}             # the paper's estimator: last subset per dataset
                acc_all = {k: Accum() for k in ks}         # every subset
                acc_std = {k: Accum() for k in ks}         # every subset, sd(y) units
            e = cate_embeddings(icl, X, t, y)
            tau = torch.as_tensor(sim.tau / s_y, dtype=torch.float32, device=dev)
            torch.manual_seed(1000 * bi + seed)
            ugen = torch.Generator(device=dev).manual_seed(1000 * bi + seed)
            if kind == "causalpfn":
                if a.paper_temperatures:
                    temp = float(np.load(f"{a.paper_temperatures}/{dgp}_seed{seed:05d}.npz")["temperature"])
                else:
                    cal = CATEEstimator(device="cuda", verbose=False, calibrate=True)
                    cal.fit(X, t, y)
                    temp = float(cal.temperature)
                temps.append(temp)
                mu_i, sd_i = causalpfn_moments(e, bins, temp, s_y)     # per-unit moments at the paper's setting
            else:
                u = unit_features(e, s_y, bins)
                with torch.no_grad():
                    mu_i, sd_i = per_unit(head, kind, u)
            E_all = mu_i - tau
            nrmse.append(float(torch.sqrt((E_all ** 2).mean())))
            E_raw, SD_raw = (E_all * s_y).double().cpu(), (sd_i * s_y).double().cpu()   # pooled in outcome units
            row_cov = {g: [] for g in aggs}
            row_wid = {g: [] for g in aggs}
            for k in ks:
                mask = (torch.rand(1 if k == n else N_SUB, n, generator=gen).argsort(1).argsort(1) < k).to(dev)
                mk = mask.float()
                theta = (mk @ tau) / k
                ints = {}
                with torch.no_grad():
                    if kind == "causalpfn":
                        for g, T in (("unit_T1", 1.0), ("unit_cal", temp)):
                            lo, hi = unit_level_interval(e, bins, mask, 1000, ugen, T)
                            ints[g] = (lo / s_y, hi / s_y)
                    elif kind == "latent":
                        lo, hi, ulo, uhi = head.intervals(u, mask)
                        ints["joint"], ints["unit"] = (lo, hi), (ulo, uhi)
                    else:
                        mean, sd = head(u, mask)
                        ints["joint"] = (mean - 1.96 * sd, mean + 1.96 * sd)
                        um, usd = (mk @ mu_i) / k, torch.sqrt(mk @ sd_i ** 2) / k
                        ints["unit"] = (um - 1.96 * usd, um + 1.96 * usd)
                for g, (lo, hi) in ints.items():
                    row_cov[g].append(float(((lo <= theta) & (theta <= hi)).float().mean()))
                    row_wid[g].append(float((hi - lo).mean()))
                idx = mask.nonzero()[:, 1].view(mask.shape[0], k).cpu()
                acc[k].add(E_raw[idx[-1:]], SD_raw[idx[-1:]])
                acc_all[k].add(E_raw[idx], SD_raw[idx])
                acc_std[k].add(E_raw[idx] / s_y, SD_raw[idx] / s_y)
            for g in aggs:
                cov[g].append(row_cov[g])
                wid[g].append(row_wid[g])
        m_k = [acc[k].m() for k in ks]
        rho_k = [acc[k].rho() for k in ks]
        m_a = [acc_all[k].m() for k in ks]
        rho_a = [acc_all[k].rho() for k in ks]
        m_s = [acc_std[k].m() for k in ks]
        rho_s = [acc_std[k].rho() for k in ks]
        res = dict(n=n, ks=ks, coverage={g: cov[g] for g in aggs}, width={g: wid[g] for g in aggs},
                   m_by_k=m_k, rho_by_k=rho_k, m_S=float(np.nanmedian(m_k)), rho_S=float(np.nanmedian(rho_k)),
                   m_by_k_all=m_a, rho_by_k_all=rho_a, m_S_all=float(np.nanmedian(m_a)),
                   rho_S_all=float(np.nanmedian(rho_a)), m_by_k_std=m_s, rho_by_k_std=rho_s,
                   m_S_std=float(np.nanmedian(m_s)), rho_S_std=float(np.nanmedian(rho_s)),
                   nrmse_y=float(np.mean(nrmse)))
        if temps:
            res["temperature_median"] = float(np.median(temps))
        out[dgp] = res
        print(f"{dgp:13s} m_S {res['m_S']:.2f} ({res['m_S_all']:.2f}, {res['m_S_std']:.2f}) rho_S {res['rho_S']:.4f} "
              f"({res['rho_S_all']:.4f}, {res['rho_S_std']:.4f}) "
              f"nRMSE {res['nrmse_y']:.3f} | k=n coverage "
              + " ".join(f"{g} {np.mean([c[-1] for c in cov[g]]):.3f}" for g in aggs), flush=True)
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
