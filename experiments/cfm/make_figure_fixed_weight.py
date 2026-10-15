"""cfm_fixed_weight_<head>.pdf: same weights, same forward pass, two aggregations.

From eval_final.py (logs/cfm/final/*.json), for one head type on the frozen,
released CausalPFN, three seeds, on the paper's six benchmarks:

    solid    the head's own interval for the subset, SCM-marginalized (CNP:
             its direct decode of theta_S; latent NP: units share each draw
             of z)
    dashed   the same head's per-unit predictive distributions (CNP queried
             one unit at a time) aggregated as if independent: unit-level
             aggregation
    dotted   the paper's prediction for the dashed curve, 2 Phi(1.96 sqrt(m /
             (1 + (k - 1) rho))) - 1, from each seed's measured per-unit
             calibration ratio m_S and error correlation rho_S

Lines are means over the seeds, each seed's coverage averaged over 40 datasets
x 48 subsets per size (one at k = n); bands are the range (min-max) over the
seeds; the dotted line is the mean of each seed's own prediction.
--estimator picks how m_S and rho_S are estimated (eval_final.py): "paper",
as the paper's benchmark table computes CausalPFN's (the default); "all", the
same from every subset; "std", every subset in per-dataset sd(y) units.

    python experiments/cfm/make_figure_fixed_weight.py --head cnp [--estimator paper] [--out PATH]

Earlier head variants: make_figure_fixed_weight_earlier.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = "docs/figures/cfm_fixed_weight_{head}.pdf"
SHOW = [("HillA", "Hill-A"), ("FriedmanConf", "Friedman"), ("IHDP", "IHDP"),
        ("HillB", "Hill-B"), ("KangSchafer", "Kang–Schafer"), ("ACIC2016", "ACIC 2016")]
RUNS = {"cnp": ["v2_cnp_rand_s0", "v2_cnp_rand_s1", "v2_cnp_rand_s2"],
        "latent": ["v2_lathist_s0", "v2_lathist_s1", "v2_lathist_s2"]}
JOINT = {"cnp": "SCM-marginalized (CNP head)", "latent": "SCM-marginalized (latent-NP head)"}
# figure title naming the model, so the CNP and latent-NP versions cannot be confused
TITLE = {"cnp": "CNP head (3 seeds): SCM-marginalized vs unit-level aggregation, same weights",
         "latent": "Latent-NP head (3 seeds): SCM-marginalized vs unit-level aggregation, same weights"}
ESTIMATOR = {"paper": "", "all": "_all", "std": "_std"}


def load(runs, est):
    """{benchmark: {run: {ks, n, m, rho, joint_cov, unit_cov, pred_unit_cov}}}."""
    out = {}
    for r in runs:
        d = json.load(open(f"logs/cfm/final/{r}.json"))
        for b, _ in SHOW:
            x = d[b]
            m, rho = x[f"m_S{est}"], x[f"rho_S{est}"]
            R = m / (1 + (np.array(x["ks"], float) - 1) * rho)
            out.setdefault(b, {})[r] = dict(
                ks=x["ks"], n=x["n"], m=m, rho=rho,
                joint_cov=np.mean(x["coverage"]["joint"], 0).tolist(),
                unit_cov=np.mean(x["coverage"]["unit"], 0).tolist(),
                pred_unit_cov=(2 * norm.cdf(1.96 * np.sqrt(R)) - 1).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=list(RUNS), default="cnp")
    ap.add_argument("--estimator", choices=list(ESTIMATOR), default="paper",
                    help="m_S / rho_S estimator for the predicted curve")
    ap.add_argument("--out", default=None, help=f"output path (default: {OUT})")
    a = ap.parse_args()
    figstyle.apply()
    runs = RUNS[a.head]
    d = load(runs, ESTIMATOR[a.estimator])
    fig, axes = plt.subplots(2, 3, figsize=(figstyle.WIDTH, 4.1), sharex=True, sharey=True)
    fig.suptitle(TITLE[a.head], fontsize=plt.rcParams["axes.titlesize"], fontweight="semibold")
    for ax, (b, lab) in zip(axes.flat, SHOW, strict=True):
        r0 = d[b][runs[0]]
        x = np.array(r0["ks"], float) / r0["n"]
        for key, name, col, ls, mk in (("joint_cov", JOINT[a.head], figstyle.OKABE_ITO[0], figstyle.STYLES[0], "o"),
                                       ("unit_cov", "unit-level aggregation, same weights", figstyle.OKABE_ITO[1],
                                        figstyle.STYLES[1], "s")):
            c = np.array([d[b][r][key] for r in runs])
            ax.fill_between(x, c.min(0), c.max(0), color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(x, c.mean(0), color=col, ls=ls, marker=mk, zorder=4, label=name)
        p = np.array([d[b][r]["pred_unit_cov"] for r in runs]).mean(0)
        ax.plot(x, p, color=figstyle.OKABE_ITO[1], lw=0.9, ls=(0, (1, 1.4)), zorder=5,
                label=r"predicted, $2\Phi(1.96\sqrt{m/(1+(k-1)\rho)})-1$")
        ax.axhline(0.95, color=figstyle.GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_title(rf"{lab} ($n$={r0['n']})")
    axes[0, 0].set_ylim(0.0, 1.03)
    for ax in axes[:, 0]:
        ax.set_ylabel("coverage")
    for ax in axes[1]:
        ax.set_xlabel("$k/n$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, columnspacing=1.2, handletextpad=0.5)
    figstyle.save(fig, a.out or OUT.format(head=a.head))
    for b, lab in SHOW:
        rs = [d[b][r] for r in runs]
        print(f"{lab:14s} m {np.mean([r['m'] for r in rs]):6.2f} rho {np.mean([r['rho'] for r in rs]):.3f} | k=n: "
              f"joint {np.mean([r['joint_cov'][-1] for r in rs]):.2f} unit {np.mean([r['unit_cov'][-1] for r in rs]):.2f} "
              f"pred {np.mean([r['pred_unit_cov'][-1] for r in rs]):.2f}")


if __name__ == "__main__":
    main()
