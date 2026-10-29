"""Figure 1 (benchmark_ladder.pdf): CausalPFN coverage against k/n per benchmark.

Random subsets, with the predicted curve 2 Phi(z sqrt(R)) - 1 for each
benchmark. The x-axis is k/n since sample sizes differ by more than 6x.
HillBHeavy is omitted; its curve lies on Hill-B's.
"""

from __future__ import annotations

import json

import os
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = "docs/figures/benchmark_ladder.pdf"
Z = 1.96
# Ordered by k*/n. rho_S and m_S are in Table 4.
SHOW = [("HillA", "Hill-A"), ("FriedmanConf", "Friedman"), ("IHDP", "IHDP"),
        ("HillB", "Hill-B"), ("KangSchafer", "Kang\u2013Schafer"),
        ("ACIC2016", "ACIC 2016")]


def main():
    figstyle.apply()
    d = json.load(open("logs/inspec/benchmark_ladder.json"))
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 3.4))
    rows = []
    for i, (key, lab) in enumerate(SHOW):
        col = figstyle.OKABE_ITO[i]
        dash = figstyle.STYLES[i]
        mk = figstyle.MARKERS[i]
        r = d[key]
        n = r["n"]
        ks = np.array(r["ks"], float)
        cov = np.array(r["coverage"], float)
        se = np.array(r["cov_se_clustered"], float)
        rho = float(np.nanmedian(r["rho_by_k"]))
        m = float(np.nanmedian(r["m_by_k"]))
        kx = r["breaking_k"]
        pred = 2 * norm.cdf(Z * np.sqrt(m / (1 + (ks - 1) * rho))) - 1
        ax.plot(ks / n, cov, color=col, ls=dash, marker=mk, zorder=4,
                label=rf"{lab} ($n$={n})")
        ax.fill_between(ks / n, cov - 1.96 * se, cov + 1.96 * se, color=col,
                        alpha=0.10, lw=0, zorder=2)
        ax.plot(ks / n, pred, color=col, lw=0.8, ls=(0, (1, 1.4)), zorder=5,
                alpha=0.8)
        rows.append((lab, n, rho, m, kx, kx / n if kx == kx else float("nan"),
                     cov[-1]))
    ax.plot([], [], color=figstyle.GREY, lw=0.8, ls=(0, (1, 1.4)),
            label=r"predicted, $2\Phi(z\sqrt{R})-1$")
    ax.axhline(0.95, color=figstyle.GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("fraction of the sample aggregated, $k/n$")
    ax.set_ylabel("coverage of nominal 95% interval")
    ax.legend(loc="lower left", ncol=2, columnspacing=1.1, handletextpad=0.5)
    ax.set_ylim(0.10, 1.03)
    # Label k = n just above the axis, right-aligned inside the frame.
    ax.annotate("full sample", xy=(1.0, 0.012), xycoords="axes fraction",
                xytext=(-2, 0), textcoords="offset points", ha="right",
                va="bottom", fontsize=plt.rcParams["xtick.labelsize"])
    figstyle.save(fig, OUT)
    print(f"{'benchmark':15s} {'n':>5s} {'rho_S':>8s} {'m_S':>6s} "
          f"{'k*':>6s} {'k*/n':>7s} {'cov@n':>7s}")
    for lab, n, rho, m, kx, kn, cn in rows:
        print(f"{lab:15s} {n:5d} {rho:8.4f} {m:6.2f} "
              f"{('%.0f' % kx) if kx == kx else 'none':>6s} "
              f"{('%.3f' % kn) if kn == kn else '---':>7s} {cn:7.3f}")


if __name__ == "__main__":
    main()
