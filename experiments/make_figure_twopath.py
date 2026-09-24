"""twopath.pdf: one checkpoint, two query paths, both arms in one row.

Rows are the arm (CNP, latent NP), columns are what is measured (reported
interval width, coverage). x is shared down each column and y across each row,
so the two arms are read against the same scales without repeating tick labels.

Replaces twopath_cnp.pdf and twopath_latent.pdf, which were two full-width
figures carrying one claim. No titles: the caption says what the panels are.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifacts  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "docs/figures/twopath.pdf"
# One legend serves both rows, so its labels must be construction-generic:
# the SCM-marginalized path is an amortized decode for the CNP and latent resampling
# for the latent NP, and the caption says which is which.
ARMS = [(list(artifacts.FIGURE2_CNP), "CNP"),
        (list(artifacts.FIGURE2_LATENT), "latent NP")]
# unit-level path, then the SCM-marginalized path: distinct colour AND dash AND
# marker, so the panels survive greyscale. "Coherent" is not used anywhere in
# the paper, so it is not used here either.
C_MARG, C_JOINT = figstyle.OKABE_ITO[1], figstyle.OKABE_ITO[0]
D_MARG, D_JOINT = figstyle.STYLES[1], figstyle.STYLES[0]
M_MARG, M_JOINT = figstyle.MARKERS[0], figstyle.MARKERS[1]


def series(tags, key):
    runs = [json.load(open(f"logs/inspec/twopath_{t}.json")) for t in tags
            if os.path.exists(f"logs/inspec/twopath_{t}.json")]
    ks = np.array(runs[0]["ks"], float)
    a = np.array([r[key] for r in runs], float)
    se = (a.std(axis=0, ddof=1) / np.sqrt(len(a)) if len(a) > 1
          else np.zeros(a.shape[1]))
    return ks, a.mean(axis=0), se


def main():
    figstyle.apply()
    plt.rcParams.update({"axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "lines.markersize": 2.6})
    # One row: each model gets a subfigure holding its width and coverage
    # panels, so its name can sit centred over exactly that pair.
    fig = plt.figure(figsize=(figstyle.WIDTH, 1.7))
    subs = fig.subfigures(1, 2, wspace=0.02)
    rows, handles, pairs = [], None, []
    for sub, (tags, short) in zip(subs, ARMS):
        axw, axc = sub.subplots(1, 2)
        pairs.append((sub, axw, axc))
        sub.suptitle(short, fontsize=9)
        ks, wI, _ = series(tags, "width_B")
        _, wJ, _ = series(tags, "width_A")
        ref = wI[0] * np.sqrt(ks[0] / ks)
        axw.plot(ks, ref, color=figstyle.GREY, lw=0.9, ls=(0, (1, 1.5)),
                 zorder=2, label=r"$k^{-1/2}$")
        axw.plot(ks, wI, color=C_MARG, ls=D_MARG, marker=M_MARG, zorder=4,
                 label="unit-level aggregation")
        axw.plot(ks, wJ, color=C_JOINT, ls=D_JOINT, marker=M_JOINT, zorder=5,
                 label="SCM-marginalized aggregation")
        axw.set_xscale("log")
        axw.set_yscale("log")
        axw.set_ylabel("width")
        handles = axw.get_legend_handles_labels()

        ks, cI, sI = series(tags, "cov_B")
        _, cJ, sJ = series(tags, "cov_A")
        for c, sd, col, dd, mk in ((cI, sI, C_MARG, D_MARG, M_MARG),
                                   (cJ, sJ, C_JOINT, D_JOINT, M_JOINT)):
            axc.plot(ks, c, color=col, ls=dd, marker=mk, zorder=4)
            axc.fill_between(ks, c - 1.96 * sd, c + 1.96 * sd, color=col,
                             alpha=0.13, lw=0, zorder=2)
        axc.axhline(0.95, color=figstyle.GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        axc.set_xscale("log")
        axc.set_ylim(0.0, 1.05)
        axc.set_ylabel("coverage")
        for ax in (axw, axc):
            ax.set_xlabel("subset size $k$")
        bI = np.polyfit(np.log(ks), np.log(wI), 1)[0]
        bJ = np.polyfit(np.log(ks), np.log(wJ), 1)[0]
        rows.append((short, bI, bJ, cI[-1], cJ[-1]))

    # One legend for all four panels, above them.
    fig.legend(*handles, loc="outside upper center", ncol=3, frameon=False,
               handlelength=2.4, columnspacing=1.4)
    # Centre each model's name over its two axes rather than over its whole
    # subfigure, whose left margin holds the y label and pulls the centre left.
    fig.canvas.draw()
    for sub, axw, axc in pairs:
        mid = 0.5 * (axw.get_window_extent().x0 + axc.get_window_extent().x1)
        sub._suptitle.set_x(sub.transSubfigure.inverted().transform((mid, 0))[0])
    figstyle.save(fig, OUT)
    print(f"  {'arm':10s} {'width exp marg':>14s} {'width exp coh':>13s} "
          f"{'cov marg':>9s} {'cov coh':>8s}")
    for s_, bI, bJ, cI, cJ in rows:
        print(f"  {s_:10s} {bI:+14.3f} {bJ:+13.3f} {cI:9.3f} {cJ:8.3f}")


if __name__ == "__main__":
    main()
