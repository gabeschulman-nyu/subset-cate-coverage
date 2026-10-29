"""Figure 4 (design_effect.pdf): coverage against the variance ratio.

Under unit-level aggregation coverage depends on k only through

    R = m_S / [1 + (k-1) rho_S],        coverage = 2 Phi(z sqrt(R)) - 1.

Plots the twopath cells of our models (both paths, at the R of their
unit-level marginals) and CausalPFN's cells against R, and prints the
deviation from the curve per series.
"""

from __future__ import annotations

import glob
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
from scipy.stats import norm  # noqa: E402

OUT = "docs/figures/design_effect.pdf"
GREY = "#666666"
IND, COH, SHIP = (figstyle.OKABE_ITO[1], figstyle.OKABE_ITO[3],
                  figstyle.OKABE_ITO[0])
# CausalPFN whole-sample cells; m and rho from its own per-unit errors.
SHIPPED = artifacts.SCATTER_TAGS
# Above R_MAX the curve is within a few thousandths of 1. Those points are not
# plotted; the printed statistics include them.
R_MAX = 1.5
# Thinning for a series labelled "per-unit". No current label is, so every
# point is plotted.
THIN = 3
# CausalPFN pool-fraction ladders, with per-k m_S and rho_S measured inside
# the pool.
SHIPPED_LADDERS = tuple(f"{t}.json" for t in artifacts.SHIPPED_LADDERS)
# (marker, legend name) per model. The legend groups series by aggregation,
# unit-level or SCM-marginalized, as in Figure 2.
ARM_MARK = {"route_a": ("o", "latent NP"),
            "route_b": ("s", "CNP")}
Z = 1.96


def collect(recipe_only=True):
    rows = []
    # Cells with the same checkpoint, k grid, selection and coverage curve are
    # one measurement and are counted once.
    seen = set()
    # Cell selection is defined in artifacts.py.
    for t, j in artifacts.twopath_cells(recipe_only=recipe_only):
        # rho_S and m are those of the checkpoint's unit-level marginals, and
        # both paths are plotted at that R. pool_frac is not in the key since
        # some cells record it as None.
        fingerprint = (j.get("checkpoint"), tuple(j["ks"]), j.get("selection"),
                       tuple(j["cov_B"]))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        rho, m = j.get("rho_S", j["rho"]), j["m"]
        for i, k in enumerate(j["ks"]):
            R = m / (1.0 + (k - 1) * rho)
            for path, key in (("independent", "cov_B"), ("scm_aware", "cov_A")):
                c = j[key][i]
                if c != c:
                    continue
                rows.append({"R": R, "cov": c, "path": path,
                             "arm": j.get("arm", "route_b"), "k": k, "cell": t})
    return rows


def collect_shipped():
    rows = []
    for tag in SHIPPED:
        f = f"logs/inspec/{tag}.json"
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        # Uniform subsets, so the pool is the whole sample and rho_S = rho.
        rho, m = j["rho"], j["m"]
        for k, c in zip(j["ks"], j["coverage_draw"], strict=True):
            if c == c:
                rows.append({"R": m / (1.0 + (k - 1) * rho), "cov": c,
                             "k": k, "cell": tag})
    for fn in SHIPPED_LADDERS:
        f = f"logs/inspec/{fn}"
        if not os.path.exists(f):
            continue
        for tag, modes in json.load(open(f)).items():
            for mode, v in modes.items():
                # A whole-sample rung repeats res_n500's setting.
                if artifacts.is_whole_sample_rung(mode) or not v.get("m_by_k"):
                    continue
                for k, c, rho, m in zip(v["ks"], v["coverage"], v["rho_by_k"],
                                        v["m_by_k"], strict=True):
                    if c == c and rho == rho and m == m and rho > 0:
                        rows.append({"R": m / (1.0 + (k - 1) * rho), "cov": c,
                                     "k": k, "cell": f"{tag}/{mode}"})
    return rows


def main():
    rows_all = collect()
    rows = [r for r in rows_all if r["R"] <= R_MAX]
    R = np.array([r["R"] for r in rows])
    C = np.array([r["cov"] for r in rows])
    ind = np.array([r["path"] == "independent" for r in rows])

    figstyle.apply()
    # Included at 0.64\textwidth (about 3.5 in), so drawn at 3.6 in.
    plt.rcParams.update({"xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.labelsize": 10, "legend.fontsize": 8})
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    xs = np.geomspace(max(R.min(), 1e-3), R.max() * 1.1, 300)
    ax.plot(xs, 2 * norm.cdf(Z * np.sqrt(xs)) - 1, color="k", lw=1.3, zorder=6,
            label=r"$2\Phi(z\sqrt{R})-1$")

    for arm, (mk, short) in ARM_MARK.items():
        for msk, col, lab, face in (
                (ind, IND, f"unit-level: {short}", None),
                (~ind, COH, f"SCM-marginalized: {short}", "none")):
            s = msk & np.array([r["arm"] == arm for r in rows])
            if not s.any():
                continue
            if lab.startswith("per-unit"):
                idx = np.where(s)[0][::THIN]
                s = np.zeros_like(s)
                s[idx] = True
            ax.scatter(R[s], C[s], s=14, marker=mk, alpha=0.7, zorder=4,
                       color=col if face is None else "none",
                       edgecolor=col if face else "white", linewidth=1.0,
                       label=lab)

    sh_all = collect_shipped()
    sh = [r for r in sh_all if r["R"] <= R_MAX]
    if sh:
        ax.scatter([r["R"] for r in sh], [r["cov"] for r in sh], s=18,
                   marker="D", alpha=0.85, zorder=5, color=SHIP,
                   edgecolor="white", linewidth=0.4,
                   label="unit-level: CausalPFN")

    ax.axhline(0.95, color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axvline(1.0, color=GREY, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"variance ratio $R = m_S\,/\,[1+(k-1)\rho_S]$")
    ax.set_ylabel("coverage")
    # Legend below the axes, two columns: unit-level on the left,
    # SCM-marginalized and the curve on the right.
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in ("unit-level: CNP", "unit-level: latent NP",
                                  "unit-level: CausalPFN",
                                  "SCM-marginalized: CNP",
                                  "SCM-marginalized: latent NP",
                                  r"$2\Phi(z\sqrt{R})-1$")]
    fig.legend([h[i] for i in order], [l[i] for i in order],
               loc="outside lower center", ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.0, labelspacing=0.25)
    ax.set_ylim(0.0, 1.06)
    figstyle.save(fig, OUT)

    Rf = np.array([r["R"] for r in rows_all])
    Cf = np.array([r["cov"] for r in rows_all])
    indf = np.array([r["path"] == "independent" for r in rows_all])
    pred = 2 * norm.cdf(Z * np.sqrt(Rf)) - 1
    print(f"wrote {OUT}   plotted {len(rows)} of {len(rows_all)} points "
          f"(R <= {R_MAX}), {len({r['cell'] for r in rows_all})} cells")
    R, C, ind = Rf, Cf, indf
    for lab, msk in (("unit-level", ind), ("SCM-aware", ~ind)):
        e = np.abs(C[msk] - pred[msk])
        print(f"  {lab:12s} n={msk.sum():4d}  median |obs-pred|={np.median(e):.3f}"
              f"  90th={np.percentile(e, 90):.3f}")
    if sh:
        Rs = np.array([r["R"] for r in sh_all])
        Cs = np.array([r["cov"] for r in sh_all])
        e = np.abs(Cs - (2 * norm.cdf(Z * np.sqrt(Rs)) - 1))
        print(f"  {'CausalPFN':12s} n={len(sh):4d}  median |obs-pred|={np.median(e):.3f}"
              f"  90th={np.percentile(e, 90):.3f}"
              f"  ({len({r['cell'] for r in sh})} cells)")
    # The curve is nearly flat for R >= 1, so R < 1 is reported separately.
    print("  -- restricted to R < 1 (the informative range) --")
    for lab, Rr, Cr in (("unit-level", R[ind], C[ind]),
                        ("SCM-aware", R[~ind], C[~ind]),
                        ("CausalPFN", np.array([r["R"] for r in sh_all]),
                         np.array([r["cov"] for r in sh_all]))):
        msk = Rr < 1.0
        if not msk.any():
            print(f"  {lab:12s} no points with R < 1")
            continue
        e = np.abs(Cr[msk] - (2 * norm.cdf(Z * np.sqrt(Rr[msk])) - 1))
        print(f"  {lab:12s} n={msk.sum():4d}  median |obs-pred|={np.median(e):.3f}"
              f"  90th={np.percentile(e, 90):.3f}")


if __name__ == "__main__":
    main()
