"""The collapse figure: every measurement against the variance ratio.

Coverage vs k needs one curve per condition, the interesting part sits at the
right edge, and arms trained at different n cover different k ranges. None of
that is intrinsic; it is an artefact of plotting against k.

The derivation says coverage depends on k only through the ratio of reported
to true variance,

    R = m_S / [1 + (k-1) rho_S],        coverage = 2 Phi(z sqrt(R)) - 1,

so plotting against R should collapse every (arm, head, selection, k) cell
onto ONE curve. It does, for the unit-level aggregated path. The SCM-aware
path does not lie on it, because it never made the assumption the curve
encodes. That contrast is the paper's claim in a single panel, and it needs
neither matched n nor a shared k grid.

TNP cells are excluded outright rather than hidden behind a default-off flag.
The arm is defined nowhere in the paper, so a switch that can put it back into
a figure is a way for an undefined label to reappear on a regeneration.
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
# The shipped checkpoint composes per-unit posteriors independently, so it is
# subject to (eq:bias) exactly as our composing arms are. Plotting it here
# shows the curve predicting the model the paper is about, not only the arms
# we trained. Its m and rho come from its own per-unit errors, same as
# everything else on the panel.
SHIPPED = artifacts.SCATTER_TAGS
# Points above R_MAX sit on the flat shoulder of 2*Phi(z*sqrt(R))-1, where the
# curve is within a few thousandths of 1 and any monotone function through
# (inf, 1) would fit. They inflate a pooled residual without testing anything,
# so the panel is restricted to the range where the prediction has content.
R_MAX = 1.5
# The marginals series has ~1200 points and saturates into a solid band.
# Thinning is presentational only -- every reported statistic is computed on
# the full set before this is applied.
THIN = 3
# Random-subset cells alone leave the shipped model at R > 2, on the flat
# shoulder where any monotone curve through (inf, 1) fits. The stratum
# ladders carry per-k m_S and rho_S measured inside the pool, which is where
# the informative low-R points come from.
SHIPPED_LADDERS = tuple(f"{t}.json" for t in artifacts.SHIPPED_LADDERS)
# The distinction is whether the aggregate is built by averaging unit-level
# marginals or from the joint, and the two arms reach the joint by different
# means: the CNP amortizes the integral into a direct decode, the latent NP
# resamples the shared latent and averages within each draw. "Coherent" is not
# used anywhere in the paper, so it is not used here.
# (marker, short name for the marginal series, full name for our series)
# One name per arm. The legend is grouped by how the aggregate is formed,
# using the same two names as Fig. twopath -- unit-level and SCM-marginalized
# -- so a reader moving between the figures is not asked to learn a second
# vocabulary for the same distinction.
ARM_MARK = {"route_a": ("o", "latent NP"),
            "route_b": ("s", "CNP")}
Z = 1.96


def collect(recipe_only=True):
    rows = []
    # Two cells scored from the same checkpoint on the same grid with the same
    # selection ARE the same measurement, whatever they are called. A
    # re-scored cell (a verification rerun, say) would otherwise enter the
    # pool twice and be counted twice, which is how a regression check of
    # this very script once added six points to the R<1 count. Identity is
    # taken from content, not from the filename.
    seen = set()
    # Which cells count is defined in experiments/artifacts.py, so the
    # figure and the release build cannot disagree about it.
    for t, j in artifacts.twopath_cells(recipe_only=recipe_only):
        # rho_S and m come from this checkpoint's own unit-level marginals
        # (unit_err / unit_sd in two_path_ablation), so R is the ratio that
        # WOULD apply if its aggregate were built from those marginals. Both
        # series are plotted at that same R, which is what makes the vertical
        # gap between them readable as the effect of the aggregation choice.
        # Deliberately not keyed on pool_frac: cells written before that field
        # existed record None, which would make an older cell and its rerun
        # look distinct. Checkpoint, grid and the realised coverage curve
        # already pin the measurement.
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
        # Subsets are uniform here, so the pool is the whole sample and the
        # task-level rho and m ARE the pool-restricted ones.
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
                if not v.get("m_by_k"):
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
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 3.4))
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
            ax.scatter(R[s], C[s], s=9, marker=mk, alpha=0.7, zorder=4,
                       color=col if face is None else "none",
                       edgecolor=col if face else "white", linewidth=1.0,
                       label=lab)

    sh_all = collect_shipped()
    sh = [r for r in sh_all if r["R"] <= R_MAX]
    if sh:
        ax.scatter([r["R"] for r in sh], [r["cov"] for r in sh], s=11,
                   marker="D", alpha=0.85, zorder=5, color=SHIP,
                   edgecolor="white", linewidth=0.4,
                   label="unit-level: CausalPFN")

    ax.axhline(0.95, color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axvline(1.0, color=GREY, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"variance ratio $R = m_S\,/\,[1+(k-1)\rho_S]$")
    ax.set_ylabel("coverage of nominal 95% interval")
    # Legend outside the axes: inside it covered the low-R corner, which is
    # the part of the curve the panel exists to show.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
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
    # A fit pooled over all R is dominated by points on the flat shoulder,
    # where the curve is nearly constant and any monotone function would do.
    # The restricted fit is the one that carries information.
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
