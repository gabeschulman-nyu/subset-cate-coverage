"""correction.pdf: which of the two corrections does the work.

Three constructions of the same interval, per cell:

  A  as reported                   sigma_rep^2(k) = sigmabar_S^2 / k
  B  per-unit recalibration only   divide the per-unit variance by m_S,
                                   then compose independently exactly as A does
  C  recalibration + design effect divide by m_S and multiply by 1+(k-1)rho_S

Registered before running: A falls; B falls FASTER than A wherever m_S > 1,
because the cushion carrying A to large k is precisely what recalibration
removes; C removes the k-dependence.

Two panels, chosen to span the logical space rather than to repeat it. Where a
cushion exists (m_S >> 1), removing it hurts. Where it does not (m_S < 1),
removing the deficit helps at small k and still leaves the k-dependence
untouched. The soft-overlap and IHDP cells tell the first story again at lower
intensity and live in the table instead.

C is NOT flat at nominal: on the clean cell it runs 0.877 to 0.929, below
nominal throughout, and across cells it sits between 0.88 and 0.99. The claim
is that it removes the k-dependence -- slope +0.006 against A's -0.007 and B's
-0.079 -- and lands within a few points of nominal, the residual being the
shape of the standardized error.

Arm C is not a deployable fix and the caption says so: it substitutes the
measured error variance for the model's reported one, so it builds the oracle
interval. What it shows is that (rho_S, m_S) are jointly SUFFICIENT -- no
third quantity is needed to account for the k-dependence. Arm B is what keeps
the figure from being tautological: it isolates which correction does the
work, and the answer is the one nobody applies.
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

OUT = "docs/figures/correction.pdf"
GREY = figstyle.GREY
COL = {"A": figstyle.OKABE_ITO[1], "B": figstyle.OKABE_ITO[3],
       "C": figstyle.OKABE_ITO[0]}
DASH = {"A": figstyle.STYLES[1], "B": figstyle.STYLES[2], "C": figstyle.STYLES[0]}
MK = {"A": figstyle.MARKERS[0], "B": figstyle.MARKERS[2], "C": figstyle.MARKERS[1]}
LAB = {"A": "A: as reported", "B": r"B: recalibrated by $m_S$",
       "C": r"C: recalibrated $+$ design effect"}


def jload(p):
    with open(f"logs/inspec/{p}") as f:
        return json.load(f)


def panels(all_cells=False):
    """all_cells=True returns every cell, for the table; the figure keeps the
    two that span the logical space."""
    sw = jload("threearm_n500.json")
    out = []
    r = sw["n500"]["propensity 1.00"]
    out.append(("CausalPFN, clean, all units", r["m_bar"],
                np.array(r["ks"], float), r["coverage"], r["cov_recal"],
                r["cov_full"], None))
    if all_cells:
        r = sw["ovsoft500"]["propensity 0.10"]
        out.append(("CausalPFN, soft overlap, bottom 10%", r["m_bar"],
                    np.array(r["ks"], float), r["coverage"], r["cov_recal"],
                    r["cov_full"], None))
    j = jload("twopath_g1k_s1.json")
    out.append(("Ours (CNP), $m_S<1$", j["m"], np.array(j["ks"], float),
                j["cov_B"], j["cov_B_recal"], j["cov_B_full"], None))
    if all_cells:
        with open("logs/c3read/fw_legacy_direct.json") as f:
            lg = json.load(f)["IHDP"]
        out.append(("IHDP (kurtosis $1.3\\times10^3$)", lg["m"],
                    np.array(lg["ks"], float), lg["coverage"],
                    lg["cov_recal"], lg["cov_full"], lg["kurtosis"]))
    return out


def main():
    figstyle.apply()
    ps = panels()
    fig, axes = plt.subplots(1, len(ps), figsize=(figstyle.WIDTH, 2.9),
                             sharey=True)
    rows = []
    for ax, (title, m, ks, a, b, c, kurt) in zip(axes, ps, strict=True):
        for key, v in (("A", a), ("B", b), ("C", c)):
            v = np.array(v, float)
            ok = ~np.isnan(v)
            ax.plot(ks[ok], v[ok], color=COL[key], ls=DASH[key],
                    marker=MK[key], zorder=4, label=LAB[key])
            sl = float(np.polyfit(np.log(ks[ok]), v[ok], 1)[0])
            rows.append((title, m, key, v[ok][0], v[ok][-1], sl))
        ax.axhline(0.95, color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("subset size $k$")
        # m_S annotated in-axes rather than as a title: it identifies the
        # panel, and the caption carries the rest.
        ax.annotate(rf"$m_S$={m:.2f}", xy=(0.97, 0.04),
                    xycoords="axes fraction", ha="right", fontsize=8)
    axes[0].set_ylabel("coverage of nominal 95% interval")
    axes[0].set_ylim(0.05, 1.03)
    axes[0].legend(loc="lower left")

    figstyle.save(fig, OUT)
    print(f"{'cell':34s} {'m_S':>6s} {'arm':>3s} {'first':>6s} {'last':>6s} {'slope':>8s}")
    for t, m, k, f0, f1, sl in rows:
        print(f"{t:34s} {m:6.2f} {k:>3s} {f0:6.3f} {f1:6.3f} {sl:+8.4f}")


if __name__ == "__main__":
    main()


def table():
    """Table alternative to the figure. Built from the same rows so the two
    cannot disagree.

    Reports each arm at both ends of the k range plus its slope in log k,
    because the claim the appendix makes about arm C is that it removes the
    k-dependence -- which a single coverage number at k_max cannot show.
    """
    ps = panels(all_cells=True)
    rows = []
    for title, m, ks, a, b, c, _ in ps:
        for key, v in (("A", a), ("B", b), ("C", c)):
            v = np.array(v, float)
            ok = ~np.isnan(v)
            if ok.sum() < 2:
                continue
            lk = np.log(ks[ok])
            rows.append((title, m, int(ks[ok][0]), int(ks[ok][-1]), key,
                         v[ok][0], v[ok][-1],
                         float(np.polyfit(lk, v[ok], 1)[0])))
    # Titles are written for matplotlib, where % is literal; LaTeX needs it
    # escaped or the rest of the row is commented out.
    body = []
    for i, (t, m, k0, k1, arm, c0, c1, sl) in enumerate(rows):
        # One cell spans three rows; name it once so the eye groups them.
        head = (f"{t.replace('%', chr(92) + '%')} & {m:.2f} & {k0} & {k1}"
                if i % 3 == 0 else " &  &  & ")
        body.append(f"{head} & {arm} & {c0:.3f} & {c1:.3f} & {sl:+.4f} \\\\")
        if i % 3 == 2 and i + 1 < len(rows):
            body.append("\\addlinespace[2pt]")
    return (
        "\\begin{table}[H]\n\\centering\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{lrrrlrrr}\n\\toprule\n"
        "\\textbf{cell} & \\textbf{$m_S$} & \\textbf{$k_{\\min}$} & "
        "\\textbf{$k_{\\max}$} & \\textbf{arm} & \\textbf{cov at $k_{\\min}$} & "
        "\\textbf{cov at $k_{\\max}$} & \\textbf{slope} \\\\\n\\midrule\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Three interval constructions on the same stored unit-level "
        "outputs, so no retraining is involved and the arms differ only in how "
        "the reported half-width is formed. A is the interval as reported, B "
        "rescales the unit-level variance by $m_S$, and C additionally "
        "multiplies by the design effect $1+(k-1)\\rho_S$. Slope is the "
        "ordinary least-squares slope of coverage in $\\log k$, which is what "
        "separates the arms: A and B fall with $k$, while C is flat to within "
        "a hundredth per decade. B is worse than A at $k_{\\max}$ wherever "
        "$m_S>1$, since rescaling removes the cushion that was carrying A "
        "without touching the error structure. C is the oracle interval, "
        "reported to show that $(\\rho_S, m_S)$ suffice; both are measured "
        "against the true effects, so it is a diagnostic rather than a "
        "deployable correction.}\n"
        "\\label{tab:threearm}\n\\end{table}\n")
