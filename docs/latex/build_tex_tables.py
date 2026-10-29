"""Write the paper's Tables 2-6 from the measurement artifacts: threearm
(Appendix C), subset_selection (Appendix D), legacy_breaking, n2000 and nsweep
(Appendix F). experiments/artifacts.py selects the artifact for each setting.

    python docs/latex/build_tex_tables.py   -> docs/latex/tables/*.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import artifacts  # noqa: E402

OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)

# Tables report the measured crossing \hat k^* (as benchmark_ladder.kstar and
# inspec_stage3.first_below compute it); Eq. (kstar) gives the predicted k^*,
# 1 + (m_S - 1)/rho_S.
KHAT = "$\\hat k^*$"
KHAT_DEF = ("$\\hat k^*$ is the subset size at which observed coverage "
            "first falls below nominal, interpolated in $\\log k$")
KHAT_XREF = "$\\hat k^*$ is defined in Section~\\ref{sec:expsetup}."


def esc(s):
    # Escape outside $...$ only, so math such as $X_1>0$ is left intact.
    out, math = [], False
    for part in str(s).split("$"):
        out.append(part if math else
                   part.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%"))
        math = not math
    return "$".join(out)


def latex_table(headers, rows, align=None, caption="", label="", note=""):
    align = align or ("l" + "r" * (len(headers) - 1))
    head = " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\"
    # A None row renders as \addlinespace; ("GROUP", title) as a spanning
    # italic header.
    ncol = len(headers)

    def render(r):
        if r is None:
            return "\\addlinespace[3pt]"
        if isinstance(r, tuple) and r and r[0] == "GROUP":
            return (f"\\multicolumn{{{ncol}}}{{l}}{{\\itshape {esc(r[1])}}}"
                    " \\\\")
        return " & ".join(esc(c) for c in r) + " \\\\"

    body = "\n".join(render(r) for r in rows)
    notestr = f"\n\\vspace{{2pt}}\\footnotesize {note}" if note else ""
    # No \resizebox, so table text stays at \footnotesize.
    return (f"\\begin{{table}}[tp]\n\\centering\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}{notestr}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


INSPEC = ROOT / "logs/inspec"


def _inspec(name):
    with open(INSPEC / name) as f:
        return json.load(f)


# Benchmark measurements, read by Tables 2 and 4.
bl = _inspec(f"{artifacts.BENCHMARK}.json")


# ---- Table 2, threearm (Appendix C): three intervals from the same outputs ----
def threearm_settings():
    """(title, m_S, ks, reported, rescaled, +design effect) per setting, from
    the run used for that setting throughout the paper (experiments/artifacts.py).
    CausalPFN first, the CNP last.
    """
    out = []
    # CausalPFN, s=1, whole sample; the n=500 run of Tables 3 and 6.
    r = _inspec("res_n500.json")
    out.append(("CausalPFN|$s=1$, all units", r["m"], np.array(r["ks"], float),
                r["coverage_draw"], r["cov_recal"], r["cov_full"]))
    # CausalPFN, sharpness s=4, bottom 10% of the propensity (Table 3).
    v = _inspec("ladder_propensity_m_ovsoft500.json")["ovsoft500"]["propensity 0.10"]
    out.append(("CausalPFN|$s=4$, bottom 10%", v["m_bar"],
                np.array(v["ks"], float), v["coverage"], v["cov_recal"],
                v["cov_full"]))
    # CausalPFN on IHDP; the run of Figure 1 and Table 4.
    b = bl["IHDP"]
    out.append(("CausalPFN|IHDP", float(np.nanmedian(b["m_by_k"])),
                np.array(b["ks"], float), b["coverage"], b["cov_recal"],
                b["cov_full"]))
    # Gaussian-head CNP at n=500, Figure 2's three seeds averaged. Each seed's
    # corrections use its own m_S and rho_S; m_S shown is their mean.
    seeds = [_inspec(f"twopath_{t}.json") for t in artifacts.FIGURE2_CNP]
    assert all(j["ks"] == seeds[0]["ks"] for j in seeds)
    avg = lambda key: np.mean([j[key] for j in seeds], axis=0).tolist()
    out.append(("CNP (ours)|$s=1$", float(np.mean([j["m"] for j in seeds])),
                np.array(seeds[0]["ks"], float), avg("cov_B"),
                avg("cov_B_recal"), avg("cov_B_full")))
    return out

def threearm_table():
    """Each setting appears three times, once per interval, at both ends of its
    k range."""
    arm_label = {"A": "reported", "B": "rescaled", "C": "$+$ design effect"}
    rows = []
    for title, m, ks, a, b, c in threearm_settings():
        for key, v in (("A", a), ("B", b), ("C", c)):
            v = np.array(v, float)
            ok = ~np.isnan(v)
            if ok.sum() < 2:
                continue
            # For the CNP, the corrected interval is its unit-level one.
            label = ("unit-level" if key == "A" and title.startswith("CNP")
                     else arm_label[key])
            rows.append((title, m, int(ks[ok][-1]), label,
                         v[ok][0], v[ok][-1]))
    body = []
    for i, (t, m, k1, arm, c0, c1) in enumerate(rows):
        # A setting spans three rows; title is "model|data", printed on the
        # first. Escape % or LaTeX treats the rest of the row as a comment.
        model, data = t.split("|")
        safe = data.replace("%", chr(92) + "%")
        head = (f"{model} & {safe} & {m:.2f} & {k1}"
                if i % 3 == 0 else " &  &  & ")
        body.append(f"{head} & {arm} & {c0:.2f} & {c1:.2f} \\\\")
        if i % 3 == 2 and i + 1 < len(rows):
            body.append("\\addlinespace[2pt]")
    return (
        "\\begin{table}[tp]\n\\centering\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{llrrlrr}\n\\toprule\n"
        "\\textbf{model} & \\textbf{data} & \\textbf{$m_S$} & "
        "\\textbf{$k_{\\max}$} & "
        "\\textbf{interval} & \\textbf{cov, smallest $k$} & "
        "\\textbf{cov, $k_{\\max}$} \\\\\n\\midrule\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Coverage of the three intervals of "
        "Appendix~\\ref{app:correction} at the smallest and largest subset in "
        "each setting. $k_{\\max}$ is the largest evaluated subset size within "
        "the pool. The \\emph{$+$ design effect} interval uses $\\rho_S$ "
        "and $m_S$ measured against the true effects and is not available in "
        "practice.}\n"
        "\\label{tab:threearm}\n\\end{table}\n")


(OUT / "threearm.tex").write_text(threearm_table())

# ---- Table 3, subset-selection (Appendix D) ---------------------------------
# R = m_S / [1 + (k-1) rho_S] and k is bounded by the pool, so the table pairs
# m_S with the design effect at k_max.
LADDERS = [
    ("$s=1$", "ladder_propensity_m_n500.json", "n500", "propensity stratum", [
        ("propensity 1.00", "all units"), ("propensity 0.50", "bottom 50%"),
        ("propensity 0.25", "bottom 25%"), ("propensity 0.10", "bottom 10%")]),
    ("$s=1$", "ladder_nested_n500.json", "n500", "nested covariate cell", [
        ("$X_1$>0", "$X_1>0$"),
        ("$X_1$>0, $X_2$>0", "$\\wedge\\, X_2>0$"),
        ("$X_1$>0, $X_2$>0, $X_3$>0", "$\\wedge\\, X_3>0$")]),
    ("$s=1$", "ladder_neighbor_ball_m_n500.json", "n500", "covariate neighbourhood", [
        ("neighbor_ball 0.50", "nearest 50%"),
        ("neighbor_ball 0.25", "nearest 25%"),
        ("neighbor_ball 0.10", "nearest 10%")]),
    ("$s=4$", "ladder_propensity_m_ovsoft500.json", "ovsoft500",
     "propensity stratum", [
        ("propensity 1.00", "all units"), ("propensity 0.50", "bottom 50%"),
        ("propensity 0.25", "bottom 25%"), ("propensity 0.10", "bottom 10%")]),
    ("$s=4$", "ladder_nested_ovsoft500.json", "ovsoft500",
     "nested covariate cell", [
        ("$X_1$>0", "$X_1>0$"),
        ("$X_1$>0, $X_2$>0", "$\\wedge\\, X_2>0$"),
        ("$X_1$>0, $X_2$>0, $X_3$>0", "$\\wedge\\, X_3>0$")]),
]
rows_ss, seen = [], set()
for dgp, fn, tag, rule, rungs in LADDERS:
    d = _inspec(fn)[tag]
    for key, lab in rungs:
        v = d[key]
        if artifacts.is_whole_sample_rung(key):
            # Whole-sample baseline from the n-sweep run, as in Tables 2 and 6;
            # the ladder's whole-sample rung is a separate run of that setting.
            r = _inspec(f"res_{tag}.json")
            rho, m, kmax, kb = r["rho"], r["m"], r["n"], r["breaking_k_draw"]
        else:
            rho = float(np.nanmedian(v["rho_by_k"]))
            m = float(np.nanmedian(v["m_by_k"]))
            kmax = max(v["ks"])
            kb = v["breaking_k"]
        # One spanning header per (DGP, rule) block.
        block = (dgp, rule)
        first = block not in seen
        if first:
            if seen:
                rows_ss.append(None)      # rendered as \addlinespace
            rows_ss.append(("GROUP", f"{dgp}, {rule}"))
        # "all units" is the unselected baseline, not a stratum of the rule.
        shown = "all units (baseline)" if lab == "all units" else lab
        rows_ss.append([shown, f"{rho:.3f}",
                        # two decimals, as in Table 2
                        f"{m:.2f}", f"{kmax}",
                        f"{1 + (kmax - 1) * rho:.1f}",
                        f"{kb:.0f}" if kb == kb else "---"])
        seen.add(block)
(OUT / "subset_selection.tex").write_text(latex_table(
    ["subset", "$\\rho_S$", "$m_S$", "$k_{\\max}$",
     "$1+(k_{\\max}\\!-\\!1)\\rho_S$", KHAT],
    rows_ss, align="lrrrrr",
    caption="Subset selection rules on CausalPFN at $n=500$, with "
            "intervals formed by unit-level aggregation, over $64$ tasks with "
            "$48$ subsets per task. $k_{\\max}$ is the largest evaluated subset "
            "size within the pool and $1+(k_{\\max}\\!-\\!1)\\rho_S$ the design "
            "effect at that size; ``---'' means coverage never falls "
            "below nominal within the pool. " + KHAT_XREF,
    label="tab:subset-selection"))


# ---- Table 4, legacy (Appendix F): the benchmark suite --------------------
# Rows are sorted by k*/n below.
ORDER = ["HillA", "FriedmanConf", "IHDP", "HillB", "KangSchafer", "ACIC2016",
         "HillBHeavy"]
# Display names as in Figure 1 and Table 1.
DISPLAY = {"HillA": "Hill-A", "FriedmanConf": "Friedman", "IHDP": "IHDP",
           "HillB": "Hill-B", "KangSchafer": "Kang--Schafer",
           "ACIC2016": "ACIC 2016", "HillBHeavy": "Hill-B heavy-tailed"}
rows4 = []
for dgp in ORDER:
    v = bl[dgp]
    ks, cov, n = v["ks"], v["coverage"], v["n"]
    kb = v["breaking_k"]
    rho = float(np.nanmedian(v["rho_by_k"]))
    m = float(np.nanmedian(v["m_by_k"]))
    # No k*: "---" if coverage never falls below nominal, "$<1$" if it is
    # below nominal already at k = 1. k* and k*/n share a column, "85 (11%)".
    if kb == kb:
        # One decimal below 10%, so small fractions stay distinct.
        frac = 100.0 * kb / n
        kb_s = f"{kb:.0f} ({frac:.0f}%)" if frac >= 10 else f"{kb:.0f} ({frac:.1f}%)"   # esc() escapes the %
    elif cov[0] < 0.95:
        kb_s = "$<1$"
    else:
        kb_s = "---"
    se = v["cov_se_clustered"][-1]
    # Sort key: never crosses first, crosses at k < 1 last, else by decreasing k*/n.
    order_key = (kb / n if kb == kb else
                 float("-inf") if cov[0] < 0.95 else float("inf"))
    rows4.append([order_key, DISPLAY[dgp], str(n), f"{rho:.4f}", f"{m:.2f}",
                  # nRMSE divides by sd(y); sd(tau) is zero on Hill-A and Kang-Schafer.
                  f"{v['nrmse_y']:.2f}", f"{cov[-1]:.2f}", f"{se:.2f}",
                  kb_s])
rows4 = [r[1:] for r in sorted(rows4, key=lambda r: -r[0])]
# Legend for the markers the table prints.
_marks = {r[-1] for r in rows4}
_legend = " and ".join(x for x in (
    "``---'' means coverage never falls below nominal" if "---" in _marks else "",
    "$<1$ that it is already below at the smallest subset" if "$<1$" in _marks else "",
) if x)
_legend = f"; {_legend}" if _legend else ""
(OUT / "legacy_breaking.tex").write_text(latex_table(
    ["benchmark", "$n$", "$\\rho_S$", "$m_S$", "nRMSE", "cov at $k$=$n$",
     "se", "$\\hat k^*$ ($\\hat k^*/n$)"], rows4,
    align="lrrrrrrr",
    caption="The benchmarks of Figure~\\ref{fig:benchmarks} and the "
            "heavy-tailed variant of Hill-B, which the figure omits: CausalPFN with "
            "its temperature calibration, unit-level aggregation, subsets drawn "
            "uniformly at random, 40 datasets per benchmark. " + KHAT_DEF + ", with "
            "$\\hat k^*/n$ as a percentage of the sample" + _legend + ". "
            "nRMSE is the root-mean-square error of the "
            "per-unit effect estimates divided by the standard deviation of the "
            "observed outcome, within each dataset and averaged over "
            "datasets. Rows are ordered by $\\hat k^*/n$.",
    label="tab:legacy"))


# ---- Table 5, n2000 (Appendix F) ---------------------------------------------
# The three n=2000 cells scored by two_path_ablation on one task stream. cov_B
# is unit-level aggregation, cov_A the model's SCM-marginalized aggregation.
AGGREGATION = {"cov_B": "unit-level", "cov_A": "SCM-marginalized"}
N2SRC = artifacts.N2000_SOURCES
_n2 = {t: _inspec(f"twopath_{t}.json") for t, _, _ in N2SRC}
_ks = _n2[N2SRC[0][0]]["ks"]
assert all(_n2[t]["ks"] == _ks for t, _, _ in N2SRC), "n=2000 cells disagree on k"
assert all(_n2[t]["num_tasks"] == 96 for t, _, _ in N2SRC), "task count drift"
(OUT / "n2000.tex").write_text(latex_table(
    ["model", "aggregation"] + [f"$k$={k}" for k in _ks],
    [[model, AGGREGATION[key]] + [f"{c:.2f}" for c in _n2[t][key]]
     for t, model, key in N2SRC], align="ll" + "r" * len(_ks),
    caption="Coverage of nominal $95\\%$ subset-CATE intervals at $n=2000$ over "
            "96 tasks with 24 subsets each. The two CNP rows are separately "
            "trained models. Each model was trained once, with optimization "
            "settings that differ from Appendix~\\ref{app:protocol} and from "
            "each other, so the comparison is not controlled for optimization.",
    label="tab:n2000"))


# ---- Table 6, nsweep (Appendix F) --------------------------------------------
# CausalPFN on the clean polynomial family at five sample sizes, one model throughout.
_NS = []
for _tag in artifacts.NSWEEP_TAGS:
    _d = _inspec(f"res_{_tag}.json")
    _k = _d.get("breaking_k_draw")
    _NS.append([f"{_d['n']}", f"{_d['rho']:.4f}", f"{_d['m']:.2f}",
                # "---" for a rule that never crosses, as in the other tables
                f"{_k:.0f}" if _k == _k else "---",
                f"{1 + (_d['n'] - 1) * _d['rho']:.0f}",
                f"{_d['coverage_draw'][-1]:.2f} $\\pm$ "
                f"{_d['cov_se_clustered'][-1]:.2f}"])
(OUT / "nsweep.tex").write_text(latex_table(
    ["$n$", "$\\rho_S$", "$m_S$", KHAT, "$1+(n-1)\\rho_S$",
     "cov at $k$=$n$"],
    _NS, align="lrrrrr",
    caption="CausalPFN on the synthetic family of "
            "Appendix~\\ref{app:dgps} at five sample sizes, with subsets drawn "
            "uniformly at random, 64 tasks and 48 subsets per task; coverage "
            "at $k=n$ is given $\\pm$ one standard error clustered by task. "
            "The same "
            "CausalPFN model is used throughout, so no training choice varies "
            "down the table. "
            "$1+(n-1)\\rho_S$ is the design effect at the full sample; ``---'' "
            "means coverage never falls below nominal. " + KHAT_XREF,
    label="tab:nsweep"))

print(f"wrote the paper's five tables to {OUT}")
