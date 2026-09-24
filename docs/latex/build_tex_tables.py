"""Write the paper's six tables from the measurement artifacts.

    python docs/latex/build_tex_tables.py   -> docs/latex/tables/*.tex

Tables are emitted in paper order: benchmarks (Appendix A), threearm
(Appendix C), subset_selection (Appendix D), legacy_breaking, n2000 and
nsweep (Appendix F). Which artifact supplies each setting is decided in
experiments/artifacts.py, not here.
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

# The tables report the MEASURED crossing. Eq. (kstar) gives the PREDICTED one,
# 1 + (m_S - 1)/rho_S, and the two differ by up to 40% on the synthetic family,
# so they must not share a symbol: a reader who plugs a table's rho_S and m_S
# into the equation would otherwise conclude the theory is off. Every producer
# (benchmark_ladder.kstar, inspec_stage3.first_below) computes it the same way.
KHAT = "$\\hat k^*$"
KHAT_DEF = ("$\\hat k^*$ is the subset size at which observed coverage "
            "first falls below nominal, interpolated in $\\log k$")
KHAT_XREF = "$\\hat k^*$ is defined in Section~\\ref{sec:benchmarks}."


def esc(s):
    # Escape only OUTSIDE math mode: a cell like "$X_1>0$" must keep its
    # subscript, and escaping inside $...$ silently produces "$X\_1>0$".
    out, math = [], False
    for part in str(s).split("$"):
        out.append(part if math else
                   part.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%"))
        math = not math
    return "$".join(out)


def latex_table(headers, rows, align=None, caption="", label="", note=""):
    align = align or ("l" + "r" * (len(headers) - 1))
    head = " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\"
    # A None row is a block separator, not data: it renders as vertical space
    # so a table that groups rows can breathe without a rule through it.
    ncol = len(headers)

    def render(r):
        if r is None:                       # block separator
            return "\\addlinespace[3pt]"
        if isinstance(r, tuple) and r and r[0] == "GROUP":
            # A group header spanning the table, so the grouping is visible
            # without spending a column on it.
            return (f"\\multicolumn{{{ncol}}}{{l}}{{\\itshape {esc(r[1])}}}"
                    " \\\\")
        return " & ".join(esc(c) for c in r) + " \\\\"

    body = "\n".join(render(r) for r in rows)
    notestr = f"\n\\vspace{{2pt}}\\footnotesize {note}" if note else ""
    # No automatic \resizebox. Scaling to \textwidth magnifies a narrow table
    # above body font size and shrinks a wide one below it; either way the
    # table stops matching the text. Tables render at their natural width in
    # \footnotesize, and anything that overflows loses columns instead.
    return (f"\\begin{{table}}[H]\n\\centering\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}{notestr}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


INSPEC = ROOT / "logs/inspec"


def _inspec(name):
    with open(INSPEC / name) as f:
        return json.load(f)


# The benchmark measurements, read by Tables benchmarks, threearm and legacy.
bl = _inspec(f"{artifacts.BENCHMARK}.json")


# ---- Table benchmarks (Appendix A) ------------------------------------------
# The sample sizes are read from the measurements so they cannot drift from what was actually evaluated; the
# covariate counts and provenance are metadata, verified once against the
# loaders in benchmarks/ rather than measured.
BENCH_META = [
    ("HillA", "Hill-A", "simulated $\\mathcal{N}(0, I)$", 25,
     "\\citet{hill2011} setting A"),
    ("FriedmanConf", "Friedman", "simulated $\\mathrm{U}(0,1)$", 10,
     "\\citet{friedman1991} surface, confounded"),
    ("IHDP", "IHDP", "real, IHDP study", 25,
     "\\citet{hill2011}; splits of \\citet{shalit2017}"),
    ("HillB", "Hill-B", "simulated $\\mathcal{N}(0, I)$", 25,
     "\\citet{hill2011} setting B"),
    ("KangSchafer", "Kang--Schafer", "simulated, nonlinear in 4 latents", 4,
     "\\citet{kang2007}"),
    ("ACIC2016", "ACIC 2016", "real, competition covariate table", 58,
     "\\citet{dorie2019}"),
]
(OUT / "benchmarks.tex").write_text(latex_table(
    ["benchmark", "covariates", "$n$", "$d$", "design"],
    [[disp, cov, str(bl[key]["n"]), str(d), design]
     for key, disp, cov, d, design in BENCH_META],
    align="llrrl",
    caption="The six benchmarks, ordered as in Figure~\\ref{fig:benchmarks}; "
            "$d$ is the number of covariates supplied to the estimator. IHDP "
            "uses the \\texttt{ihdp\\_npci\\_1-100} splits and ACIC 2016 the "
            "official \\texttt{aciccomp2016} generator, and the four simulated "
            "designs are reimplemented from the cited papers. We use 40 "
            "replications of each: splits 1--40 of 100 for IHDP, settings "
            "1--40 of 77 for ACIC 2016, and seeds 1--40 for the simulated "
            "designs.",
    label="tab:benchmarks"))


# ---- Table threearm (Appendix C): three intervals from the same outputs ----
def threearm_settings():
    """(title, m_S, ks, reported, rescaled, +design effect) per setting, each
    read from the one run that supplies that setting everywhere in the paper
    (see experiments/artifacts.py). CausalPFN first, our model last.
    """
    out = []
    # CausalPFN, clean (s=1), whole sample: the n-sweep's n=500 run, the same
    # run as Table nsweep's n=500 row and Table subset-selection's baseline.
    r = _inspec("res_n500.json")
    out.append(("CausalPFN|$s=1$, all units", r["m"], np.array(r["ks"], float),
                r["coverage_draw"], r["cov_recal"], r["cov_full"]))
    # CausalPFN, soft overlap (s=4), bottom 10% of the propensity: the rung of
    # Table subset-selection's soft-overlap ladder.
    v = _inspec("ladder_propensity_m_ovsoft500.json")["ovsoft500"]["propensity 0.10"]
    out.append(("CausalPFN|$s=4$, bottom 10%", v["m_bar"],
                np.array(v["ks"], float), v["coverage"], v["cov_recal"],
                v["cov_full"]))
    # CausalPFN on IHDP: the benchmark run of Figure benchmarks and Table
    # legacy, reporting the same median m_S.
    b = bl["IHDP"]
    out.append(("CausalPFN|IHDP", float(np.nanmedian(b["m_by_k"])),
                np.array(b["ks"], float), b["coverage"], b["cov_recal"],
                b["cov_full"]))
    # Our Gaussian-head CNP at n=500: Figure 2's three seeds, averaged. Each
    # seed's corrections use that seed's own m_S and rho_S; the m_S shown is
    # the mean of the three.
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
            rows.append((title, m, int(ks[ok][-1]), arm_label[key],
                         v[ok][0], v[ok][-1]))
    body = []
    for i, (t, m, k1, arm, c0, c1) in enumerate(rows):
        # One setting spans three rows; name it once so the eye groups them.
        # "model|data": one column each, so a row never leaves the reader
        # guessing which model produced it or which data it ran on. A % in the
        # title must be escaped or LaTeX comments out the rest of the row.
        model, data = t.split("|")
        safe = data.replace("%", chr(92) + "%")
        head = (f"{model} & {safe} & {m:.2f} & {k1}"
                if i % 3 == 0 else " &  &  & ")
        body.append(f"{head} & {arm} & {c0:.2f} & {c1:.2f} \\\\")
        if i % 3 == 2 and i + 1 < len(rows):
            body.append("\\addlinespace[2pt]")
    return (
        "\\begin{table}[H]\n\\centering\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{llrrlrr}\n\\toprule\n"
        "\\textbf{model} & \\textbf{data} & \\textbf{$m_S$} & "
        "\\textbf{$k_{\\max}$} & "
        "\\textbf{interval} & \\textbf{cov, smallest $k$} & "
        "\\textbf{cov, $k_{\\max}$} \\\\\n\\midrule\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Coverage of the three intervals of "
        "Appendix~\\ref{app:correction} at the smallest and largest subset in "
        "each setting. The \\emph{$+$ design effect} interval uses $\\rho_S$ "
        "and $m_S$ measured against the true effects and is not available in "
        "practice.}\n"
        "\\label{tab:threearm}\n\\end{table}\n")


(OUT / "threearm.tex").write_text(threearm_table())

# ---- Table subset-selection (Appendix D) ------------------------------------
# One selection-rule comparison replacing three overlapping presentations of
# the same result. The column that decides whether a rule breaks is not rho_S
# but the reachable design effect (k_max - 1) rho_S paired with m_S, since
# R = m_S / [1 + (k-1) rho_S] and k is bounded by the pool.
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
            # The unselected baseline is the whole-sample setting, which the
            # n-sweep run supplies everywhere else (Tables threearm, nsweep);
            # the ladder's own whole-sample rung is a second run of it.
            r = _inspec(f"res_{tag}.json")
            rho, m, kmax, kb = r["rho"], r["m"], r["n"], r["breaking_k_draw"]
        else:
            rho = float(np.nanmedian(v["rho_by_k"]))
            m = float(np.nanmedian(v["m_by_k"]))
            kmax = max(v["ks"])
            kb = v["breaking_k"]
        # DGP and rule are printed once per block. Seventeen rows repeating
        # "clean / propensity stratum" is most of why this was hard to scan.
        block = (dgp, rule)
        first = block not in seen
        if first:
            if seen:
                rows_ss.append(None)      # rendered as \addlinespace
            # A spanning italic header per group, rather than two columns of
            # mostly-blank continuation cells.
            rows_ss.append(("GROUP", f"{dgp}, {rule}"))
        # "all units" is the unselected baseline, not a stratum of the rule
        # above it; saying so stops it reading as the widest rung of a ladder.
        shown = "all units (baseline)" if lab == "all units" else lab
        rows_ss.append([shown, f"{rho:.3f}",
                        # two decimals so the cell shared with Tab. threearm
                        # prints the same number in both places
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
            "$48$ subsets per task. $k_{\\max}$ is the size of the pool a rule "
            "selects and $1+(k_{\\max}\\!-\\!1)\\rho_S$ the largest design "
            "effect reachable within it; ``---'' means coverage never falls "
            "below nominal within the pool. " + KHAT_XREF,
    label="tab:subset-selection"))


# ---- Table legacy (Appendix F): the benchmark suite in numbers --------------
ORDER = ["HillA", "FriedmanConf", "IHDP", "HillB", "KangSchafer", "ACIC2016",
         "HillBHeavy"]
# Display names must match Fig. benchmarks and the Appendix A table; the JSON
# keys are the internal DGP identifiers.
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
    # Eleven columns needed \resizebox well below 90%, which is what the
    # style guide says to fix by dropping columns rather than shrinking. The
    # intermediate k columns are redundant with Fig. main, which shows coverage
    # at every k; what the table adds is the per-benchmark constants and the
    # crossing. Decimals are fixed per column: coverage 3, rho_S 4, m_S 2,
    # k* integer.
    # A missing k* means two opposite things: Hill-A never falls below
    # nominal, while Hill-B-heavy is already below it at k=1 and so has no
    # safe subset size at all. Printing "---" for both hides that, so the
    # second case is marked "$<1$".
    # k* and k*/n restate one fact, so they share a column as "85 (11%)".
    if kb == kb:
        # To one decimal below 10%: at two decimals of a fraction,
        # Kang--Schafer (0.014) and ACIC (0.006) both print 0.01, which hides
        # the ordering the rows are sorted by.
        frac = 100.0 * kb / n
        kb_s = f"{kb:.0f} ({frac:.0f}%)" if frac >= 10 else f"{kb:.0f} ({frac:.1f}%)"   # esc() escapes the %
    elif cov[0] < 0.95:
        kb_s = "$<1$"
    else:
        kb_s = "---"
    # The full-sample coverages rest on 40 replications with one subset each,
    # so they carry a standard error of six to eight points. Printing three
    # decimals next to that would imply precision the estimate does not have.
    se = v["cov_se_clustered"][-1]
    rows4.append([DISPLAY[dgp], str(n), f"{rho:.4f}", f"{m:.2f}",
                  f"{v['nrmse_unit']:.2f}", f"{cov[-1]:.2f}", f"{se:.2f}",
                  kb_s])
(OUT / "legacy_breaking.tex").write_text(latex_table(
    ["benchmark", "$n$", "$\\rho_S$", "$m_S$", "nRMSE", "cov at $k$=$n$",
     "se", "$\\hat k^*$ ($\\hat k^*/n$)"], rows4,
    align="lrrrrrrr",
    caption="The benchmarks of Figure~\\ref{fig:benchmarks}: CausalPFN, "
            "unit-level aggregation, subsets drawn uniformly at "
            "random, 40 replications per benchmark. " + KHAT_DEF + ", with "
            "$\\hat k^*/n$ as a percentage of the sample; ``---'' means coverage "
            "never falls below nominal and $<1$ that it is already below at the "
            "smallest subset. nRMSE is per-unit error divided by the spread of "
            "the true effect, so $1.00$ matches predicting the mean. Rows are "
            "ordered by $\\hat k^*/n$.",
    label="tab:legacy"))


# ---- Table n2000 (Appendix F) -----------------------------------------------
# Sourced from the three n=2000 cells scored by two_path_ablation on one task
# stream. The aggregation a row reports is fixed by which path it reads: the
# unit-level path (cov_B) or the model's own SCM-marginalized one (cov_A).
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


# ---- Table nsweep (Appendix F): does more data fix it? ----------------------
# CausalPFN applied to the clean polynomial family at five sample sizes. One fixed
# model throughout, so nothing here confounds n with a training choice.
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
            "uniformly at random, 64 tasks and 48 subsets per task. One checkpoint "
            "is used throughout, so no training choice varies down the table. "
            "$1+(n-1)\\rho_S$ is the design effect at the full sample; ``---'' "
            "means coverage never falls below nominal. " + KHAT_XREF,
    label="tab:nsweep"))

print(f"wrote the paper's six tables to {OUT}")
