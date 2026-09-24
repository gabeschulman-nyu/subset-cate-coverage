"""Emit LaTeX table fragments for main.tex from the same result artifacts.

python docs/latex/build_tex_tables.py     -> writes docs/latex/tables/*.tex
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = ROOT / "logs/c3read"
sys.path.insert(0, str(ROOT / "experiments"))
import artifacts  # noqa: E402

OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


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
    body = "\n".join(" & ".join(esc(c) for c in r) + " \\\\" for r in rows)
    notestr = f"\n\\vspace{{2pt}}\\footnotesize {note}" if note else ""
    # No automatic \resizebox. Scaling to \textwidth magnifies a narrow table
    # above body font size and shrinks a wide one below it; either way the
    # table stops matching the text. Tables render at their natural width in
    # \footnotesize, and anything that overflows loses columns instead.
    open_box = close_box = ""
    return (f"\\begin{{table}}[H]\n\\centering\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
            f"{open_box}\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}{close_box}{notestr}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def jload(n):
    return json.load(open(LOGS / n))


def has(n):
    return (LOGS / n).exists()


KS = (2, 16, 128, 500)
CELLS = [("latent NP", "poly", "log-uniform", "fw_latent_poly"),
         ("latent NP", "hetero", "log-uniform", "fw_latent_het"),
         ("CNP", "poly", "log-uniform", "fw_cnp_poly"),
         ("CNP", "hetero", "log-uniform", "fw_cnp_het"),
         ("CNP", "poly", "fixed $k$=50", [f"fw_cur50s{s}_b" for s in (0, 1, 2)]),
         ("CNP", "poly", "fixed $k$=500", [f"fw_cur500s{s}_b" for s in (0, 1, 2)]),
         ("TNP", "poly", "log-uniform", "fw_tnp_poly"),
         ("TNP", "poly", "fixed $k$=50", [f"fw_cur50s{s}_c3" for s in (0, 1, 2)]),
         ("TNP", "poly", "fixed $k$=500", [f"fw_cur500s{s}_c3" for s in (0, 1, 2)]),
         ("TNP (MDN)", "poly", "log-uniform", "fw_mdn_tnp_poly"),
         ("CNP (MDN)", "poly", "log-uniform", "fw_mdn_cnp_poly")]
rows = []
for arm, dgp, reg, keys in CELLS:
    keys = [keys] if isinstance(keys, str) else keys
    rs = [jload(k + ".json") for k in keys if has(k + ".json")]
    if not rs:
        continue
    rows.append([arm, dgp, reg]
                + [f"{np.mean([r['per_k'][str(k)]['cov_gate'] for r in rs]):.3f}" for k in KS])

INSPEC = ROOT / "logs/inspec"

# ---- The benchmark suite in numbers.
ORDER = ["HillA", "FriedmanConf", "IHDP", "HillB", "KangSchafer", "ACIC2016",
         "HillBHeavy"]
# Display names must match Fig. benchmarks and the Appendix A table; the JSON
# keys are the internal DGP identifiers.
DISPLAY = {"HillA": "Hill-A", "FriedmanConf": "Friedman", "IHDP": "IHDP",
           "HillB": "Hill-B", "KangSchafer": "Kang--Schafer",
           "ACIC2016": "ACIC 2016", "HillBHeavy": "Hill-B heavy-tailed"}
bl = json.load(open(ROOT / "logs/inspec" / "benchmark_ladder.json"))
rows4 = []
for dgp in ORDER:
    if dgp not in bl:
        continue
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
    if kb == kb:
        kb_s, kn_s = f"{kb:.0f}", f"{kb / n:.3f}"
    elif cov[0] < 0.95:
        kb_s, kn_s = "$<1$", "$<1/n$"
    else:
        kb_s, kn_s = "---", "---"
    rows4.append([DISPLAY[dgp], str(n), f"{rho:.4f}", f"{m:.2f}",
                  f"{v['nrmse_unit']:.2f}", f"{cov[-1]:.3f}", kb_s, kn_s])
(OUT / "legacy_breaking.tex").write_text(latex_table(
    ["benchmark", "$n$", "$\\rho_S$", "$m_S$", "nRMSE", "cov at $k$=$n$",
     "$k^*$", "$k^*/n$"], rows4,
    align="lrrrrrrr",
    caption="The benchmarks of Fig.~\\ref{fig:benchmarks} in numbers, from the same "
            "measurements. Throughout, $k$ is the number of units in the subset whose "
            "average effect is being reported, not a count of intervals: the $k$ columns "
            "give coverage of the aggregate interval at that subset size. Aggregation is "
            "the independent composition of \\eqref{eq:sigma}, subsets are drawn uniformly "
            "at random, and $40$ replications are used per benchmark. $k^*$ is the largest "
            "subset that still covers and $k^*/n$ expresses it as a fraction of the sample, "
            "which is the comparable quantity across benchmarks of different size. nRMSE is "
            "per-unit error over the spread of the true effect, so $1.00$ matches predicting "
            "the mean. Rows are ordered by $k^*/n$, which is not the order of $\\rho_S$: "
            "$m_S$ enters the threshold too, and Hill-B breaks earlier than IHDP on a "
            "fourteenth of its $\\rho_S$ because it has no cushion.",
    label="tab:legacy"))


PAIRS = [(2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 500)]
rows5 = []
for key, lab in (("fw_cnp_poly", "CNP (histogram) / poly"), ("fw_tnp_poly", "TNP (histogram) / poly"),
                 ("fw_mdn_cnp_poly", "CNP (MDN) / poly"), ("fw_mdn_tnp_poly", "TNP (MDN) / poly"),
                 ("fw_latent_poly", "latent NP / poly"), ("fw_cnp_het", "CNP (histogram) / hetero")):
    if not has(key + ".json"):
        continue
    r = jload(key + ".json")
    rat = [np.sqrt(r["per_k"][str(b)]["sigma2_hat"] / r["per_k"][str(a)]["sigma2_hat"]) for a, b in PAIRS]
    tot = np.sqrt(r["per_k"]["500"]["sigma2_hat"] / r["per_k"]["2"]["sigma2_hat"])
    rows5.append([lab, f"{np.mean(rat):.3f}", f"{tot:.3f}"])
rows5.append(["\\emph{uncorrelated errors (reference)}", "\\emph{0.707}", "\\emph{0.063}"])

# ---- Constructions at n=2000.
CONSTRUCTION = {
    "CNP, unit-level": "unit-level, independent",
    "latent NP": "unit-level, within-draw",
    "CNP": "direct decode",
}
# Sourced from the three n=2000 cells scored by two_path_ablation on one task
# stream, replacing docs/n2000_demonstration.csv whose harness no longer
# exists. The CSV's numbers were close (within 2-7 points, same shape); these
# are simply reproducible. The unit-level row is the vanilla arm's only path;
# the other two rows are each arm's own aggregate construction.
N2SRC = artifacts.N2000_SOURCES
_n2 = {t: json.loads((ROOT / f"logs/inspec/twopath_{t}.json").read_text())
       for t, _, _ in N2SRC}
_ks = _n2["n2krepro_nv"]["ks"]
assert all(_n2[t]["ks"] == _ks for t, _, _ in N2SRC), "n=2000 cells disagree on k"
assert all(_n2[t]["num_tasks"] == 96 for t, _, _ in N2SRC), "task count drift"
(OUT / "n2000.tex").write_text(latex_table(
    ["arm", "construction"] + [f"$k$={k}" for k in _ks],
    [[lab, CONSTRUCTION[lab]] + [f"{c:.3f}" for c in _n2[t][key]]
     for t, lab, key in N2SRC], align="ll" + "r" * len(_ks),
    caption="Coverage of nominal $95\\%$ subset-CATE intervals against subset "
            "size at $n=2000$, by how each arm builds the aggregate. "
            "``Unit-level, independent\'\' decodes a posterior per unit and "
            "averages draws across units at independent draw indices; "
            "``unit-level, within-draw\'\' averages within a shared latent "
            "draw; ``direct decode\'\' queries the subset effect in one "
            "forward pass. Only the independent row loses coverage as $k$ "
            "grows. All three arms share the backbone, head and embedding "
            "width, and are scored on one task stream of $96$ tasks with $24$ "
            "subsets per cell, so the rows differ in construction. They were "
            "trained at $n=2000$ outside the frozen recipe used elsewhere in "
            "this paper, and at differing learning rates and batch sizes "
            "($10^{-3}$/32, $10^{-3}$/16 and $3\\times10^{-4}$/32 for the "
            "three rows), so the comparison is not controlled for "
            "optimization. The separation it shows spans a factor of five in "
            "coverage, which no such difference accounts for, but we report "
            "it as corroboration at larger $n$ rather than as a controlled "
            "result. Single seed per arm.",
    label="tab:n2000"))

# ---- Sample-size sweep.
# ---- Appendix E: does more data fix it?  The shipped CausalPFN checkpoint
# applied to the clean polynomial family at five sample sizes. One fixed
# model throughout, so nothing here confounds n with a training choice.
_NS = []
for _tag in artifacts.NSWEEP_TAGS:
    _d = json.loads((ROOT / f"logs/inspec/res_{_tag}.json").read_text())
    _k = _d.get("breaking_k_draw")
    _NS.append([f"{_d['n']}", f"{_d['rho']:.4f}", f"{_d['m']:.1f}",
                f"{_k:.0f}" if _k == _k else "none",
                f"{(_d['n'] - 1) * _d['rho']:.0f}",
                f"{_d['coverage_draw'][-1]:.3f}",
                f"{_d['cov_se_clustered'][-1]:.3f}"])
(OUT / "nsweep.tex").write_text(latex_table(
    ["$n$", "$\\rho_S$", "$m_S$", "$k^*$", "$(n-1)\\rho_S$",
     "cov at $k$=$n$", "se"],
    _NS, align="lrrrrrr",
    caption="Sample size does not dissolve the problem. The shipped CausalPFN "
            "checkpoint on the synthetic family of Appendix~\\ref{app:dgps} at "
            "five sample sizes, with subsets drawn uniformly at random, $64$ "
            "tasks and $48$ subsets per task. One fixed checkpoint is used "
            "throughout, so no training choice of ours varies down the table. "
            "$\\rho_S$ falls as $n$ grows, but only as roughly $n^{-0.24}$: a "
            "sixteenfold increase in $n$ divides it by $1.7$, where the "
            "$n^{-1}$ decay that would make the design effect vanish would "
            "divide it by $16$. Because the reachable subset size grows with "
            "$n$ while $\\rho_S$ barely falls, the design effect available at "
            "the full sample, $(n-1)\\rho_S$, grows from $5$ to $46$ down the "
            "table. Coverage at $k=n$ nonetheless stays near nominal, because "
            "$m_S$ grows too, at roughly $n^{0.62}$: the checkpoint's per-unit "
            "intervals widen relative to its per-unit errors faster than the "
            "design effect eats the margin. It is the cushion that holds "
            "coverage up here, not the absence of the correlation.",
    label="tab:nsweep"))
print("wrote nsweep.tex")

# ---- Selection rules (Appendix D).
LADDERS = [
    ("clean", "ladder_propensity_m_n500.json", "n500", "propensity stratum", [
        ("propensity 1.00", "all units"), ("propensity 0.50", "bottom 50%"),
        ("propensity 0.25", "bottom 25%"), ("propensity 0.10", "bottom 10%")]),
    ("clean", "ladder_nested_n500.json", "n500", "nested covariate cell", [
        ("$X_1$>0", "$X_1>0$"),
        ("$X_1$>0, $X_2$>0", "$\\wedge\\, X_2>0$"),
        ("$X_1$>0, $X_2$>0, $X_3$>0", "$\\wedge\\, X_3>0$")]),
    ("clean", "ladder_neighbor_ball_m_n500.json", "n500", "covariate neighbourhood", [
        ("neighbor_ball 0.50", "nearest 50%"),
        ("neighbor_ball 0.25", "nearest 25%"),
        ("neighbor_ball 0.10", "nearest 10%")]),
    ("soft overlap", "ladder_propensity_m_ovsoft500.json", "ovsoft500",
     "propensity stratum", [
        ("propensity 1.00", "all units"), ("propensity 0.50", "bottom 50%"),
        ("propensity 0.25", "bottom 25%"), ("propensity 0.10", "bottom 10%")]),
    ("soft overlap", "ladder_nested_ovsoft500.json", "ovsoft500",
     "nested covariate cell", [
        ("$X_1$>0", "$X_1>0$"),
        ("$X_1$>0, $X_2$>0", "$\\wedge\\, X_2>0$"),
        ("$X_1$>0, $X_2$>0, $X_3$>0", "$\\wedge\\, X_3>0$")]),
]
rows_ss, seen = [], set()
for dgp, fn, tag, rule, rungs in LADDERS:
    if not (INSPEC / fn).exists():
        continue
    d = json.load(open(INSPEC / fn))[tag]
    for key, lab in rungs:
        if key not in d:
            continue
        v = d[key]
        rho = float(np.nanmedian(v["rho_by_k"]))
        m = float(np.nanmedian(v["m_by_k"]))
        kmax = max(v["ks"])
        kb = v["breaking_k"]
        cell = (dgp, lab)
        rows_ss.append([dgp if cell not in seen else "",
                        rule, lab, f"{rho:.3f}", f"{m:.1f}", f"{kmax}",
                        f"{(kmax - 1) * rho:.1f}",
                        f"{kb:.0f}" if kb == kb else "---"])
        seen.add(cell)
(OUT / "subset_selection.tex").write_text(latex_table(
    ["DGP", "rule", "subset", "$\\rho_S$", "$m_S$", "$k_{\\max}$",
     "$(k_{\\max}\\!-\\!1)\\rho_S$", "$k^*$"],
    rows_ss, align="lllrrrrr",
    caption="Selection rules on the shipped checkpoint at $n=500$. $k$ is the subset size and "
            "$k^*$ the smallest $k$ past which coverage stays below nominal; ``---'' means the "
            "rule never crosses within its pool. Aggregation is the independent composition of "
            "\\eqref{eq:sigma} throughout. Whether a rule breaks is decided by $m_S$ together with "
            "the reachable design effect $(k_{\\max}\\!-\\!1)\\rho_S$, not by $\\rho_S$ alone: "
            "narrowing a pool raises $\\rho_S$ but shrinks $k_{\\max}$, so the product is roughly "
            "fixed, and only the propensity ladder also drives $m_S$ down. Nested cells and "
            "covariate neighbourhoods reach comparable $\\rho_S$ on the clean DGP and never cross; "
            "under soft overlap, where $m_S$ is smaller to begin with, the same nested cells do "
            "cross.",
    label="tab:subset-selection"))

print(f"wrote 4 LaTeX table fragments to {OUT}")
