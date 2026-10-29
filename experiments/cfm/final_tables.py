"""Paper Tables 7 and 8 from eval_final.py output (logs/cfm/final/*.json), with
CausalPFN's numbers from logs/inspec/benchmark_ladder.json.

Writes docs/latex/tables/heads_coverage.tex (Table 7, coverage at k = n),
docs/latex/tables/heads_moments.tex (Table 8, m_S, rho_S, nRMSE and widths) and
docs/cfm_arxiv_numbers.md: coverage at k = n with seed ranges and standard
errors clustered by dataset (sd over datasets of the seed-averaged coverage
/ sqrt(40)), m_S and rho_S under all three estimators, and |predicted -
observed| coverage of unit-level aggregation over the k grid.

    python experiments/cfm/final_tables.py      (writes docs/cfm_arxiv_numbers.md and the two tables)
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm

BENCH = [("HillA", "Hill-A"), ("HillB", "Hill-B"), ("FriedmanConf", "Friedman"),
         ("KangSchafer", "Kang–Schafer"), ("IHDP", "IHDP"), ("ACIC2016", "ACIC 2016")]
HEADS = {"CNP": [f"v2_cnp_rand_s{i}" for i in range(3)], "latent NP": [f"v2_lathist_s{i}" for i in range(3)]}
AGG_NAME = {("CNP", "joint"): "CNP, subset-level output", ("CNP", "unit"): "CNP, unit-level aggregation",
            ("latent NP", "joint"): "latent NP, SCM-marginalized", ("latent NP", "unit"): "latent NP, unit-level aggregation"}


EST = ("", "_all", "_std")      # m_S / rho_S estimators: the paper's, all subsets, standardized


def load(run):
    return json.load(open(f"logs/cfm/final/{run}.json"))


def pred(ks, m, rho):
    R = m / (1 + (np.asarray(ks, float) - 1) * rho)
    return 2 * norm.cdf(1.96 * np.sqrt(R)) - 1, R


def at_n(ds, b, g, key="coverage"):
    """[seeds, datasets] outcomes at k = n."""
    return np.array([[row[-1] for row in d[b][key][g]] for d in ds])


def fmt_cov(c):
    per_seed = c.mean(1)
    per_data = c.mean(0)
    se = per_data.std(ddof=1) / np.sqrt(len(per_data))
    rng = f" [{per_seed.min():.2f}–{per_seed.max():.2f}]" if len(per_seed) > 1 else ""
    return f"{per_seed.mean():.2f} ± {se:.2f}{rng}"


def fmt_rng(v, p=2):
    v = np.asarray(v, float)
    return f"{v.mean():.{p}f} [{v.min():.{p}f}–{v.max():.{p}f}]"


TEX_BENCH = [("HillA", "Hill-A"), ("HillB", "Hill-B"), ("FriedmanConf", "Friedman"),
             ("KangSchafer", "Kang--Schafer"), ("IHDP", "IHDP"), ("ACIC2016", "ACIC 2016")]


def _short(v):
    """0.12 -> .12, 1.00 -> 1.00, for the bracketed seed ranges."""
    t = f"{v:.2f}"
    return t[1:] if t.startswith("0.") else t


def heads_coverage_tex(lad, ds):
    """Table 7: coverage at k = n, CausalPFN and both aggregations of both heads."""
    cols = [("CNP", "unit"), ("CNP", "joint"), ("latent NP", "unit"), ("latent NP", "joint")]
    L = [r"\begin{table}[tp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & \textbf{CausalPFN} & \multicolumn{2}{c}{\textbf{CNP head}} & "
         r"\multicolumn{2}{c}{\textbf{latent NP head}} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"\textbf{benchmark} & \textbf{unit-level} & \textbf{unit-level} & \textbf{SCM-marg.} & "
         r"\textbf{unit-level} & \textbf{SCM-marg.} \\",
         r"\midrule"]
    for b, lab in TEX_BENCH:
        cells = [f"{lad[b]['coverage'][-1]:.2f}"]
        for h, g in cols:
            c = at_n(ds[h], b, g).mean(1)                                 # per seed
            cells.append(f"{c.mean():.2f} {{\\scriptsize[{_short(c.min())}, {_short(c.max())}]}}")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    cells = [f"{sum(round(40 * lad[b]['coverage'][-1]) for b, _ in TEX_BENCH)}"]
    for h, g in cols:
        n = sum(at_n(ds[h], b, g).sum(1) for b, _ in TEX_BENCH)          # per seed
        cells.append(f"{n.mean():.0f} {{\\scriptsize[{n.min():.0f}, {n.max():.0f}]}}")
    L += [r"\addlinespace[2pt]", "covered, of 240 & " + " & ".join(cells) + r" \\", r"\bottomrule",
          r"\end{tabular}",
          r"\caption{Coverage of nominal $95\%$ intervals for the full-sample average ($k=n$) on the six "
          r"benchmarks, 40 datasets each. CausalPFN's column is the full-sample column of Table~\ref{tab:legacy}. "
          r"The heads' columns give the mean over three seeds, with the range of the per-seed values in brackets. "
          r"Standard errors clustered by dataset are at most $" + f"{max_se(ds):.2f}" + r"$ for the heads. The last "
          r"row counts datasets whose full-sample interval contains the true value.}",
          r"\label{tab:heads-coverage}", r"\end{table}"]
    return "\n".join(L) + "\n"


def max_se(ds):
    """Largest standard error, clustered by dataset, of the heads' coverage at k = n."""
    out = 0.0
    for h in ds:
        for g in ("unit", "joint"):
            for b, _ in TEX_BENCH:
                per_data = at_n(ds[h], b, g).mean(0)
                out = max(out, per_data.std(ddof=1) / np.sqrt(len(per_data)))
    return out


def heads_moments_tex(lad, ds, cp):
    """Table 8: m_S, rho_S (the paper's estimator), nRMSE and widths at k = n."""
    L = [r"\begin{table}[tp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"\textbf{benchmark} & \textbf{model} & \textbf{$m_S$} & \textbf{$\rho_S$} & \textbf{nRMSE} & "
         r"\textbf{width, own} & \textbf{width, unit-level} \\",
         r"\midrule"]
    for i, (b, lab) in enumerate(TEX_BENCH):
        r = lad[b]
        if i:
            L.append(r"\addlinespace[2pt]")
        L.append(f"{lab} & CausalPFN & {np.nanmedian(r['m_by_k']):.2f} & {np.nanmedian(r['rho_by_k']):.4f} & "
                 f"{r['nrmse_y']:.2f} & --- & {np.mean(at_n([cp], b, 'unit_cal', 'width')):.3f} " + r"\\")
        for h, name in (("CNP", "CNP head"), ("latent NP", "latent NP head")):
            d = ds[h]
            vals = [np.mean([x[b]["m_S"] for x in d]), np.mean([x[b]["rho_S"] for x in d]),
                    np.mean([x[b]["nrmse_y"] for x in d]),
                    np.mean([np.mean(at_n([x], b, "joint", "width")) for x in d]),
                    np.mean([np.mean(at_n([x], b, "unit", "width")) for x in d])]
            L.append(f" & {name} & {vals[0]:.2f} & {vals[1]:.4f} & {vals[2]:.2f} & {vals[3]:.3f} & "
                     f"{vals[4]:.3f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Unit-level calibration, error correlation, accuracy, and interval width on the six "
          r"benchmarks. $m_S$ and $\rho_S$ use the estimator of Appendix~\ref{app:protocol}, from each model's "
          r"unit-level means and standard deviations (for the CNP head, the head queried with each unit alone), "
          r"and nRMSE is defined as in Table~\ref{tab:legacy}. CausalPFN's $m_S$, $\rho_S$, and nRMSE are those "
          r"of Table~\ref{tab:legacy}. Widths are mean widths of the full-sample interval in units of "
          r"$\mathrm{sd}(y)$: \emph{own} is the head's SCM-marginalized interval, and \emph{unit-level} is "
          r"unit-level aggregation from the same model. CausalPFN's width is computed with the whole sample as "
          r"context, at the temperatures fitted for Table~\ref{tab:legacy}. Heads: mean over three seeds.}",
          r"\label{tab:heads-moments}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    lad = json.load(open("logs/inspec/benchmark_ladder.json"))
    ds = {h: [load(r) for r in runs] for h, runs in HEADS.items()}
    cp = load("causalpfn_paperT")       # CausalPFN scored by eval_final.py at the paper's fitted temperatures
    L = ["# Paper numbers from the consolidated evaluation", "",
         "Generated by `experiments/cfm/final_tables.py` from `logs/cfm/final/*.json` "
         "(`experiments/cfm/eval_final.py`). Heads: CNP `v2_cnp_rand_s{0,1,2}`, latent NP "
         "`v2_lathist_s{0,1,2}`, on the frozen CausalPFN. 40 datasets per benchmark; subsets as in every "
         "other script (48 per size, one at k = n). Coverage: mean over the three seeds ± standard error "
         "clustered by dataset [range of the three per-seed values]. CausalPFN (paper): the paper's Table / "
         "Figure 1 numbers, `logs/inspec/benchmark_ladder.json`, unchanged.", ""]

    L += ["## 1. Coverage of nominal 95% intervals at k = n", "",
          "| benchmark | n | CausalPFN, unit-level (paper) | " + " | ".join(AGG_NAME.values()) + " |",
          "|---|---:|---:|" + "---:|" * len(AGG_NAME)]
    for b, lab in BENCH:
        r = lad[b]
        cells = [f"{r['coverage'][-1]:.2f} ± {r['cov_se_clustered'][-1]:.2f}"]
        cells += [fmt_cov(at_n(ds[h], b, g)) for h, g in AGG_NAME]
        L.append(f"| {lab} | {r['n']} | " + " | ".join(cells) + " |")
    tot = ["| datasets covered (of 240) | | " + f"{sum(round(40 * lad[b]['coverage'][-1]) for b, _ in BENCH)}"]
    for h, g in AGG_NAME:
        per_seed = sum(at_n(ds[h], b, g).sum(1) for b, _ in BENCH)
        tot.append(f"{per_seed.mean():.0f} [{per_seed.min():.0f}–{per_seed.max():.0f}]")
    L += [" | ".join(tot) + " |", ""]

    L += ["## 2. Per-unit quantities and interval width", "",
          "m_S and rho_S come from each method's per-unit predictive mean and sd (CNP: queried at |S| = 1; "
          "latent NP: marginal over z), three ways: **paper**, exactly as the paper's table computes "
          "CausalPFN's (benchmark_ladder.py: errors in outcome units pooled over datasets; per size, ONE subset "
          "per dataset; median over sizes); **all**, the same from all 48 subsets per dataset and size; "
          "**std**, all subsets with each dataset's errors in its own sd(y) units (the estimator behind the "
          "earlier figures). CausalPFN (paper) uses the paper's own numbers in the paper column; its all/std "
          "values (*) come from the heads' pass at the temperatures the paper's run fitted. Heads: seed mean "
          "[range].", "",
          "| benchmark | method | m_S paper | m_S all | m_S std | rho_S paper | rho_S all | rho_S std |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for b, lab in BENCH:
        r, c = lad[b], cp[b]
        L.append(f"| {lab} | CausalPFN | {np.nanmedian(r['m_by_k']):.2f} | {c['m_S_all']:.2f}* | {c['m_S_std']:.2f}* | "
                 f"{np.nanmedian(r['rho_by_k']):.4f} | {c['rho_S_all']:.4f}* | {c['rho_S_std']:.4f}* |")
        for h, d in ds.items():
            L.append(f"| | {h} | " + " | ".join(fmt_rng([x[b][f"m_S{e}"] for x in d]) for e in EST) + " | "
                     + " | ".join(fmt_rng([x[b][f"rho_S{e}"] for x in d], 4) for e in EST) + " |")
    L += ["", "nRMSE: per-unit RMSE / sd(y), within each dataset, averaged over datasets (the paper's table "
          "column). Width: mean over the 40 datasets of the 95% interval width at k = n, in sd(y) units "
          "(CausalPFN's from the heads' pass at the paper's temperatures).", "",
          "| benchmark | method | nRMSE | width, own aggregate | width, unit-level aggregation |",
          "|---|---|---:|---:|---:|"]
    for b, lab in BENCH:
        L.append(f"| {lab} | CausalPFN | {lad[b]['nrmse_y']:.3f} | — | "
                 f"{np.mean(at_n([cp], b, 'unit_cal', 'width')):.3f} |")
        for h, d in ds.items():
            wj = [np.mean(at_n([x], b, "joint", "width")) for x in d]
            wu = [np.mean(at_n([x], b, "unit", "width")) for x in d]
            L.append(f"| | {h} | {fmt_rng([x[b]['nrmse_y'] for x in d], 3)} | {fmt_rng(wj, 3)} | {fmt_rng(wu, 3)} |")
    L.append("")

    L += ["## 3. Predicted vs observed coverage of unit-level aggregation", "",
          "|2 Phi(1.96 sqrt(m_S / (1 + (k - 1) rho_S))) - 1 - observed| over the 16 sizes of the k grid: median "
          "/ max. Each run against its own prediction; heads: mean over three runs. CausalPFN, paper column: "
          "Figure 1's curve against its dashed prediction; all/std columns: the heads' pass. The median is "
          "diluted by the sizes where both curves sit near 1; the max is where they part. Last rows: pooled over "
          "benchmarks and runs, median over all points, and over the points with R < 1 (the restriction of "
          "the paper's pooled design-effect figure).", "",
          "| benchmark | method | paper estimator | all subsets | standardized |", "|---|---|---:|---:|---:|"]
    pooled = {(m, e): ([], []) for m in ["CausalPFN", *ds] for e in EST}

    def err(key, ks, obs, m, rho):
        p, R = pred(ks, m, rho)
        e = np.abs(np.asarray(obs) - p)
        pooled[key][0].extend(e)
        pooled[key][1].extend(e[R < 1])
        return np.median(e), e.max()

    for b, lab in BENCH:
        r, c = lad[b], cp[b]
        obs_c = np.mean(c["coverage"]["unit_cal"], 0)
        cells = [err(("CausalPFN", ""), r["ks"], r["coverage"], np.nanmedian(r["m_by_k"]), np.nanmedian(r["rho_by_k"]))]
        cells += [err(("CausalPFN", e), c["ks"], obs_c, c[f"m_S{e}"], c[f"rho_S{e}"]) for e in EST[1:]]
        L.append(f"| {lab} | CausalPFN | " + " | ".join(f"{a:.3f} / {m:.2f}" for a, m in cells) + " |")
        for h, d in ds.items():
            cells = []
            for e in EST:
                v = np.array([err((h, e), x[b]["ks"], np.mean(x[b]["coverage"]["unit"], 0), x[b][f"m_S{e}"],
                                  x[b][f"rho_S{e}"]) for x in d])
                cells.append(f"{v[:, 0].mean():.3f} / {v[:, 1].mean():.2f}")
            L.append(f"| | {h} | " + " | ".join(cells) + " |")
    for m in ["CausalPFN", *ds]:
        L.append(f"| pooled | {m} | " + " | ".join(
            f"{np.median(pooled[m, e][0]):.3f} (R<1: {np.median(pooled[m, e][1]):.3f}, {len(pooled[m, e][1])} pts)"
            for e in EST) + " |")
    L.append("")

    L += ["## Check: CausalPFN re-scored in the heads' pass", "",
          "The same model on the heads' single full-context pass (eval_final.py --method causalpfn "
          "--paper_temperatures), at the temperatures the paper's run fitted and stored; same estimators. "
          "Figure 1 uses the public API, which on ACIC 2016 conditions each window of units on its nearest "
          "neighbours rather than the whole sample.", "",
          "| benchmark | coverage at k = n: paper / this pass | m_S: paper / this pass | "
          "rho_S: paper / this pass | nRMSE: paper / this pass |", "|---|---|---|---|---|"]
    for b, lab in BENCH:
        r, c = lad[b], cp[b]
        L.append(f"| {lab} | {r['coverage'][-1]:.2f} / {at_n([cp], b, 'unit_cal').mean():.2f} | "
                 f"{np.nanmedian(r['m_by_k']):.2f} / {c['m_S']:.2f} | {np.nanmedian(r['rho_by_k']):.4f} / "
                 f"{c['rho_S']:.4f} | {r['nrmse_y']:.3f} / {c['nrmse_y']:.3f} |")
    L.append("")
    open("docs/cfm_arxiv_numbers.md", "w").write("\n".join(L))
    open("docs/latex/tables/heads_coverage.tex", "w").write(heads_coverage_tex(lad, ds))
    open("docs/latex/tables/heads_moments.tex", "w").write(heads_moments_tex(lad, ds, cp))
    print("\n".join(L))


if __name__ == "__main__":
    main()
