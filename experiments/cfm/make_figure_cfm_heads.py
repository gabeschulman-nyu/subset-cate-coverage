"""cfm_heads_vs_causalpfn.pdf (Figure 3): coverage of CausalPFN's unit-level
aggregation and of the two heads' SCM-marginalized aggregation.

Left: CausalPFN, temperature-calibrated, unit-level aggregation (Figure 1's
curves, logs/inspec/benchmark_ladder.json). Middle and right: direct-decode CNP
and latent NP heads on CausalPFN as released, three seeds each, from
eval_final.py (logs/cfm/final/*.json) on the same datasets, k grid and targets.
The CNP's direct decode of theta_S is labelled SCM-marginalized as its amortized
form. Lines are means over seeds, bands the range over seeds. Figure 1 uses
CausalPFN's public API (neighbour windows on ACIC 2016); the heads use one
full-context pass.

    python experiments/cfm/make_figure_cfm_heads.py [--out PATH]
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

OUT = "docs/figures/cfm_heads_vs_causalpfn.pdf"
CNP = [f"logs/cfm/final/v2_cnp_rand_s{i}.json" for i in range(3)]
LATENT = [f"logs/cfm/final/v2_lathist_s{i}.json" for i in range(3)]
SHOW = [("HillA", "Hill-A"), ("FriedmanConf", "Friedman"), ("IHDP", "IHDP"),
        ("HillB", "Hill-B"), ("KangSchafer", "Kang–Schafer"),
        ("ACIC2016", "ACIC 2016")]
PANELS = [("CausalPFN, unit-level", "logs/inspec/benchmark_ladder.json", "coverage"),
          ("CNP head (ours)\nSCM-marginalized, 3 seeds", CNP, "final:joint"),
          ("latent-NP head (ours)\nSCM-marginalized, 3 seeds", LATENT, "final:joint")]


def curve(r, key):
    """Coverage over the k grid, averaged over datasets for eval_final.py results."""
    if key.startswith("final:"):
        return np.mean(r["coverage"][key[6:]], 0)
    return np.asarray(r[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH, 2.75), sharey=True, sharex=True)
    for ax, (title, path, key) in zip(axes, PANELS, strict=True):
        ds = [json.load(open(p)) for p in (path if isinstance(path, list) else [path])]
        for i, (b, lab) in enumerate(SHOW):
            r = ds[0][b]
            ks = np.array(r["ks"], float)
            cov = np.array([curve(d[b], key) for d in ds])
            if len(ds) > 1:
                ax.fill_between(ks / r["n"], cov.min(0), cov.max(0), color=figstyle.OKABE_ITO[i], alpha=0.12,
                                lw=0, zorder=2)
            ax.plot(ks / r["n"], cov.mean(0), color=figstyle.OKABE_ITO[i], ls=figstyle.STYLES[i],
                    marker=figstyle.MARKERS[i], zorder=4, label=rf"{lab} ($n$={r['n']})")
        ax.axhline(0.95, color=figstyle.GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_title(title)
    axes[0].set_ylabel("coverage of nominal 95% interval")
    axes[0].set_ylim(0.10, 1.03)
    # fig.supxlabel would collide with the legend below the axes.
    axes[1].set_xlabel("fraction of the sample aggregated, $k/n$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, columnspacing=2.0,
               handlelength=2.0, handletextpad=0.5)
    figstyle.save(fig, a.out)
    for title, path, key in PANELS:
        ds = [json.load(open(p)) for p in (path if isinstance(path, list) else [path])]
        print(f"{title.replace(chr(10), ' '):48s} coverage at k = n (mean [min, max]): " + "  ".join(
            f"{lab} {np.mean([curve(d[b], key)[-1] for d in ds]):.2f} [{min(curve(d[b], key)[-1] for d in ds):.2f}, "
            f"{max(curve(d[b], key)[-1] for d in ds):.2f}]" for b, lab in SHOW))


if __name__ == "__main__":
    main()
