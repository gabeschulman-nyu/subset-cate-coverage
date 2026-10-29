"""Which result files the paper's figures and tables read.

Used by make_figure_collapse, make_figure_twopath, docs/latex/build_tex_tables
and the release build. Two selection rules:

Final configurations only: the Gaussian-head CNP and the latent NP, three seeds
each, trained as in Appendix B, and CausalPFN as released. The one exception is
the n=2000 table, whose models were trained differently (see its caption).

One run per setting, where a setting is (model, training seed, selection rule,
pool fraction, evaluation DGP). A rule at pool fraction 1.0 selects the whole
sample and is the random-subset setting.
"""

from __future__ import annotations

import glob
import json
import os

INSPEC = "logs/inspec"

# Final checkpoints by sweep directory, and whether each was trained on
# per-task standardized tasks. A cell counts only if it was scored the same
# way. Cells without a `standardize` field were scored standardized.
TRAINED_STANDARDIZED = {"sweep_gauss1_s": False, "sweep_latfix_s": True}
FINAL_CHECKPOINTS = tuple(TRAINED_STANDARDIZED)

# Figure 2's cells, also the run used for each checkpoint's random-subset
# setting.
FIGURE2_CNP = ("g1_s0", "g1_s1", "g1_s2")
FIGURE2_LATENT = ("lat_s0", "lat_s1", "lat_s2")
FIGURE2_CELLS = FIGURE2_CNP + FIGURE2_LATENT

# The n=2000 table (Table 5): (cell tag, model, path reported).
N2000_SOURCES = (("n2krepro_nv", "CNP", "cov_B"),
                 ("n2krepro_a", "latent NP", "cov_A"),
                 ("n2krepro_b", "CNP", "cov_A"))
N2000_CELLS = tuple(t for t, _, _ in N2000_SOURCES)

# CausalPFN on the synthetic family, random subsets, one run per (n, s)
# setting, for Figure 4. Where a setting was scored twice the tables' run is
# used: n1000 over hi_clean, ov40_n1000 over hi_ovsoft, ovsoft500 over
# ov40_n500.
SCATTER_TAGS = (("res_n250", "res_n500", "res_n1000", "res_n2000",
                 "res_ovsoft500")
                + tuple(f"res_ov20_n{n}" for n in (250, 500, 1000, 2000))
                + tuple(f"res_ov40_n{n}" for n in (250, 1000, 2000))
                + tuple(f"res_ov60_n{n}" for n in (250, 500, 1000, 2000)))

# Sample sizes of the n-sweep table (Table 6), in table order. Also the
# source of every random-subset CausalPFN setting at s=1 in the tables.
NSWEEP_TAGS = ("n125", "n250", "n500", "n1000", "n2000")

# CausalPFN pool-fraction ladders at n=500 (Figure 4, Table 3). Their
# whole-sample rungs duplicate res_n500 and res_ovsoft500 and are not read.
SHIPPED_LADDERS = ("ladder_propensity_m_n500",
                   "ladder_effect_oracle_m_n500",
                   "ladder_neighbor_ball_m_n500")
TABLE3_LADDERS = ("ladder_propensity_m_n500", "ladder_nested_n500",
                  "ladder_neighbor_ball_m_n500", "ladder_propensity_m_ovsoft500",
                  "ladder_nested_ovsoft500")

# CausalPFN on the benchmarks (Figures 1 and 3, Tables 2, 4, 7 and 8).
BENCHMARK = "benchmark_ladder"


def is_smoke(tag: str) -> bool:
    """True for probe cells."""
    return tag == "cputime" or "smoke" in tag


def is_whole_sample_rung(mode: str) -> bool:
    """A ladder rung at pool fraction 1.0 selects every unit."""
    return mode.endswith(" 1.00")


def setting(j):
    """The setting a twopath cell measures, independent of its tag and grid."""
    frac = j.get("pool_frac")
    rule = j.get("selection")
    if rule == "random" or frac == 1.0:
        rule, frac = "random", None
    return (j.get("checkpoint"), rule, frac, bool(j.get("eval_ood")))


def twopath_cells(root: str = ".", recipe_only: bool = True, exempt=()):
    """Yield (tag, parsed) for every twopath cell a figure or table reads.

    Selects final-configuration checkpoints scored with their training
    preprocessing, skips smoke cells, and takes one run per setting: Figure 2's
    run for the random-subset setting, otherwise the run on the densest k grid.
    Cells in `exempt` (the n=2000 table's) are yielded unconditionally.
    """
    runs = []
    for f in sorted(glob.glob(os.path.join(root, INSPEC, "twopath_*.json"))):
        tag = os.path.basename(f)[len("twopath_"):-len(".json")]
        if is_smoke(tag):
            continue
        with open(f) as fh:
            j = json.load(fh)
        if tag in exempt:
            yield tag, j
            continue
        family = next((c for c in FINAL_CHECKPOINTS
                       if c in j.get("checkpoint", "")), None)
        if family is None:
            continue
        if j.get("standardize", True) != TRAINED_STANDARDIZED[family]:
            continue
        if recipe_only and j.get("recipe_drift"):
            continue
        runs.append((tag, j))
    chosen = {}
    for tag, j in runs:
        key = setting(j)
        rank = (tag in FIGURE2_CELLS, len(j["ks"]))
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, tag, j)
    for _, tag, j in sorted(chosen.values(), key=lambda r: r[1]):
        yield tag, j


def shipped_paths(root: str = "."):
    """Repo-relative paths of every result file the figures and tables read."""
    paths = {f"{INSPEC}/{BENCHMARK}.json"}
    paths.update(f"{INSPEC}/twopath_{t}.json"
                 for t, _ in twopath_cells(root, exempt=N2000_CELLS))
    paths.update(f"{INSPEC}/{t}.json" for t in SCATTER_TAGS)
    paths.update(f"{INSPEC}/{t}.json" for t in SHIPPED_LADDERS + TABLE3_LADDERS)
    paths.update(f"{INSPEC}/res_{t}.json" for t in NSWEEP_TAGS)
    return sorted(p for p in paths if os.path.exists(os.path.join(root, p)))
