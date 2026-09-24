"""Which measurement artifacts the paper's figures and tables consume.

One definition, several consumers: make_figure_collapse builds its pooled
scatter from it, make_figure_twopath and docs/latex/build_tex_tables read
their sources from it, and devtools/build_release.sh ships exactly what it
names. A rule that appears once cannot disagree with
itself; restating these rules separately is what let them drift apart before.

Two rules govern everything here.

Final configurations only. Our models are the Gaussian-head CNP and the latent
NP, three seeds each, trained as in Appendix B; the third model is the shipped
CausalPFN checkpoint. Checkpoints trained any other way (other heads, other
sample sizes, other budgets) are not used, with one stated exception: the
n=2000 table, whose caption says its models were trained differently.

One run per setting. A setting is (model, training seed, selection rule, pool
fraction, evaluation DGP). Several settings were scored more than once; every
figure and table reads the same one. Any selection rule at pool fraction 1.0
selects the whole sample and so is the random-subset setting again.
"""

from __future__ import annotations

import glob
import json
import os

INSPEC = "logs/inspec"

# The final checkpoints of our two models, identified by their sweep directory,
# and whether each was trained on per-task standardized tasks. A cell counts
# only if it was scored the way its checkpoint was trained. Cells that predate
# the `standardize` field were scored standardized, which was then hardcoded.
TRAINED_STANDARDIZED = {"sweep_gauss1_s": False, "sweep_latfix_s": True}
FINAL_CHECKPOINTS = tuple(TRAINED_STANDARDIZED)

# Figure 2's cells. Each is also the run kept for its checkpoint's
# random-subset setting wherever that setting was scored more than once.
FIGURE2_CNP = ("g1_s0", "g1_s1", "g1_s2")
FIGURE2_LATENT = ("lat_s0", "lat_s1", "lat_s2")
FIGURE2_CELLS = FIGURE2_CNP + FIGURE2_LATENT

# The n=2000 table: (cell tag, model, which path that row reports). These
# checkpoints are not final configurations; the table's caption says how they
# were trained, and they are used nowhere else. They ship because the table is
# built from them.
N2000_SOURCES = (("n2krepro_nv", "CNP", "cov_B"),
                 ("n2krepro_a", "latent NP", "cov_A"),
                 ("n2krepro_b", "CNP", "cov_A"))
N2000_CELLS = tuple(t for t, _, _ in N2000_SOURCES)

# CausalPFN on the synthetic family, whole-sample subsets, one run per
# (n, overlap) setting. Where a setting was scored twice, the run kept is the
# one the tables use: n1000 over hi_clean, ov40_n1000 over hi_ovsoft, and
# ovsoft500 over ov40_n500 (the same tasks, scored twice). The scatter starts
# at n=250 and the sample-size table at n=125, so these are not NSWEEP_TAGS.
SCATTER_TAGS = (("res_n250", "res_n500", "res_n1000", "res_n2000",
                 "res_ovsoft500")
                + tuple(f"res_ov20_n{n}" for n in (250, 500, 1000, 2000))
                + tuple(f"res_ov40_n{n}" for n in (250, 1000, 2000))
                + tuple(f"res_ov60_n{n}" for n in (250, 500, 1000, 2000)))

# Sample sizes in the n-sweep table, in table order. Also the source of every
# whole-sample CausalPFN setting at s=1 in the tables.
NSWEEP_TAGS = ("n125", "n250", "n500", "n1000", "n2000")

# CausalPFN pool-fraction ladders at n=500. Their whole-sample rungs duplicate
# res_n500 and res_ovsoft500 and are never read; see is_whole_sample_rung.
SHIPPED_LADDERS = ("ladder_propensity_m_n500",
                   "ladder_effect_oracle_m_n500",
                   "ladder_neighbor_ball_m_n500")
TABLE3_LADDERS = ("ladder_propensity_m_n500", "ladder_nested_n500",
                  "ladder_neighbor_ball_m_n500", "ladder_propensity_m_ovsoft500",
                  "ladder_nested_ovsoft500")

# The benchmark measurements: Figure 1, Tables 1 and 4, and Table 2's IHDP row.
BENCHMARK = "benchmark_ladder"


def is_smoke(tag: str) -> bool:
    """True for probe cells, whatever they are called."""
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
    """Yield (tag, parsed) for every twopath cell a figure or table draws on.

    Keeps final-configuration checkpoints scored with the preprocessing they
    were trained on, drops smoke cells, and keeps one run per setting: Figure 2's run for the random-subset setting, and
    otherwise the run on the densest k grid (the pool-fraction ladders).
    `exempt` names cells kept regardless -- the n=2000 table's -- which are
    neither final configurations nor on the recipe, as its caption states.
    They are never passed by the pooled scatter, which compares models.
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
    """Every artifact the figures and tables read, as repo-relative paths.

    This is what the release ships: naming it here rather than in the build
    script is what stops the two from disagreeing.
    """
    paths = {f"{INSPEC}/{BENCHMARK}.json"}
    paths.update(f"{INSPEC}/twopath_{t}.json"
                 for t, _ in twopath_cells(root, exempt=N2000_CELLS))
    paths.update(f"{INSPEC}/{t}.json" for t in SCATTER_TAGS)
    paths.update(f"{INSPEC}/{t}.json" for t in SHIPPED_LADDERS + TABLE3_LADDERS)
    paths.update(f"{INSPEC}/res_{t}.json" for t in NSWEEP_TAGS)
    return sorted(p for p in paths if os.path.exists(os.path.join(root, p)))
