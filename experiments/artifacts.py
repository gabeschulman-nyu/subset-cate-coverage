"""Which measurement artifacts the paper's figures and tables consume.

One definition, three consumers: make_figure_collapse builds its pooled
scatter from it, build_tex_tables reads its table sources from it, and
devtools/build_release.sh ships exactly what it names.

This module exists because those three restated the same rules separately and
drifted apart three times: a smoke cell that one filter caught and another did
not, a fingerprint keyed on a field older cells lack, and a cut arm that kept
shipping. A rule that appears once cannot disagree with itself.
"""

from __future__ import annotations

import glob
import json
import os

INSPEC = "logs/inspec"

# Cells excluded from the pooled scatter regardless of content.
#   *smoke*  -- ran a handful of tasks against the 64 a real cell uses
#   cputime  -- a timing probe, not a measurement
# route_c3 (the transformer arm) is excluded because the paper cuts it.
CUT_ARMS = ("route_c3",)

# Cells whose checkpoints depart from the frozen recipe are normally excluded,
# because a recipe difference confounds any arm comparison drawn from them.
# The n=2000 cells are the exception: all three of those checkpoints are off
# the recipe, the table that uses them says so, and they are its only source.
DRIFT_EXEMPT = ("n2krepro_nv", "n2krepro_a", "n2krepro_b")

# CausalPFN scored on the synthetic family, as the shipped-model series in the
# design-effect scatter. Deliberately NOT the same set as NSWEEP_TAGS below:
# the scatter starts at n=250 and the sample-size table starts at n=125, so
# folding them together would quietly add a point to a published statistic.
SCATTER_TAGS = (("res_n250", "res_n500", "res_n1000", "res_n2000",
                 "res_hi_clean", "res_hi_ovsoft", "res_ovsoft500")
                + tuple(f"res_ov{s}_n{n}" for s in ("20", "40", "60")
                        for n in (250, 500, 1000, 2000)))

# Per-k m_S and rho_S measured inside a selection pool, which is where the
# informative low-R points come from.
SHIPPED_LADDERS = ("ladder_propensity_m_n500",
                   "ladder_effect_oracle_m_n500",
                   "ladder_neighbor_ball_m_n500")

# Sample sizes in the n-sweep table, in table order.
NSWEEP_TAGS = ("n125", "n250", "n500", "n1000", "n2000")

# The n=2000 table: (cell tag, row label, which path that row reports).
N2000_SOURCES = (("n2krepro_nv", "CNP, unit-level", "cov_B"),
                 ("n2krepro_a", "latent NP", "cov_A"),
                 ("n2krepro_b", "CNP", "cov_A"))


def is_smoke(tag: str) -> bool:
    """True for probe cells, whatever they are called. Named-based rather
    than task-count based so a cell is excluded before it is opened."""
    return tag == "cputime" or "smoke" in tag


def twopath_cells(root: str = ".", recipe_only: bool = True, exempt=()):
    """Yield (tag, parsed) for every twopath cell a caller draws on.

    Applies, in order: the smoke exclusion, the cut-arm exclusion, and the
    recipe-drift exclusion. Callers that pool these still need to deduplicate
    on content -- two tags can name one measurement.

    `exempt` names cells that ship despite drifting from the recipe. It is
    deliberately NOT the default: the n=2000 cells belong in the release,
    because a table is built from them and its caption states the drift, but
    they must stay out of the pooled scatter, which compares arms and would
    be confounded by them. Shipping and pooling are different questions.
    """
    for f in sorted(glob.glob(os.path.join(root, INSPEC, "twopath_*.json"))):
        tag = os.path.basename(f)[len("twopath_"):-len(".json")]
        if is_smoke(tag):
            continue
        with open(f) as fh:
            j = json.load(fh)
        if j.get("arm") in CUT_ARMS:
            continue
        if recipe_only and j.get("recipe_drift") and tag not in exempt:
            continue
        yield tag, j


def shipped_paths(root: str = "."):
    """Every artifact the figures and tables read, as repo-relative paths.

    This is what the release ships: naming it here rather than in the build
    script is what stops the two from disagreeing.
    """
    paths = {f"{INSPEC}/benchmark_ladder.json"}
    paths.update(f"{INSPEC}/twopath_{t}.json"
                 for t, _ in twopath_cells(root, exempt=DRIFT_EXEMPT))
    paths.update(f"{INSPEC}/{t}.json" for t in SCATTER_TAGS)
    paths.update(f"{INSPEC}/{t}.json" for t in SHIPPED_LADDERS)
    paths.update(f"{INSPEC}/res_{t}.json" for t in NSWEEP_TAGS)
    return sorted(p for p in paths if os.path.exists(os.path.join(root, p)))
