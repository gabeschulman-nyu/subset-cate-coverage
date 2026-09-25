"""The unit-level target each benchmark is scored against.

The paper's estimand averages the conditional effect tau(X_i) = mu1 - mu0.
IHDP and ACIC 2016 store exactly that in the parts' `tau_true`. The per-unit
draws for the five simulated designs were written when sim_lib stored
y1 - y0, which adds the independent noise eps1 - eps0 of the two potential
outcomes. sim_lib now returns mu1 - mu0, but the stored parts predate that, so
for those designs the target is recomputed by rerunning the generator from the
part's seed. The rerun must reproduce the part's covariates, treatment and
potential outcomes, or no target is returned.

check_target is the guard: a target equal to y1 - y0 carries outcome noise.
"""

from __future__ import annotations

import os
import sys

import numpy as np

SIMULATED = ("HillA", "HillB", "HillBHeavy", "FriedmanConf", "KangSchafer")

# Where sim_lib.py lives: benchmarks/ in the release, or set SIM_LIB_DIR.
SIM_LIB_DIR = os.environ.get(
    "SIM_LIB_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "benchmarks"))


def _generators():
    if SIM_LIB_DIR not in sys.path:
        sys.path.insert(0, SIM_LIB_DIR)
    from sim_lib import DGPS
    return DGPS


def conditional_effect(dgp: str, part) -> np.ndarray:
    """tau(X_i) for every unit of one stored simulated replication."""
    seed = int(part["seed"])
    sim = _generators()[dgp](seed=seed, n=part["X"].shape[0])
    # The parts store float32; Kang--Schafer's fourth covariate is ~400.
    same = (np.allclose(sim.X, part["X"], rtol=1e-5, atol=1e-5)
            and np.array_equal(sim.z, part["t"])
            and np.allclose(sim.y0, part["y0_true"], rtol=1e-5, atol=1e-4)
            and np.allclose(sim.y1, part["y1_true"], rtol=1e-5, atol=1e-4))
    if not same:
        raise RuntimeError(f"{dgp} seed {seed}: the generator no longer "
                           "reproduces the stored replication")
    return sim.tau


def target(dgp: str, part) -> np.ndarray:
    tau = conditional_effect(dgp, part) if dgp in SIMULATED else part["tau_true"]
    check_target(dgp, part, tau)
    return tau


def check_target(dgp: str, part, tau) -> None:
    """Fail if the target is the individual effect y1 - y0.

    Equality is judged at the float32 precision of the stored outcomes, which
    for Kang--Schafer (outcomes near 210) is about 1e-5 in y1 - y0. Outcome
    noise differs by eps1 - eps0, of order one, in every unit.
    """
    y0, y1 = part["y0_true"], part["y1_true"]
    ite = y1 - y0
    scale = 1.0 + max(float(np.abs(y0).max()), float(np.abs(y1).max()))
    if np.allclose(tau, ite, rtol=0.0, atol=1e-5 * scale):
        raise RuntimeError(f"{dgp} seed {int(part['seed'])}: the target equals "
                           "y1 - y0 and so contains outcome noise")
