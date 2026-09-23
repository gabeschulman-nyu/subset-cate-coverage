"""Real-benchmark loaders for the CausalPFN validation phase.

Returns SimData (from sim_lib) so the existing run/analyze pipeline works
unchanged. Two benchmarks, mirroring the CausalPFN repo's `benchmarks/`
loaders (IHDP via Shalit's `ihdp_npci_1-100.npz`; ACIC 2016 via the official
aciccomp2016 generator, all 77 settings — see `generate_acic.R`).

Both loaders index by `seed` so the existing
    sim = DGPS[name](seed=seed)
contract is preserved:
    IHDP:  split_idx = (seed - 1) % 100,  n_splits = 100
    ACIC:  setting_idx = (seed - 1) % 77, n_settings = 77

Each call returns ONE table's worth of (X, t, y, y0, y1, mu0, mu1, p_treat).
For real benchmarks the true propensity is unknown, so `p_treat` is estimated
with a logistic regression on the table itself (used only for downstream
propensity-quintile cuts; not used by CausalPFN).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sim_lib import SimData, DGPS

_DATA_DIR = Path(__file__).resolve().parent / "data"
_IHDP_DIR = _DATA_DIR / "IHDP"
_ACIC_DIR = _DATA_DIR / "ACIC2016"

N_IHDP_SPLITS = 100
N_ACIC_SETTINGS = 77


def _estimate_propensity(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Logistic-regression propensity estimate (for propensity-quintile cuts)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xs, t.astype(int))
    return lr.predict_proba(Xs)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# IHDP (Hill 2011 / Shalit-Johansson NPCI splits)
# ---------------------------------------------------------------------------

_IHDP_CACHE = {"train": None, "test": None}


def _load_ihdp_npz():
    if _IHDP_CACHE["train"] is None:
        _IHDP_CACHE["train"] = np.load(_IHDP_DIR / "ihdp_npci_1-100.train.npz")
        _IHDP_CACHE["test"] = np.load(_IHDP_DIR / "ihdp_npci_1-100.test.npz")
    return _IHDP_CACHE["train"], _IHDP_CACHE["test"]


def ihdp(seed: int = 1) -> SimData:
    """Load one IHDP NPCI split.

    Combines train+test into a single table (matches CausalPFN's ATE_Dataset
    construction), then deterministically permutes by `seed`.
    Counterfactual y0/y1 are recovered from (yf, ycf, t); mu0/mu1 are the
    true conditional means used for the ground-truth ATE.
    """
    tr, te = _load_ihdp_npz()
    idx = (seed - 1) % N_IHDP_SPLITS

    X_tr = tr["x"][..., idx].astype(np.float32)         # (672, 25)
    X_te = te["x"][..., idx].astype(np.float32)         # (75,  25)
    t_tr = tr["t"][..., idx].astype(np.float32)
    t_te = te["t"][..., idx].astype(np.float32)
    yf_tr = tr["yf"][..., idx].astype(np.float32)
    yf_te = te["yf"][..., idx].astype(np.float32)
    ycf_tr = tr["ycf"][..., idx].astype(np.float32)
    ycf_te = te["ycf"][..., idx].astype(np.float32)
    mu0_tr = tr["mu0"][..., idx].astype(np.float32)
    mu0_te = te["mu0"][..., idx].astype(np.float32)
    mu1_tr = tr["mu1"][..., idx].astype(np.float32)
    mu1_te = te["mu1"][..., idx].astype(np.float32)

    X = np.concatenate([X_tr, X_te], axis=0)
    t = np.concatenate([t_tr, t_te], axis=0)
    yf = np.concatenate([yf_tr, yf_te], axis=0)
    ycf = np.concatenate([ycf_tr, ycf_te], axis=0)
    mu0 = np.concatenate([mu0_tr, mu0_te], axis=0)
    mu1 = np.concatenate([mu1_tr, mu1_te], axis=0)

    y0 = np.where(t == 0, yf, ycf).astype(np.float32)
    y1 = np.where(t == 1, yf, ycf).astype(np.float32)
    y = yf.astype(np.float32)
    tau = (mu1 - mu0).astype(np.float32)
    true_ate = float(np.mean(mu1 - mu0))
    true_att = float(np.mean((mu1 - mu0)[t == 1])) if (t == 1).any() else float("nan")

    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(X.shape[0])
    X, t, y, y0, y1, tau = X[perm], t[perm], y[perm], y0[perm], y1[perm], tau[perm]

    p_treat = _estimate_propensity(X, t)
    return SimData(
        X=X.astype(np.float32), z=t.astype(np.float32), y=y,
        tau=tau, y0=y0, y1=y1, p_treat=p_treat,
        true_ate=true_ate, true_att=true_att, dgp="IHDP",
    )


# ---------------------------------------------------------------------------
# ACIC 2016 (Dorie et al. 2019) — full 77-setting benchmark.
# zymu_{1..77}.csv generated via aciccomp2016::dgp_2016 on data/ACIC2016/x.csv
# (byte-identical to aciccomp2016::input_2016), one rep per setting.
# See generate_acic.R. Original causallib 10-setting subset is preserved in
# data/ACIC2016/orig_causallib_backup/.
# ---------------------------------------------------------------------------

_ACIC_X_CACHE = None


def _load_acic_x() -> np.ndarray:
    global _ACIC_X_CACHE
    if _ACIC_X_CACHE is not None:
        return _ACIC_X_CACHE
    x_df = pd.read_csv(_ACIC_DIR / "x.csv")
    x_df.columns = [f"x_{i+1}" for i in range(x_df.shape[1])]
    cat_cols = ["x_2", "x_21", "x_24"]
    num_cols = [c for c in x_df.columns if c not in cat_cols]
    for c in cat_cols:
        x_df[f"{c}_numeric"] = x_df[c].astype("category").cat.codes
    num_cols = num_cols + [f"{c}_numeric" for c in cat_cols]
    _ACIC_X_CACHE = x_df.loc[:, num_cols].values.astype(np.float32)
    return _ACIC_X_CACHE


def acic2016(seed: int = 1) -> SimData:
    """Load one ACIC 2016 setting (i = ((seed-1) % 77) + 1)."""
    idx = (seed - 1) % N_ACIC_SETTINGS
    X = _load_acic_x()
    sim = pd.read_csv(_ACIC_DIR / f"zymu_{idx + 1}.csv")
    sim.columns = ["z", "y0", "y1", "mu0", "mu1"]
    t = sim["z"].values.astype(np.float32)
    y0 = sim["y0"].values.astype(np.float32)
    y1 = sim["y1"].values.astype(np.float32)
    mu0 = sim["mu0"].values.astype(np.float32)
    mu1 = sim["mu1"].values.astype(np.float32)
    y = np.where(t == 1, y1, y0).astype(np.float32)
    tau = (mu1 - mu0).astype(np.float32)
    true_ate = float(np.mean(mu1 - mu0))
    true_att = float(np.mean((mu1 - mu0)[t == 1])) if (t == 1).any() else float("nan")
    p_treat = _estimate_propensity(X, t)
    return SimData(
        X=X.astype(np.float32), z=t, y=y,
        tau=tau, y0=y0, y1=y1, p_treat=p_treat,
        true_ate=true_ate, true_att=true_att, dgp="ACIC2016",
    )


DGPS.update({"IHDP": ihdp, "ACIC2016": acic2016})
