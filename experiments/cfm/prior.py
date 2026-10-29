"""A CausalPFN-style task prior with exact, noise-free conditional effects.

    X          random-MLP table (TabPFN-v1-style base table), n x d
    mu0(x)     random MLP of the observed X, noise-free
    tau(x)     a * (gamma * tau_raw(x) + (1 - gamma) * lambda),  a ~ U(0.1, 1)
    mu1(x)     mu0(x) + tau(x)
    e(x)       xi * sigmoid(f(x)) + (1 - xi) * 1[f(x) > 0], clipped to [e_min, 1 - e_min]
               f is constant (RCT), linear, or a random MLP of X
    T          Bernoulli(e(X))
    Y          mu_T(X) + eta(X) * eps,  Var(eta * eps) = u * Var(mu_T),  u ~ U(0.001, 1)

mu0 and mu1 are deterministic functions of X, so tau(x_i) and the subset-CATE
theta_S = mean_{i in S} tau(x_i) are exact. T depends on X only (ignorability)
and e_min > 0 gives positivity. The knobs follow CausalPFN's description of its
causal-phase prior. With `openml_prob > 0`, covariates are random rows and
columns of an OpenML table packed by `pack_tables`, standardized.

`V2` sets the rates of the further mechanisms in PriorConfig, all off by
default so that v1 configs give the same tasks.

numpy only, so it runs in either environment.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda a: np.maximum(a, 0.0),
    "sin": np.sin,
    "abs": np.abs,
    "square": lambda a: a ** 2 / 2.0,
}


@dataclass
class PriorConfig:
    n_range: tuple[int, int] = (100, 2048)
    d_range: tuple[int, int] = (1, 60)
    base_width_range: tuple[int, int] = (8, 64)     # hidden width of the covariate MLP
    base_depth_range: tuple[int, int] = (1, 4)
    base_noise_range: tuple[float, float] = (0.0, 0.3)
    mu_width: int = 32
    mu_depth_range: tuple[int, int] = (1, 3)
    propensity_kinds: tuple[str, ...] = ("rct", "linear", "mlp")
    xi_range: tuple[float, float] = (0.5, 1.0)       # 1 = smooth sigmoid; lower = sharper
    e_min: float = 0.02                              # positivity floor
    heteroskedastic_prob: float = 0.3
    activations: tuple[str, ...] = tuple(ACTIVATIONS)
    outcome_transform_prob: float = 0.0   # share of tasks with a skewed / heavy-tailed outcome surface
    openml_prob: float = 0.0          # share of tasks whose covariates come from a real table
    openml_pack: str | None = None    # path prefix written by pack_tables
    # v2 mechanisms, off by default
    discrete_prob: float = 0.0        # synthetic table: some columns binary / ordinal
    source_dist_prob: float = 0.0     # synthetic table: non-Gaussian sources
    surface_weights: tuple[float, float, float] = (1.0, 0.0, 0.0)   # mlp, trees, additive
    sparse_prob: float = 0.0          # a random function reads only 1-5 covariates
    tau_from_mu0_prob: float = 0.0    # effect modification by baseline prognosis
    alignment_prob: float = 0.0       # selection on prognosis or on gains
    imbalance_prob: float = 0.0       # treated share pushed away from 1/2
    noise_weights: tuple[float, float, float] = (1.0, 0.0, 0.0)     # gaussian, student-t, skewed
    binary_outcome_prob: float = 0.0  # Bernoulli outcomes


# v2 rates, applied on top of a v1 config.
V2 = dict(discrete_prob=0.5, source_dist_prob=0.3, surface_weights=(0.5, 0.25, 0.25), sparse_prob=0.3,
          tau_from_mu0_prob=0.25, alignment_prob=0.3, imbalance_prob=0.3, noise_weights=(0.6, 0.2, 0.2),
          binary_outcome_prob=0.1)


@dataclass
class Task:
    X: np.ndarray
    t: np.ndarray
    y: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    tau: np.ndarray
    propensity: np.ndarray
    meta: dict = field(default_factory=dict)

    def theta(self, mask: np.ndarray) -> float:
        """Exact subset-CATE over the realized units in mask."""
        return float(self.tau[mask].mean())


def _random_mlp(rng, d_in, width, depth, activations):
    """A random MLP R^d_in -> R, with the activation drawn once per network."""
    act = ACTIVATIONS[rng.choice(activations)]
    Ws, bs, dim = [], [], d_in
    for _ in range(depth):
        Ws.append(rng.normal(0, 1 / np.sqrt(dim), size=(dim, width)))
        bs.append(rng.normal(0, 0.5, size=width))
        dim = width
    w_out = rng.normal(0, 1 / np.sqrt(dim), size=dim)

    def f(X):
        h = X
        for W, b in zip(Ws, bs, strict=True):
            h = act(h @ W + b)
        return h @ w_out

    return f


def _standardize(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-8 else 1.0)


def _choose(rng, names, weights):
    """Weighted choice. Draws no random numbers when only the first weight is
    nonzero, so v1 configs reproduce their random stream."""
    if not any(weights[1:]):
        return names[0]
    w = np.asarray(weights, float)
    return str(names[int(rng.choice(len(names), p=w / w.sum()))])


def _random_trees(rng, X):
    """Sum of 1-5 random regression trees of depth 1-4: axis-aligned splits at
    random data quantiles, N(0, 1) leaves. Piecewise constant."""
    n, d = X.shape
    out = np.zeros(n)
    for _ in range(int(rng.integers(1, 6))):
        depth = int(rng.integers(1, 5))
        node = np.zeros(n, dtype=np.int64)
        for level in range(depth):
            feat = rng.integers(0, d, size=2 ** level)
            q = rng.uniform(0.1, 0.9, size=2 ** level)
            thr = np.array([np.quantile(X[:, f], qq) for f, qq in zip(feat, q, strict=True)])
            node = 2 * node + (X[np.arange(n), feat[node]] > thr[node])
        out += rng.normal(size=2 ** depth)[node]
    return out


def _random_additive(rng, X, activations):
    """A random univariate function per covariate plus 0-3 pairwise products."""
    n, d = X.shape
    out = np.zeros(n)
    for j in range(d):
        out += _random_mlp(rng, 1, 16, 1, activations)(X[:, j:j + 1])
    for _ in range(int(rng.integers(0, min(3, d * (d - 1) // 2) + 1))):
        i, j = rng.choice(d, 2, replace=False)
        out += rng.normal() * X[:, i] * X[:, j]
    return out


def _surface(rng, X, cfg, depth):
    """Standardized random function of X from a random family, possibly
    reading only a few covariates. With v1 knobs, the same draws as v1's
    random MLP."""
    d = X.shape[1]
    sparse = cfg.sparse_prob > 0 and rng.uniform() < cfg.sparse_prob
    if sparse:
        X = X[:, np.sort(rng.choice(d, int(rng.integers(1, min(d, 5) + 1)), replace=False))]
    fam = _choose(rng, ("mlp", "trees", "additive"), cfg.surface_weights)
    if fam == "mlp":
        f = _random_mlp(rng, X.shape[1], cfg.mu_width, depth, cfg.activations)(X)
    elif fam == "trees":
        f = _random_trees(rng, X)
    else:
        f = _random_additive(rng, X, cfg.activations)
    return _standardize(f), fam + ("/sparse" if sparse else "")


def _discretize(rng, X):
    """Up to half the columns become binary or ordinal (3-6 levels) at random
    quantiles; re-standardized."""
    X = X.copy()
    d = X.shape[1]
    for j in rng.choice(d, int(round(rng.uniform(0, 0.5) * d)), replace=False):
        levels = int(rng.choice([2, 2, 3, 4, 5, 6]))
        cuts = np.quantile(X[:, j], np.sort(rng.uniform(0.05, 0.95, levels - 1)))
        X[:, j] = np.searchsorted(cuts, X[:, j]).astype(np.float64)
    return np.apply_along_axis(_standardize, 0, X)


def _base_table(rng, n, d, cfg):
    """Covariates as hidden units of a random MLP driven by Gaussian sources,
    with per-layer noise; d units are sampled across the layers."""
    depth = int(rng.integers(cfg.base_depth_range[0], cfg.base_depth_range[1] + 1))
    width = int(rng.integers(cfg.base_width_range[0], cfg.base_width_range[1] + 1))
    width = max(width, d)
    noise = rng.uniform(*cfg.base_noise_range)
    act = ACTIVATIONS[rng.choice(cfg.activations)]
    if cfg.source_dist_prob > 0 and rng.uniform() < cfg.source_dist_prob:
        kind = str(rng.choice(["uniform", "t", "exponential"]))
        h = {"uniform": lambda: rng.uniform(-1.7, 1.7, size=(n, width)),
             "t": lambda: rng.standard_t(rng.uniform(2.5, 10), size=(n, width)),
             "exponential": lambda: rng.exponential(size=(n, width)) - 1.0}[kind]()
    else:
        h = rng.normal(size=(n, width))
    pool = [h]
    for _ in range(depth):
        W = rng.normal(0, 1 / np.sqrt(width), size=(width, width))
        h = act(h @ W + rng.normal(0, 0.5, size=width)) + noise * rng.normal(size=(n, width))
        pool.append(h)
    cols = np.concatenate(pool, axis=1)
    X = cols[:, rng.choice(cols.shape[1], size=d, replace=False)]
    return np.apply_along_axis(_standardize, 0, X)


def pack_tables(src_dir: str, prefix: str) -> dict:
    """Pack <src_dir>/*.npz tables into <prefix>.f32 + <prefix>.json."""
    index, offset = [], 0
    with open(prefix + ".f32", "wb") as f:
        for path in sorted(glob.glob(os.path.join(src_dir, "*.npz"))):
            z = np.load(path, allow_pickle=True)
            X = np.ascontiguousarray(z["X"], dtype=np.float32)
            f.write(X.tobytes())
            index.append(dict(did=int(z["did"]), name=str(z["name"]), offset=offset,
                              rows=X.shape[0], cols=X.shape[1]))
            offset += X.size
    json.dump(index, open(prefix + ".json", "w"))
    return dict(tables=len(index), values=offset)


_TABLES: dict = {}


def _tables(prefix: str):
    """Memory-mapped pack, opened once per process."""
    if prefix not in _TABLES:
        _TABLES[prefix] = (np.memmap(prefix + ".f32", dtype=np.float32, mode="r"),
                           json.load(open(prefix + ".json")))
    return _TABLES[prefix]


def _real_table(rng, n, d, prefix):
    """n rows and up to d columns of a random packed table, standardized.
    Tables with fewer than n rows are skipped."""
    data, index = _tables(prefix)
    ok = [e for e in index if e["rows"] >= n]
    if not ok:
        raise ValueError(f"no packed table has {n} rows")
    e = ok[int(rng.integers(len(ok)))]
    T = data[e["offset"]: e["offset"] + e["rows"] * e["cols"]].reshape(e["rows"], e["cols"])
    rows = np.sort(rng.choice(e["rows"], n, replace=False))
    cols = np.sort(rng.choice(e["cols"], min(d, e["cols"]), replace=False))
    X = np.asarray(T[rows][:, cols], dtype=np.float64)
    keep = X.std(0) > 0
    X = X[:, keep] if keep.any() else X[:, :1]
    return np.apply_along_axis(_standardize, 0, X), e["did"]


def sample_task(rng: np.random.Generator, cfg: PriorConfig | None = None,
                n: int | None = None, d: int | None = None) -> Task:
    cfg = cfg or PriorConfig()
    n = n or int(rng.integers(cfg.n_range[0], cfg.n_range[1] + 1))
    d = d or int(rng.integers(cfg.d_range[0], cfg.d_range[1] + 1))
    source = "synthetic"
    discrete = False
    if cfg.openml_prob > 0 and cfg.openml_pack and rng.uniform() < cfg.openml_prob:
        X, did = _real_table(rng, n, d, cfg.openml_pack)
        source = f"openml:{did}"
    else:
        X = _base_table(rng, n, d, cfg)
        discrete = cfg.discrete_prob > 0 and rng.uniform() < cfg.discrete_prob
        if discrete:
            X = _discretize(rng, X)
    d = X.shape[1]

    depth = lambda: int(rng.integers(cfg.mu_depth_range[0], cfg.mu_depth_range[1] + 1))
    mu0, fam_mu0 = _surface(rng, X, cfg, depth())
    tau_raw, fam_tau = _surface(rng, X, cfg, depth())
    tau_from_mu0 = cfg.tau_from_mu0_prob > 0 and rng.uniform() < cfg.tau_from_mu0_prob
    if tau_from_mu0:
        # effects that grow (or shrink) with baseline prognosis
        w = rng.uniform(0.3, 1.0)
        tau_raw = _standardize(w * rng.choice([-1.0, 1.0]) * mu0 + (1 - w) * tau_raw)
    gamma = rng.uniform(0, 1)
    lam = rng.normal()
    effect_scale = rng.uniform(0.1, 1.0)
    mu1 = mu0 + effect_scale * (gamma * tau_raw + (1 - gamma) * lam)
    transform = "none"
    if rng.uniform() < cfg.outcome_transform_prob:
        # A monotone map of both conditional means gives a skewed or
        # heavy-tailed outcome surface (Hill-B's response is exponential in X).
        transform = str(rng.choice(["exp", "softplus", "sinh"]))
        c = rng.uniform(0.5, 1.5)
        g = {"exp": np.exp, "softplus": lambda z: np.logaddexp(0, 3 * z) / 3, "sinh": np.sinh}[transform]
        # Clip the argument; on a heavy-tailed surface an unbounded exp or sinh can
        # reach 1e5 times the typical scale and overflow a float32 forward pass.
        m0, m1 = g(np.clip(c * mu0, -4, 4)), g(np.clip(c * mu1, -4, 4))
        loc, sc = m0.mean(), m0.std() if m0.std() > 1e-8 else 1.0
        mu0, mu1 = (m0 - loc) / sc, (m1 - loc) / sc
    binary = cfg.binary_outcome_prob > 0 and rng.uniform() < cfg.binary_outcome_prob
    if binary:
        # conditional means become probabilities; tau is their difference
        c, b = rng.uniform(0.5, 2.0), rng.normal()
        mu0, mu1 = 1 / (1 + np.exp(-(c * mu0 + b))), 1 / (1 + np.exp(-(c * mu1 + b)))
    tau = mu1 - mu0

    kind = rng.choice(cfg.propensity_kinds)
    fam_prop = str(kind)
    if kind == "rct":
        f = np.full(n, rng.normal(0, 0.5))
    elif kind == "linear":
        f = X @ rng.normal(0, 1 / np.sqrt(d), size=d) + rng.normal(0, 0.5)
    else:
        g_prop, fam_prop = _surface(rng, X, cfg, depth())
        f = 1.5 * g_prop
    alignment = "none"
    if kind != "rct" and cfg.alignment_prob > 0 and rng.uniform() < cfg.alignment_prob:
        # selection on prognosis (mu0) or on gains (tau), either sign
        alignment = str(rng.choice(["prognosis", "gains"]))
        a = rng.uniform(0.3, 1.0)
        target = _standardize(mu0 if alignment == "prognosis" else tau)
        f = (1 - a) * f + a * 1.5 * rng.choice([-1.0, 1.0]) * target
    offset = 0.0
    if cfg.imbalance_prob > 0 and rng.uniform() < cfg.imbalance_prob:
        offset = rng.uniform(-2.5, 2.5)
        f = f + offset
    xi = rng.uniform(*cfg.xi_range)
    e = xi / (1 + np.exp(-f)) + (1 - xi) * (f > 0)
    e = np.clip(e, cfg.e_min, 1 - cfg.e_min)
    t = (rng.uniform(size=n) < e).astype(np.float64)
    for arm in (0.0, 1.0):
        # at least two units per treatment group, moved from the other group by propensity
        short = 2 - int((t == arm).sum())
        if short > 0:
            order = np.argsort(e if arm == 1.0 else -e)[::-1]
            t[[i for i in order if t[i] != arm][:short]] = arm

    mu_t = np.where(t == 1, mu1, mu0)
    if binary:
        y = (rng.uniform(size=n) < mu_t).astype(np.float64)
        u, hetero, noise = float("nan"), False, "bernoulli"
    else:
        u = rng.uniform(1e-3, 1.0)
        hetero = rng.uniform() < cfg.heteroskedastic_prob
        eta = np.exp(0.5 * _standardize(_random_mlp(rng, d, 8, 1, cfg.activations)(X))) if hetero else np.ones(n)
        noise = _choose(rng, ("gaussian", "student-t", "skewed"), cfg.noise_weights)
        if noise == "gaussian":
            eps = rng.normal(size=n)
        elif noise == "student-t":
            eps = rng.standard_t(rng.uniform(2.5, 10), size=n)
        else:
            k = rng.uniform(0.5, 3.0)
            eps = rng.choice([-1.0, 1.0]) * (rng.gamma(k, 1.0, size=n) - k)
        eps = eps * eta
        eps *= np.sqrt(u * mu_t.var() / max(eps.var(), 1e-12))
        y = mu_t + eps

    meta = dict(n=n, d=d, source=source, discrete=bool(discrete), transform=transform, gamma=gamma, lam=lam,
                effect_scale=effect_scale, surface_mu0=fam_mu0, surface_tau=fam_tau, tau_from_mu0=bool(tau_from_mu0),
                propensity=str(kind), surface_propensity=fam_prop, alignment=alignment, offset=offset, xi=xi,
                noise_frac=u, heteroskedastic=bool(hetero), noise=noise, binary=bool(binary))
    return Task(X=X, t=t, y=y, mu0=mu0, mu1=mu1, tau=tau, propensity=e, meta=meta)
