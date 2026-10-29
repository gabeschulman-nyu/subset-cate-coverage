"""Two-path ablation: one checkpoint, two ways of querying a subset-CATE.

Weights, context, tasks, target and subset membership are fixed. Path A queries
the subset once (SCM-marginalized aggregation). Path B queries each unit in S as
its own subset (|S| = 1) and aggregates the k unit-level predictives with
separate draws per unit (unit-level aggregation, as in CausalPFN). The CNP was
trained with gate_k_min = 1, so |S| = 1 is in its training range.
Writes twopath_<tag>.json, read by Figures 2 and 4 and Tables 2 and 5.
"""

from __future__ import annotations

import argparse
import os
import json

import numpy as np
import torch

from scipy.stats import norm

from subsetcate.design_effect import from_per_unit_errors

_Phi = norm.cdf
from subsetcate.exact import make_exact_ite_query_fn, make_polynomial_prior_fn
from subsetcate.functional_sampling import route_a_predictive_probs
from subsetcate.histogram import histogram_quantiles
from subsetcate.mdn import mdn_sample
from subsetcate.mixture import (
    CausalMixtureDistribution,
    PicklableLambdaWrapper,
    _worker_generate_task,
)
from subsetcate.modules import CausalMetaModel, build_backbone
from subsetcate.recipe import assert_matches

EVAL_SEED = 20260812
D = 3
FWD_CHUNK = 32
Q = torch.tensor([0.025, 0.975], dtype=torch.float64)


def _mixture_fn(scm):
    mix = CausalMixtureDistribution(causal_model=scm)
    mix.add_observational(weight=1.0)
    return mix


def generate_tasks(num_tasks, n, prior_fn, standardize=True):
    """Exact per-unit ITEs for every context row. theta_S is their mean over S,
    the target for both paths."""
    obs = ["y", "t"] + [f"x_{i + 1}" for i in range(D)]
    tasks = []
    for i in range(num_tasks):
        tasks.append(
            _worker_generate_task(
                PicklableLambdaWrapper(prior_fn),
                PicklableLambdaWrapper(_mixture_fn),
                PicklableLambdaWrapper(make_exact_ite_query_fn()),
                num_context_points=n,
                num_query_points=1,
                outcome_name="y",
                treatment_name="t",
                observed_feature_names=obs,
                query_feature_names=None,
                query_x_type="context-ite",
                standardize=standardize,
                return_propensity=True,
                worker_seed=EVAL_SEED + 7919 * i,
            )
        )
    return tasks


@torch.no_grad()
def head_out_for_masks(bb, x_feat, x_src, masks, device):
    """Raw head output for each subset mask, same context throughout."""
    out = []
    for s in range(0, masks.shape[0], FWD_CHUNK):
        mk = masks[s : s + FWD_CHUNK].to(device)
        b = mk.shape[0]
        out.append(bb(
            x_feat.unsqueeze(0).expand(b, -1, -1).to(device),
            x_src.unsqueeze(0).expand(b, -1).to(device),
            subset_mask=mk,
        ).cpu())
    return torch.cat(out, dim=0)


def sample_rows(out_rows, mask_rows, kind, edges, ndraw, g, min_sigma):
    """Draws [ndraw, R] for R head outputs, independently per row."""
    if kind == "mixture":
        return mdn_sample(out_rows, ndraw, min_sigma)
    probs = torch.softmax(out_rows, dim=-1)
    u = torch.rand(ndraw, probs.size(0), 1, generator=g)
    return histogram_quantiles(
        probs.unsqueeze(0).expand(ndraw, -1, -1), edges, u).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="logs/inspec",
                    help="where twopath_<tag>.json is written")
    ap.add_argument("--num_tasks", type=int, default=64)
    ap.add_argument("--num_draws", type=int, default=500)
    ap.add_argument("--n_sub", type=int, default=24)
    ap.add_argument("--allow_recipe_drift", action="store_true",
                    help="score a checkpoint that departs from the frozen "
                         "recipe. Preliminary reads only: any arm or head "
                         "difference it produces is confounded.")
    # Override the evaluation DGP's propensity floor and sharpness s; the
    # checkpoint is unchanged.
    ap.add_argument("--eval_overlap_floor", type=float, default=None)
    ap.add_argument("--eval_overlap_sharpness", type=float, default=None)
    ap.add_argument("--ks", type=str, default="",
                    help="explicit comma-separated subset sizes, e.g. "
                         "5,50,500,2000. Overrides --k_points/--include_k1.")
    ap.add_argument("--k_points", type=int, default=9,
                    help="number of log-spaced subset sizes between 2 and n")
    ap.add_argument("--include_k1", action="store_true",
                    help="prepend k=1, where the two paths coincide by construction")
    ap.add_argument("--pool_frac", type=float, default=0.25,
                    help="fraction of the sample forming the selection pool; "
                         "smaller means a tighter subgroup and higher rho_S")
    ap.add_argument("--selection", default="random",
                    choices=["random", "x1_quartile", "sum_quartile",
                             "covariate_cell", "quad_quartile",
                             "propensity_stratum", "effect_top",
                             "effect_oracle", "neighbor_ball"])
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # route_b is the CNP, which decodes theta_S from a subset mask. route_a is
    # the latent NP, which decodes units under a shared latent z; its paths are
    # the joint and the z-marginal of the same decodes. vanilla decodes units
    # only (the unit-level CNP of Table 5), so it has no Path A.
    assert cfg["model_variant"] in ("route_b", "route_a",
                                    "vanilla"), cfg["model_variant"]
    is_latent = cfg["model_variant"] == "route_a"
    is_vanilla = cfg["model_variant"] == "vanilla"
    per_unit = is_latent or is_vanilla
    assert cfg["gate_k_min"] == 1, f"|S|=1 outside training range: {cfg['gate_k_min']}"
    try:
        assert_matches(cfg)
    except ValueError:
        if not a.allow_recipe_drift:
            raise
        print("WARNING: scoring a checkpoint that departs from the recipe; "
              "results are confounded and marked recipe_drift=true")
        recipe_drift = True
    else:
        recipe_drift = False
    n = int(cfg["dataset_size"])

    backbone = build_backbone(
        config=cfg, backbone_type=cfg["backbone_type"], predict_ate=not per_unit,
        dim_context_features=2 + D,
        dim_query_features=(2 + D) if per_unit else 1,
        num_source_types=1,
    )
    model = CausalMetaModel.load_from_checkpoint(
        a.checkpoint, backbone=backbone, custom_baselines=None, map_location=device
    ).to(device)
    model.eval()
    bb = model.backbone
    kind = cfg.get("estimator_type", "histogram")
    edges = model.hist_edges.cpu() if kind == "histogram" else None
    torch.manual_seed(EVAL_SEED)   # mdn_sample draws from the global stream
    min_sigma = float(getattr(model, "mdn_min_sigma", 1e-3))

    prior_fn = make_polynomial_prior_fn(
        D, cfg.get("outcome_noise_scale", 0.25),
        effect_scale=cfg.get("effect_scale", 0.35),
        unobs_noise_coef=cfg.get("unobs_noise_coef", 0.0),
        unobs_conf_coef=cfg.get("unobs_conf_coef", 0.0),
        overlap_floor=(a.eval_overlap_floor if a.eval_overlap_floor is not None
                       else cfg.get("overlap_floor", 0.05)),
        overlap_sharpness=(a.eval_overlap_sharpness
                           if a.eval_overlap_sharpness is not None
                           else cfg.get("overlap_sharpness", 1.0)),
    )
    eval_ood = (a.eval_overlap_floor is not None
                or a.eval_overlap_sharpness is not None)
    # Tasks are preprocessed as the checkpoint was trained (raw or
    # standardized per task).
    standardize = bool(cfg.get("standardize", True))
    tasks = generate_tasks(a.num_tasks, n, prior_fn, standardize=standardize)
    # At k=1 there is nothing to aggregate, so the two paths agree.
    if a.ks:
        ks = sorted({min(int(v), n) for v in a.ks.split(",")})
    else:
        ks = [int(k) for k in np.unique(np.geomspace(2, n, a.k_points).astype(int))]
        if a.include_k1:
            ks = [1] + ks
    rng = np.random.default_rng(EVAL_SEED)
    g = torch.Generator().manual_seed(EVAL_SEED)

    acc = {p: {k: {"hit": [], "w": []} for k in ks} for p in ("A", "B")}
    # (centre, half-width, truth) of each Path B interval, for the post-hoc
    # rescalings once rho_S and m_S are known.
    ivals = {k: [] for k in ks}
    unit_err, unit_sd, pool_cbar, pool_vbar, pool_sdbar = [], [], [], [], []

    for ti, t in enumerate(tasks):
        x_feat, x_src = t["x_features"], t["x_sources"]
        ite = t["query_value"].reshape(-1).double()          # per-unit ITE

        if is_latent:
            # [K, n, nbins]: unit predictives, one shared z per draw.
            with torch.no_grad():
                pd = route_a_predictive_probs(
                    bb, {"x_features": x_feat.unsqueeze(0).to(device),
                         "x_sources": x_src.unsqueeze(0).to(device)},
                    a.num_draws)[:, 0].cpu()
            # z-marginal per-unit predictive, used by unit-level aggregation.
            marg = pd.mean(dim=0)

        # Path B: each unit queried as its own subset.
        eye = torch.zeros(n, n, dtype=torch.bool)
        eye[torch.arange(n), torch.arange(n)] = True
        assert eye.sum(dim=1).eq(1).all(), "Path B masks must be |S|=1"
        if is_latent:
            o_unit = None
            u1 = torch.rand(a.num_draws, n, 1, generator=g)
            du = histogram_quantiles(
                marg.unsqueeze(0).expand(a.num_draws, -1, -1), edges, u1).squeeze(-1)
            # Within-draw unit samples under the shared z, for Path A.
            u_joint = torch.rand(a.num_draws, n, 1, generator=g)
            tau_joint = histogram_quantiles(pd, edges, u_joint).squeeze(-1)  # [K, n]
        elif is_vanilla:
            # Query rows are the context rows ([y, t, x_1..x_d]), so one
            # forward gives the n unit-level head outputs.
            with torch.no_grad():
                o_unit = bb(x_feat.unsqueeze(0).to(device),
                            x_src.unsqueeze(0).to(device),
                            query_x=x_feat.unsqueeze(0).to(device)
                            ).squeeze(0).cpu()
            du = sample_rows(o_unit, eye, kind, edges, a.num_draws, g, min_sigma)
        else:
            o_unit = head_out_for_masks(bb, x_feat, x_src, eye, device)
            du = sample_rows(o_unit, eye, kind, edges, a.num_draws, g, min_sigma)
        tau_hat = du.double().mean(dim=0).numpy()   # predicted CATE per unit
        unit_err.append(tau_hat - ite.numpy())
        # Per-unit predictive sd, for m = mean(sigma_hat^2)/sigma^2.
        unit_sd.append(du.double().std(dim=0).numpy())

        # Subsets are drawn once and shared by both paths.
        # Columns are [y, t, x_1..x_d], so covariates start at index 2.
        X = x_feat[:, 2 : 2 + D].numpy()
        q = max(2, int(round(n * a.pool_frac)))
        if a.selection == "x1_quartile":
            pool_all = np.argsort(-X[:, 0])[:q]
        elif a.selection == "sum_quartile":
            pool_all = np.argsort(-X.sum(axis=1))[:q]
        elif a.selection == "covariate_cell":
            m2 = (X[:, 0] > np.median(X[:, 0])) & (X[:, 1] > np.median(X[:, 1]))
            pool_all = np.where(m2)[0]
        elif a.selection == "propensity_stratum":
            # Bottom pool_frac of the true propensity. A smaller pool_frac
            # gives a higher rho_S with the DGP and model unchanged.
            pool_all = np.argsort(t["propensity"].numpy())[:q]
        elif a.selection == "effect_top":
            # Top pool_frac by the model's predicted effect; feasible at
            # test time.
            pool_all = np.argsort(-tau_hat)[:q]
        elif a.selection == "neighbor_ball":
            # The q units nearest a random anchor unit in covariate space.
            anchor = rng.integers(n)
            d2 = ((X - X[anchor]) ** 2).sum(axis=1)
            pool_all = np.argsort(d2)[:q]
        elif a.selection == "effect_oracle":
            # Top pool_frac by the true effect. An oracle rule, as is
            # propensity_stratum, which ranks on the true propensity.
            pool_all = np.argsort(-ite.numpy())[:q]
        elif a.selection == "quad_quartile":
            # Aligned with the quadratic block of phi(X), where the effect
            # basis puts most of its mass.
            pool_all = np.argsort(-(X**2).sum(axis=1))[:q]
        else:
            pool_all = np.arange(n)
        # rho_S: error correlation within the pool. Equals rho when the pool
        # is the whole sample.
        ep = unit_err[-1][pool_all]
        np_ = len(ep)
        pool_cbar.append((ep.sum() ** 2 - (ep ** 2).sum()) / (np_ * (np_ - 1)))
        pool_vbar.append(float(np.mean(ep ** 2)))
        # m_S is defined over S, so it is also restricted to the pool.
        pool_sdbar.append(float(np.mean(unit_sd[-1][pool_all] ** 2)))

        subsets = {k: [rng.choice(pool_all, size=k, replace=False)
                       for _ in range(a.n_sub)]
                   for k in ks if k <= len(pool_all)}

        flat = [(k, j) for k in subsets for j in range(len(subsets[k]))]
        masks = torch.zeros(len(flat), n, dtype=torch.bool)
        for r, (k, j) in enumerate(flat):
            masks[r, torch.as_tensor(subsets[k][j])] = True
        o_sub = (None if per_unit
                 else head_out_for_masks(bb, x_feat, x_src, masks, device))

        for r, (k, j) in enumerate(flat):
            idx = torch.as_tensor(subsets[k][j])
            truth = float(ite[idx].mean())

            # Path A. CNP: one decode for the whole subset. Latent NP: the
            # subset mean within each z draw.
            if is_vanilla:
                # No subset decode, so Path A is undefined and reported as NaN.
                pass
            else:
                if is_latent:
                    dA = tau_joint[:, idx].double().mean(dim=1)
                else:
                    dA = sample_rows(o_sub[r : r + 1], masks[r : r + 1], kind,
                                     edges, a.num_draws, g, min_sigma).squeeze(-1)
                lo, hi = torch.quantile(dA.double(), Q)
                acc["A"][k]["hit"].append(bool(lo <= truth <= hi))
                acc["A"][k]["w"].append(float(hi - lo))

            # Path B: k unit-level predictives, drawn independently and
            # averaged. For the latent NP these are the Path A decodes with z
            # marginalised out.
            if is_latent:
                uB = torch.rand(a.num_draws, len(idx), 1, generator=g)
                dB = histogram_quantiles(
                    marg[idx].unsqueeze(0).expand(a.num_draws, -1, -1), edges, uB
                ).squeeze(-1).double().mean(dim=1)
            else:
                dB = sample_rows(o_unit[idx], eye[idx], kind, edges, a.num_draws,
                                 g, min_sigma).double().mean(dim=1)
            lo, hi = torch.quantile(dB, Q)
            acc["B"][k]["hit"].append(bool(lo <= truth <= hi))
            acc["B"][k]["w"].append(float(hi - lo))
            ivals[k].append((float(0.5 * (lo + hi)), float(0.5 * (hi - lo)),
                             truth))

        if ti % 8 == 0:
            print(f"  task {ti}/{a.num_tasks}", flush=True)

    E, SD = np.stack(unit_err), np.stack(unit_sd)
    sigma2, rho = from_per_unit_errors(E)
    m_global = float(np.mean(SD ** 2) / sigma2)
    rho_S = float(np.mean(pool_cbar) / np.mean(pool_vbar))
    sigma2_S = float(np.mean(pool_vbar))
    m = float(np.mean(pool_sdbar) / sigma2_S)
    # Predicted k*, with m_S and rho_S both restricted to the pool.
    kstar_pred = 1.0 + (m - 1.0) / rho_S if rho_S > 0 else float("inf")
    out = {"tag": a.tag, "checkpoint": a.checkpoint, "n": n, "ks": ks,
           "seed_cfg": cfg.get("seed"), "selection": a.selection, "pool_frac": a.pool_frac, "head": kind,
           "arm": cfg["model_variant"], "recipe_drift": recipe_drift,
           "eval_ood": eval_ood,
           "num_tasks": a.num_tasks, "n_sub": a.n_sub,
           "sigma2": sigma2, "sigma2_S": sigma2_S, "rho": rho, "rho_S": rho_S,
           "m": m, "m_global": m_global, "kstar_pred": kstar_pred}
    # Post-hoc rescalings of the Path B half-widths, Deff = 1 + (k-1) rho_S:
    #   cov_B_corrected  design effect only               factor sqrt(Deff)
    #   cov_B_recal      per-unit recalibration only       factor 1/sqrt(m_S)
    #   cov_B_full       recalibration + design effect     factor sqrt(Deff/m_S)
    # The design-effect rescaling turns R = m_S/Deff into m_S, so its
    # predicted coverage is 2*Phi(z*sqrt(m_S)) - 1 at every k.
    def _cov(fac_of_k):
        out_ = []
        for k in ks:
            v = ivals[k]
            if not v:
                out_.append(float("nan"))
                continue
            arr = np.array(v)
            out_.append(float(np.mean(np.abs(arr[:, 2] - arr[:, 0])
                                      <= arr[:, 1] * fac_of_k(k))))
        return out_

    def _deff(k):
        return max(1.0 + (k - 1) * rho_S, 1e-12)

    rm = max(m, 1e-12)
    cov_corr = _cov(lambda k: np.sqrt(_deff(k)))
    out["cov_B_corrected"] = cov_corr
    out["cov_B_recal"] = _cov(lambda k: 1.0 / np.sqrt(rm))
    out["cov_B_full"] = _cov(lambda k: np.sqrt(_deff(k) / rm))
    out["cov_B_corrected_pred"] = float(
        2 * _Phi(1.96 * np.sqrt(max(m, 0.0))) - 1)
    for p in ("A", "B"):
        out[f"cov_{p}"] = [float(np.mean(acc[p][k]["hit"])) if acc[p][k]["hit"]
                           else float("nan") for k in ks]
        out[f"width_{p}"] = [float(np.mean(acc[p][k]["w"])) if acc[p][k]["w"]
                             else float("nan") for k in ks]

    print(f"\n{a.tag}: n={n} seed={cfg.get('seed')} rho={rho:.4f} m={m:.3f} "
          f"rho_S={rho_S:.4f} sigma2={sigma2:.4f} k*_pred={kstar_pred:.1f}")
    print("  k      " + "".join(f"{k:>9d}" for k in ks))
    for p, lab in (("A", "direct"), ("B", "decomp")):
        print(f"  cov {lab} " + "".join(f"{c:9.3f}" for c in out[f"cov_{p}"]))
        print(f"  wid {lab} " + "".join(f"{w:9.4f}" for w in out[f"width_{p}"]))
    out["standardize"] = standardize
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"twopath_{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
