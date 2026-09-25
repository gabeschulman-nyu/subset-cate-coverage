"""Two-path aggregation ablation: one checkpoint, two query decompositions.

The main paper's claim is about the aggregation step, but every measurement so
far compares different models, so a reader can accept the numbers and still
conclude that CausalPFN in particular has a bug. This holds weights, context,
task set, target and subset membership fixed and varies only how the query is
decomposed.

  Path A (direct)      -- query theta_S once with the subset mask.
  Path B (decomposed)  -- query each unit i in S as its own subset (|S| = 1)
                          through the same forward call, then aggregate the k
                          predictive distributions independently, drawing
                          separately per unit. This is CausalPFN's rule.

Both subset-decode arms (route_b / CNP, route_c3 / TNP) were trained with
gate_k_min = 1 and log-uniform k, so |S| = 1 is inside the training
distribution and Path B needs no retraining and no second head.

Pre-registered predictions (recorded before running):
  1. Path B coverage falls with k; Path A's does not.
  2. Path B's reported width contracts as k^{-1/2}; Path A's does not.
  3. Path B's crossing k* tracks rho measured on this checkpoint's per-unit
     errors, the same way the CausalPFN rows do.
If Path A also collapses, the story is wrong and that supersedes everything.
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
from subsetcate.lotv import lotv_split
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
    """Per-unit exact ITEs for every context row, in the model's standardized
    units. theta_S for any subset is then the mean over that subset, so Path A
    and Path B are scored against the identical target."""
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
    """One forward per mask, identical context every time. Raw head output, so
    the same routine serves the histogram and LOTV heads."""
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
    """Draws [ndraw, R] for R head outputs. LOTV reads |S| off the mask
    arithmetically, so a |S|=1 row returns the single-unit variance v. The
    mixture head draws each row from its own component, so calling it on the
    k rows of a subset keeps the units independent -- which is what Path B
    is meant to assume."""
    if kind == "mixture":
        return mdn_sample(out_rows, ndraw, min_sigma)
    if kind == "lotv":
        mu, sigma, _, _ = lotv_split(out_rows, mask_rows, min_sigma)
        z = torch.randn(ndraw, out_rows.shape[0], generator=g)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * z
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
    # A rho_S ladder at FIXED pool size (n/4), so pool size is not confounded
    # with alignment. Only the projection defining the quartile changes.
    ap.add_argument("--allow_recipe_drift", action="store_true",
                    help="score a checkpoint that departs from the frozen "
                         "recipe. Preliminary reads only: any arm or head "
                         "difference it produces is confounded.")
    # The shipped model is evaluated on tasks it never trained on, so scoring
    # a trained arm on the same tasks is the matched comparison. Overriding the
    # eval prior keeps the checkpoint fixed and changes only the task stream.
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
    # route_b (CNP) and route_c3 (TNP) both decode the aggregate directly from
    # a subset mask, so both support the two-path split unchanged.
    # route_b/route_c3 decode the aggregate from a subset mask. route_a is
    # per-unit with a latent, so its two paths are the marginal and the joint
    # over the same decodes -- the sharper form of the same contrast.
    # vanilla is per-unit only: it has no subset decode, so it has no Path A.
    # It is scored here so the unit-level arm of the n=2000 table comes from
    # the same instrument as every other cell rather than from a stored CSV.
    assert cfg["model_variant"] in ("route_b", "route_c3", "route_a",
                                    "vanilla"), cfg["model_variant"]
    is_latent = cfg["model_variant"] == "route_a"
    is_vanilla = cfg["model_variant"] == "vanilla"
    per_unit = is_latent or is_vanilla
    assert cfg["gate_k_min"] == 1, f"|S|=1 outside training range: {cfg['gate_k_min']}"
    # Refuse to score a checkpoint that was not trained under the frozen
    # recipe: a silent deviation reads as a head or arm effect.
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
    # Tasks are preprocessed the way this checkpoint was trained. Scoring a
    # model trained on raw tasks against standardized ones shows it inputs it
    # never saw and inflates its reported widths; that is what happened to
    # the Gaussian-head CNP before this was read from the checkpoint.
    standardize = bool(cfg.get("standardize", True))
    tasks = generate_tasks(a.num_tasks, n, prior_fn, standardize=standardize)
    # Default grid is unchanged so existing cells stay comparable. The
    # denser options exist because k* lands in the single digits when the
    # per-unit cushion m is small, and a 9-point grid cannot resolve a
    # crossing between k=2 and k=7. k=1 is the common origin: with one unit
    # there is nothing to aggregate, so both paths must agree there.
    if a.ks:
        ks = sorted({min(int(v), n) for v in a.ks.split(",")})
    else:
        ks = [int(k) for k in np.unique(np.geomspace(2, n, a.k_points).astype(int))]
        if a.include_k1:
            ks = [1] + ks
    rng = np.random.default_rng(EVAL_SEED)
    g = torch.Generator().manual_seed(EVAL_SEED)

    acc = {p: {k: {"hit": [], "w": []} for k in ks} for p in ("A", "B")}
    # Centre/half-width/truth for the composed path, kept so the post-hoc
    # design-effect correction can be applied after rho_S is known.
    ivals = {k: [] for k in ks}
    unit_err, unit_sd, pool_cbar, pool_vbar, pool_sdbar = [], [], [], [], []

    for ti, t in enumerate(tasks):
        x_feat, x_src = t["x_features"], t["x_sources"]
        ite = t["query_value"].reshape(-1).double()          # per-unit, standardized

        if is_latent:
            # [K, n, nbins]: unit predictives under a SHARED z per draw.
            with torch.no_grad():
                pd = route_a_predictive_probs(
                    bb, {"x_features": x_feat.unsqueeze(0).to(device),
                         "x_sources": x_src.unsqueeze(0).to(device)},
                    a.num_draws)[:, 0].cpu()
            # Marginalising z gives the per-unit predictive an independent
            # aggregator would use; the joint is what keeps the shared term.
            marg = pd.mean(dim=0)

        # ---- Path B basis: every unit queried on its own, same forward call.
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
            # query_x_type is "sample-from-context" for this arm, so a query
            # row IS a context row: the same [y, t, x_1..x_d] columns. Asking
            # for every row at once gives [n, nbins] in one forward, which is
            # the same object head_out_for_masks builds for the mask arms.
            with torch.no_grad():
                o_unit = bb(x_feat.unsqueeze(0).to(device),
                            x_src.unsqueeze(0).to(device),
                            query_x=x_feat.unsqueeze(0).to(device)
                            ).squeeze(0).cpu()
            du = sample_rows(o_unit, eye, kind, edges, a.num_draws, g, min_sigma)
        else:
            o_unit = head_out_for_masks(bb, x_feat, x_src, eye, device)
            du = sample_rows(o_unit, eye, kind, edges, a.num_draws, g, min_sigma)
        tau_hat = du.double().mean(dim=0).numpy()   # the model's own CATE
        unit_err.append(tau_hat - ite.numpy())
        # Per-unit predictive sd, so m = mean(sigma_hat^2)/sigma^2 is measured
        # rather than backed out of the k=2 width.
        unit_sd.append(du.double().std(dim=0).numpy())

        # ---- Subsets, drawn once and used by BOTH paths.
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
            # Bottom pool_frac of the TRUE propensity. Tightening the stratum
            # concentrates the shared error component, so pool_frac is a
            # continuous knob on rho_S that leaves the DGP, the model and the
            # estimator untouched -- the practitioner just asks about a
            # narrower subgroup.
            pool_all = np.argsort(t["propensity"].numpy())[:q]
        elif a.selection == "effect_top":
            # Top pool_frac by the model's OWN predicted effect. This is
            # feasible at test time -- no oracle -- and is the subgroup
            # analysis practitioners actually run. It targets rho_S directly:
            # the shared error component is concentrated by effect
            # heterogeneity, which a propensity stratum only reaches through
            # its correlation with heterogeneity.
            pool_all = np.argsort(-tau_hat)[:q]
        elif a.selection == "neighbor_ball":
            # A compact cluster of OBSERVATIONS: a random anchor unit plus its
            # q nearest neighbours in covariate space. Unlike a stratum, this
            # does not rank on any single scalar -- it asks about units that
            # resemble each other, which is the other sense in which a subset
            # can be "correlated".
            anchor = rng.integers(n)
            d2 = ((X - X[anchor]) ** 2).sum(axis=1)
            pool_all = np.argsort(d2)[:q]
        elif a.selection == "effect_oracle":
            # Top pool_frac by the TRUE effect. Not a procedure an analyst can
            # run; it is a controlled probe of the mechanism, isolating the
            # effect heterogeneity that concentrates the shared error term.
            # The propensity rule is oracle in the same way -- it ranks on the
            # true propensity from the analytic spec -- so this is the same
            # class of construction, not a weaker one.
            pool_all = np.argsort(-ite.numpy())[:q]
        elif a.selection == "quad_quartile":
            # Aligned with the quadratic block of phi(X), where the effect
            # basis puts most of its mass.
            pool_all = np.argsort(-(X**2).sum(axis=1))[:q]
        else:
            pool_all = np.arange(n)
        # rho_S: the within-POOL error correlation, which is the quantity the
        # design effect depends on once the subset is selected rather than
        # uniform. Equals rho when the pool is the whole sample.
        ep = unit_err[-1][pool_all]
        np_ = len(ep)
        pool_cbar.append((ep.sum() ** 2 - (ep ** 2).sum()) / (np_ * (np_ - 1)))
        pool_vbar.append(float(np.mean(ep ** 2)))
        # m_S is defined over S in eq (5), so it is restricted to the pool for
        # the same reason rho_S is. A model can be markedly more or less
        # conservative inside a subgroup than it is on average -- inside a
        # tight covariate neighbourhood m_S roughly doubles -- so a global m
        # is the wrong number to pair with a pool-specific rho_S.
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

            # Path A: the aggregate as the model represents it. For the
            # subset-decode arms that is one distribution for the whole
            # subset; for the latent arm it is the subset mean taken WITHIN
            # each z draw, so the shared component survives the average.
            if is_vanilla:
                # No aggregate decode exists for this arm, so Path A is not
                # merely equal to Path B -- it is undefined. Leaving the
                # accumulator empty reports NaN rather than a number.
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

            # Path B: k per-unit distributions, drawn INDEPENDENTLY, averaged.
            # Independent draws per unit is the property under test. For the
            # latent arm these are the SAME decodes with z marginalised out,
            # so A and B differ only in marginal vs joint.
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
    # k* from eq. (5), which is stated over S: both m_S and rho_S are
    # pool-restricted, so pairing m_S with the task-level rho would mix
    # scopes. They coincide only when the pool is the whole sample.
    kstar_pred = 1.0 + (m - 1.0) / rho_S if rho_S > 0 else float("inf")
    out = {"tag": a.tag, "checkpoint": a.checkpoint, "n": n, "ks": ks,
           "seed_cfg": cfg.get("seed"), "selection": a.selection, "pool_frac": a.pool_frac, "head": kind,
           "arm": cfg["model_variant"], "recipe_drift": recipe_drift,
           "eval_ood": eval_ood,
           "num_tasks": a.num_tasks, "n_sub": a.n_sub,
           "sigma2": sigma2, "sigma2_S": sigma2_S, "rho": rho, "rho_S": rho_S,
           "m": m, "m_global": m_global, "kstar_pred": kstar_pred}
    # Post-hoc correction on the composed path: scale the half-width by
    # sqrt(1 + (k-1) rho_S), which turns R = m_S/[1+(k-1)rho_S] into exactly
    # m_S. The prediction is therefore 2*Phi(z*sqrt(m_S)) - 1: FLAT in k, and
    # equal to nominal only when m_S = 1. Any residual slope in k would mean
    # the design effect does not account for all of the k-dependence.
    # Three constructions, all rescalings of the stored half-width:
    #   A  as reported                    factor 1
    #   B  per-unit recalibration only    factor 1/sqrt(m_S)
    #   C  recalibration + design effect  factor sqrt(Deff/m_S)
    # On cells with m_S > 1, B is expected to fall FASTER than A, since the
    # cushion carrying A to large k is exactly what recalibration removes.
    # On cells with m_S < 1 the sign flips and B lifts coverage instead.
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
