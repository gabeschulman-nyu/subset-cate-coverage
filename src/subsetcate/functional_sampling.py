"""Inference-time functional sampling for Route A (Phase M6d).

Joint (across-draw) sampler — the method:
    z_k ~ p(z|ctx); per-unit τ_i^(k) ~ p(τ_i | z_k) sampled independently
    WITHIN the draw (the model's factorization given z); θ_S^(k) = masked
    mean over S. Quantiles across the K draws give the functional interval.
    Cross-unit correlation enters exclusively through the shared z_k.

Independent (collapse) baseline — computed on the SAME model:
    each unit's τ_i is drawn from its MARGINAL predictive
    p(τ_i) = E_z[p(τ_i|z)] independently across units, then masked-mean.
    This is exactly the per-unit-then-average aggregation (CausalPFN's
    mechanism): no shared z, so cross-unit correlation is destroyed.
    Distributionally identical to giving every unit its own independent z.

Per-unit sampling from a histogram uses inverse-CDF with a uniform draw,
which reuses histogram_quantiles (within-bin uniform).
"""

from __future__ import annotations

import torch

from subsetcate.histogram import histogram_quantiles


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """values [B, n], mask [B, n] bool -> [B]."""
    m = mask.to(values.dtype)
    return (values * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)


def _sample_from_histograms(probs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw per row: probs [B, n, nbins] -> τ [B, n]."""
    u = torch.rand(*probs.shape[:-1], 1, device=probs.device)
    return histogram_quantiles(probs, edges, u).squeeze(-1)


@torch.no_grad()
def route_a_predictive_probs(
    backbone, batch: dict[str, torch.Tensor], num_draws: int
) -> torch.Tensor:
    """Per-draw per-unit predictives under shared z: [K, B, n, nbins]."""
    h, r_agg = backbone.encode(
        batch["x_features"], batch["x_sources"], lengths=batch.get("lengths")
    )
    mu_p, sigma_p = backbone.latent_heads.prior(r_agg)
    all_probs = []
    for _ in range(num_draws):
        z = backbone.latent_heads.sample(mu_p, sigma_p)
        all_probs.append(torch.softmax(backbone.decode_units(h, r_agg, z), dim=-1))
    return torch.stack(all_probs)


@torch.no_grad()
def joint_functional_draws(
    backbone,
    batch: dict[str, torch.Tensor],
    num_draws: int,
    probs_per_draw: torch.Tensor | None = None,
    edges: torch.Tensor | None = None,
) -> torch.Tensor:
    """Across-draw functional samples θ_S^(k): [K, B]. Correlation through
    the shared z_k of each draw."""
    if probs_per_draw is None:
        probs_per_draw = route_a_predictive_probs(backbone, batch, num_draws)
    mask = batch["subset_mask"]
    draws = []
    for k in range(probs_per_draw.size(0)):
        tau = _sample_from_histograms(probs_per_draw[k], edges)
        draws.append(_masked_mean(tau, mask))
    return torch.stack(draws)


@torch.no_grad()
def independent_functional_draws(
    probs_per_draw: torch.Tensor,
    batch: dict[str, torch.Tensor],
    num_draws: int,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Collapse-baseline functional samples on the same model: units drawn
    independently from their MARGINAL predictives (z averaged out), then
    masked-mean. Returns [num_draws, B]."""
    marginal = probs_per_draw.mean(dim=0)  # [B, n, nbins]
    mask = batch["subset_mask"]
    draws = []
    for _ in range(num_draws):
        tau = _sample_from_histograms(marginal, edges)
        draws.append(_masked_mean(tau, mask))
    return torch.stack(draws)


@torch.no_grad()
def naive_average_functional_draws(
    model,
    batch: dict[str, torch.Tensor],
    num_draws: int,
    edges: torch.Tensor,
) -> torch.Tensor:
    """The naive-average BASELINE ARM (four-way ladder, M7): a vanilla
    per-unit model (predict_ate=False, no latent) queried at the subset
    units, each unit's τ drawn independently from its own predictive, then
    averaged. This is CausalPFN's aggregation — the arm that collapses and
    motivates the paper. Requires a constant subset size k within the batch
    (the eval harness generates fixed-k batches).

    The query features are the subset units' OBSERVED ROWS (carrying y_i and
    t_i, upstream's ITE-query convention), so the per-unit predictive can be
    counterfactual; the collapse this arm exhibits is purely the missing
    cross-unit correlation, not a starved query. Handles arbitrary
    (variable-k) masks: every context row is queried at its own features and
    the masked mean selects the subset.
    """
    mask = batch["subset_mask"]  # [B, n] bool, any per-task k
    logits = model(
        batch["x_features"], batch["x_sources"], batch["x_features"],
        lengths=batch.get("lengths"),
    )
    probs = torch.softmax(logits, dim=-1)  # [B, n, nbins]

    draws = []
    for _ in range(num_draws):
        tau = _sample_from_histograms(probs, edges)  # independent across units
        draws.append(_masked_mean(tau, mask))
    return torch.stack(draws)


def interval_from_draws(
    draws: torch.Tensor, confidence_level: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Empirical (lower, median, upper) across draws: [K, B] -> three [B]."""
    lo = (1 - confidence_level) / 2
    qs = torch.tensor([lo, 0.5, 1 - lo], device=draws.device, dtype=draws.dtype)
    out = torch.quantile(draws, qs, dim=0)
    return out[0], out[1], out[2]
