"""Per-draw predictives of the latent NP (route_a) under a shared z.

experiments/two_path_ablation.py forms both intervals from these draws:
SCM-marginalized by aggregating units within each draw of z, unit-level
by drawing every unit independently.
"""

from __future__ import annotations

import torch



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


