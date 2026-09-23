"""Route A latent-variable machinery (Phase M6a): Gaussian latent heads.

The latent z attaches at r_agg — the single per-dataset summary both backbones
already compute (docs/metadentify_map.md §3). This module provides the two
heads and the KL; injection into the decoders is M6b, the ELBO is M6c.

- Prior head:      p(z | ctx)      = N(μ_p, diag σ_p²),  from r_agg alone.
- Posterior head:  q(z | ctx, tgt) = N(μ_q, diag σ_q²),  from r_agg plus a
  permutation-invariant summary of the targets (per-target MLP, mean-pooled —
  targets are a set, and mean-pooling matches the deterministic path's
  aggregation bias).

σ is parameterized via softplus with a small floor so KL stays finite and
differentiable everywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_FLOOR = 1e-4


class GaussianLatentHeads(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 16,
        target_dim: int = 1,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or embed_dim
        self.latent_dim = latent_dim

        self.prior_head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * latent_dim)
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(embed_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    @staticmethod
    def _split(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw_sigma = params.chunk(2, dim=-1)
        sigma = F.softplus(raw_sigma) + SIGMA_FLOOR
        return mu, sigma

    def prior(self, r_agg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, embed_dim] -> (μ_p, σ_p), each [B, latent_dim]."""
        return self._split(self.prior_head(r_agg))

    def posterior(
        self,
        r_agg: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """r_agg [B, embed_dim], targets [B, T, target_dim] (a set) ->
        (μ_q, σ_q), each [B, latent_dim]. target_mask [B, T] optionally marks
        valid targets (True = valid)."""
        t_emb = self.target_encoder(targets)  # [B, T, hidden]
        if target_mask is not None:
            m = target_mask.unsqueeze(-1).to(t_emb.dtype)
            t_summary = (t_emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            t_summary = t_emb.mean(dim=1)
        return self._split(self.posterior_head(torch.cat([r_agg, t_summary], dim=-1)))

    @staticmethod
    def sample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterized draw: z = μ + σ·ε (differentiable in μ, σ)."""
        return mu + sigma * torch.randn_like(sigma)

    @staticmethod
    def kl_divergence(
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q‖p) between diagonal Gaussians, per-dimension: [B, latent_dim].

        Returned unsummed so M6c can apply free-bits (a per-dimension floor)
        before reducing.
        """
        var_ratio = (sigma_q / sigma_p) ** 2
        return 0.5 * (
            var_ratio
            + ((mu_q - mu_p) / sigma_p) ** 2
            - 1.0
            - torch.log(var_ratio)
        )
