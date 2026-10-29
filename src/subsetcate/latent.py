"""Gaussian latent heads for the latent NP (route_a).

- Prior head:      p(z | ctx)      = N(mu_p, diag sigma_p^2), from r_agg.
- Posterior head:  q(z | ctx, tgt) = N(mu_q, diag sigma_q^2), from r_agg and
  a mean-pooled per-target MLP embedding of the target set.

The trained latent NP samples z from the prior head only; the posterior head
is used for the logged KL diagnostic. sigma = softplus(.) + SIGMA_FLOOR, so
the KL stays finite.
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
        """[B, embed_dim] -> (mu_p, sigma_p), each [B, latent_dim]."""
        return self._split(self.prior_head(r_agg))

    def posterior(
        self,
        r_agg: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """r_agg [B, embed_dim], targets [B, T, target_dim] (a set) ->
        (mu_q, sigma_q), each [B, latent_dim]. target_mask [B, T] marks valid
        targets (True = valid)."""
        t_emb = self.target_encoder(targets)  # [B, T, hidden]
        if target_mask is not None:
            m = target_mask.unsqueeze(-1).to(t_emb.dtype)
            t_summary = (t_emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            t_summary = t_emb.mean(dim=1)
        return self._split(self.posterior_head(torch.cat([r_agg, t_summary], dim=-1)))

    @staticmethod
    def sample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterized draw z = mu + sigma * eps."""
        return mu + sigma * torch.randn_like(sigma)

    @staticmethod
    def kl_divergence(
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q || p) between diagonal Gaussians, per dimension: [B, latent_dim]."""
        var_ratio = (sigma_q / sigma_p) ** 2
        return 0.5 * (
            var_ratio
            + ((mu_q - mu_p) / sigma_p) ** 2
            - 1.0
            - torch.log(var_ratio)
        )
