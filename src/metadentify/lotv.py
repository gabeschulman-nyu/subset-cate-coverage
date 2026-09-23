"""LOTV-structured Gaussian head (pre-registered 2026-07-14).

Every fixed head learned a near-k-invariant width (histogram at any sigma,
MDN with no floor): the direct-decode NLL converges to a constant width at
the arm's modal error scale, so arms with k-heterogeneous error are
mis-calibrated somewhere. This head supplies conditional dispersion BY
CONSTRUCTION via the law-of-total-variance shape of a subset mean:

    sigma^2(S) = c_hat(S) + (v_hat(S) - c_hat(S)) / |S|

c_hat: the |S|->inf variance floor (shared/epistemic component);
v_hat: the |S|=1 variance (single-unit component). Both softplus scalars
read off the subset representation; |S| enters ARITHMETICALLY (computed
from the mask, never inferred from tokens). The mechanism check this
enables: additive families should learn v_hat ~= c_hat (k-invariant,
correctly); hetero families v_hat > c_hat (real 1/|S| decay).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lotv_split(
    params: torch.Tensor, subset_mask: torch.Tensor, min_sigma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """params [B, 3] = (mu, raw_v, raw_c); returns (mu, sigma, c, v)."""
    mu = params[:, 0]
    v = F.softplus(params[:, 1]) + min_sigma**2
    c = F.softplus(params[:, 2]) + min_sigma**2
    s = subset_mask.sum(dim=1).clamp(min=1).to(params.dtype)
    var = (c + (v - c) / s).clamp_min(min_sigma**2)
    return mu, var.sqrt(), c, v


def lotv_nll(
    params: torch.Tensor, target: torch.Tensor, subset_mask: torch.Tensor, min_sigma: float
) -> torch.Tensor:
    mu, sigma, _, _ = lotv_split(params, subset_mask, min_sigma)
    return -torch.distributions.Normal(mu, sigma).log_prob(target).mean()


def lotv_sample(
    params: torch.Tensor, subset_mask: torch.Tensor, num_draws: int, min_sigma: float
) -> torch.Tensor:
    """[num_draws, B] Gaussian draws — the shared eval-protocol format."""
    mu, sigma, _, _ = lotv_split(params, subset_mask, min_sigma)
    return mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
        num_draws, mu.size(0), device=params.device, dtype=params.dtype
    )
