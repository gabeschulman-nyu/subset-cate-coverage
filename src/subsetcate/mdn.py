"""Mixture-density output head for the direct-decode arm (route_b).

The histogram likelihood head that route_a's ELBO requires imposes a width
floor ~ 2*1.96*sigma_eff on every predictive interval (the sigma-sweep
finding: smoothing acts as a global width dial). The direct-decode arms
train a single functional NLL — a proper scoring rule under any continuous
density — so they are free to use a K-component Gaussian mixture with no
bin-width floor and no fixed smoothing: arbitrarily sharp where the model
is confident, wide where it is not.

params layout from the decoder: [..., 3K] = (pi logits, mu, raw sigma),
sigma = softplus(raw) + min_sigma. min_sigma exists only for NLL stability
on exact point targets (default 1e-3 in effect units — two orders below
any oracle interval width observed, so it is not a floor in practice).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mdn_split(
    params: torch.Tensor, min_sigma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[..., 3K] -> (log_pi, mu, sigma), each [..., K]."""
    k = params.size(-1) // 3
    logit_pi, mu, raw_sigma = params.split(k, dim=-1)
    return torch.log_softmax(logit_pi, dim=-1), mu, F.softplus(raw_sigma) + min_sigma


def mdn_nll(params: torch.Tensor, target: torch.Tensor, min_sigma: float) -> torch.Tensor:
    """Mean negative log-likelihood of target [B] under params [B, 3K]."""
    log_pi, mu, sigma = mdn_split(params, min_sigma)
    comp = torch.distributions.Normal(mu, sigma).log_prob(target.unsqueeze(-1))
    return -torch.logsumexp(log_pi + comp, dim=-1).mean()


def mdn_sample(params: torch.Tensor, num_draws: int, min_sigma: float) -> torch.Tensor:
    """Ancestral samples from params [B, 3K] -> [num_draws, B] (the draws
    format every arm's eval uses)."""
    log_pi, mu, sigma = mdn_split(params, min_sigma)
    idx = torch.distributions.Categorical(logits=log_pi).sample((num_draws,))  # [S, B]
    mu_s = torch.gather(mu.unsqueeze(0).expand(num_draws, -1, -1), 2, idx.unsqueeze(-1))
    sg_s = torch.gather(sigma.unsqueeze(0).expand(num_draws, -1, -1), 2, idx.unsqueeze(-1))
    mu_s, sg_s = mu_s.squeeze(-1), sg_s.squeeze(-1)
    return mu_s + sg_s * torch.randn_like(mu_s)


def mdn_quantiles(
    params: torch.Tensor, qs: torch.Tensor, min_sigma: float, num_grid: int = 4096
) -> torch.Tensor:
    """Analytic CDF-inversion quantiles on a shared support grid.
    params [B, 3K], qs [Q] -> [B, Q]. Used for val metrics; the eval
    batteries use mdn_sample draws for protocol parity with other arms."""
    log_pi, mu, sigma = mdn_split(params, min_sigma)
    lo = (mu - 6 * sigma).min().item()
    hi = (mu + 6 * sigma).max().item()
    grid = torch.linspace(lo, hi, num_grid, device=params.device)
    z = (grid.view(1, -1, 1) - mu.unsqueeze(1)) / sigma.unsqueeze(1)  # [B, G, K]
    cdf = (torch.distributions.Normal(0.0, 1.0).cdf(z) * log_pi.exp().unsqueeze(1)).sum(-1)
    qgrid = qs.to(params.device).view(1, -1).expand(cdf.size(0), -1).contiguous()
    idx = torch.searchsorted(cdf.contiguous(), qgrid)
    return grid[idx.clamp(max=num_grid - 1)]
