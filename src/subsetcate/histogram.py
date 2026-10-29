"""Histogram-head loss, quantiles and moments.

A histogram head decodes ``nbins`` logits over a fixed support
[support_min, support_max] of standardized effects.
Shapes: logits/probs [..., nbins], values [...].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bin_edges(support_min: float, support_max: float, nbins: int) -> torch.Tensor:
    if not support_min < support_max or nbins < 2:
        raise ValueError("Need support_min < support_max and nbins >= 2")
    return torch.linspace(support_min, support_max, nbins + 1)


def bin_centers(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])


def soft_targets(
    values: torch.Tensor, edges: torch.Tensor, smoothing_sigma_bins: float = 1.0
) -> torch.Tensor:
    """Target distribution over bins for scalar targets ``values``.

    A Gaussian bump of width smoothing_sigma_bins (in bins) on the clamped
    target, or one-hot if that width is 0. Out-of-support values clamp to the
    edge bins (see clamp_fraction).
    """
    centers = bin_centers(edges).to(values.device)
    width = (edges[1] - edges[0]).to(values.device)
    clamped = values.clamp(min=centers[0].item(), max=centers[-1].item())

    if smoothing_sigma_bins == 0:
        idx = torch.bucketize(clamped, edges.to(values.device)[1:-1])
        return F.one_hot(idx, num_classes=len(centers)).to(values.dtype)

    sigma = smoothing_sigma_bins * width
    logq = -0.5 * ((centers - clamped.unsqueeze(-1)) / sigma) ** 2
    return torch.softmax(logq, dim=-1)


def clamp_fraction(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    out = (values < edges[0]) | (values > edges[-1])
    return out.float().mean()


def histogram_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    edges: torch.Tensor,
    smoothing_sigma_bins: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy against the (smoothed) target bin distribution; the
    histogram NLL up to the constant log bin width."""
    q = soft_targets(values, edges, smoothing_sigma_bins)
    return -(q * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def histogram_quantiles(
    probs: torch.Tensor, edges: torch.Tensor, qs: torch.Tensor
) -> torch.Tensor:
    """Quantiles via linear interpolation of the CDF within bins.

    probs: [..., nbins] (rows sum to 1). qs: [Q]. Returns [..., Q].
    """
    edges = edges.to(probs.device)
    cdf = probs.cumsum(dim=-1).clamp(max=1.0)
    cdf_full = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [..., nbins+1]

    qs = qs.to(probs.device).expand(*probs.shape[:-1], -1)
    # index of first edge with cdf >= q  (in 1..nbins)
    idx = torch.searchsorted(cdf_full, qs.contiguous(), right=False).clamp(
        min=1, max=probs.shape[-1]
    )
    c_lo = torch.gather(cdf_full, -1, idx - 1)
    c_hi = torch.gather(cdf_full, -1, idx)
    e_lo = edges[idx - 1]
    e_hi = edges[idx]
    frac = (qs - c_lo) / (c_hi - c_lo).clamp(min=1e-12)
    return e_lo + frac.clamp(0.0, 1.0) * (e_hi - e_lo)


def histogram_mean_var(
    probs: torch.Tensor, edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of the histogram predictive. probs [..., nbins] ->
    (mean [...], var [...])."""
    centers = bin_centers(edges).to(probs.device)
    mean = (probs * centers).sum(dim=-1)
    second = (probs * centers**2).sum(dim=-1)
    return mean, (second - mean**2).clamp(min=0.0)


