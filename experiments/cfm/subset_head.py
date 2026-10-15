"""A direct-decode subset head on frozen CausalPFN embeddings.

Per unit i the head sees u_i = [h(x_i, 0), h(x_i, 1), m_i, s_i]: CausalPFN's
last hidden states for both arms, plus the per-unit CATE mean and sd implied
by its two histograms (task-standardized units). It pools a projection of u
over the whole task (r_ctx) and over the queried subset S (r_S) and decodes a
Gaussian for theta_S from [r_ctx, r_S, log k, k / n]. It is trained on the
exact subset-CATE by Gaussian NLL, so the SCM-level dependence between units
enters through the pooled representation rather than through an independence
assumption at aggregation time.

The unit-level baseline, `unit_level_interval`, is CausalPFN's own aggregate
construction applied to S: one independent draw per unit from each arm's
histogram, differenced, averaged over S, 2.5/97.5% percentiles.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB = 384


def unit_features(e: dict, s_y: float, bin_centers: torch.Tensor) -> torch.Tensor:
    """[n, 2*384 + 2] per-unit inputs from `causalpfn_embed.cate_embeddings`."""
    def moments(logits, scale):
        p = torch.softmax(logits, -1)
        m = (p * bin_centers).sum(-1)
        v = (p * bin_centers ** 2).sum(-1) - m ** 2
        return m * scale, v.clamp(min=0) * scale ** 2
    m0, v0 = moments(e["logits0"], e["scale0"])
    m1, v1 = moments(e["logits1"], e["scale1"])
    cate = ((m1 + e["shift1"]) - (m0 + e["shift0"])) / s_y
    sd = torch.sqrt(v0 + v1) / s_y
    return torch.cat([e["h0"], e["h1"], cate[:, None], sd[:, None]], dim=-1)


class SubsetHead(nn.Module):
    def __init__(self, d_in: int = 2 * EMB + 2, width: int = 256):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, width))
        self.dec = nn.Sequential(nn.Linear(2 * width + 2, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, u: torch.Tensor, mask: torch.Tensor):
        """u [n, d_in] or [B, n, d_in]; mask [M, n] or [B, M, n] boolean
        subsets -> (mean, sd), each [M] or [B, M]."""
        single = u.dim() == 2
        if single:
            u, mask = u[None], mask[None]
        z = self.phi(u)                                   # [B, n, W]
        m = mask.float()
        k = m.sum(-1, keepdim=True)                       # [B, M, 1]
        r_ctx = z.mean(1, keepdim=True).expand(-1, m.shape[1], -1)
        r_s = (m @ z) / k
        n = u.shape[1]
        out = self.dec(torch.cat([r_ctx, r_s, torch.log(k), k / n], dim=-1))
        mean, sd = out[..., 0], F.softplus(out[..., 1]) + 1e-4
        return (mean[0], sd[0]) if single else (mean, sd)


def make_head(args: dict) -> tuple[nn.Module, str]:
    """The head a checkpoint's args describe, and its kind: "cnp" (a Gaussian
    for theta_S) or "latent"."""
    kind = args.get("head", "cnp")
    if kind == "latent" and args.get("latent_output", "gaussian") == "histogram":
        return LatentHistSubsetHead(width=args["width"], nbins=args["hist_nbins"],
                                    support=(args["hist_support_min"], args["hist_support_max"]),
                                    smoothing_bins=args["hist_smoothing"]), kind
    if kind == "latent":
        return LatentSubsetHead(width=args["width"]), kind
    return SubsetHead(width=args["width"]), "cnp"


def gaussian_nll(mean, sd, target):
    return (torch.log(sd) + 0.5 * ((target - mean) / sd) ** 2).mean() + 0.5 * math.log(2 * math.pi)


def sample_masks(n: int, m: int, gen: torch.Generator) -> torch.Tensor:
    """m random subsets, size log-uniform in [1, n]."""
    k = torch.exp(torch.rand(m, generator=gen) * math.log(n)).round().clamp(1, n).long()
    ranks = torch.rand(m, n, generator=gen).argsort(1).argsort(1)
    return ranks < k[:, None]


def sample_subgroup_masks(X: torch.Tensor, t: torch.Tensor, m: int, gen: torch.Generator,
                          p_random: float = 0.7, p_treated: float = 0.1) -> torch.Tensor:
    """m subsets of one task: uniform random (size log-uniform in [1, n]),
    treated-only (all treated units, or a random share of them), and
    covariate-defined (units above or below a random quantile of a random
    covariate) -- the queries an analyst would ask. Empty draws fall back
    to one random unit."""
    n, d = X.shape
    kind = torch.rand(m, generator=gen)
    masks = sample_masks(n, m, gen)
    treated = t > 0.5
    for i in range(m):
        if kind[i] < p_random:
            continue
        if kind[i] < p_random + p_treated:
            keep = torch.rand(n, generator=gen) < torch.rand(1, generator=gen).clamp(min=0.1)
            masks[i] = treated & keep if torch.rand(1, generator=gen) < 0.5 else treated
        else:
            j = int(torch.randint(d, (1,), generator=gen))
            q = torch.quantile(X[:, j], float(torch.rand(1, generator=gen) * 0.8 + 0.1))
            masks[i] = X[:, j] > q if torch.rand(1, generator=gen) < 0.5 else X[:, j] <= q
        if not masks[i].any():
            masks[i, int(torch.randint(n, (1,), generator=gen))] = True
    return masks


@torch.no_grad()
def per_unit(head, kind, u):
    """The head's per-unit predictive mean and sd (sd(y) units), each [n]: the
    CNP queried with every single unit as its subset; the latent head's
    marginal over 200 draws of z (mean, and total variance)."""
    n = u.shape[0]
    if kind == "latent":
        m, s = head.unit_params(u[None], 200)              # [K, 1, n]
        m, s = m[:, 0], s[:, 0]
        return m.mean(0), torch.sqrt(m.var(0, unbiased=False) + (s ** 2).mean(0))
    means, sds = [], []
    for c in range(0, n, 1024):
        idx = torch.arange(c, min(c + 1024, n), device=u.device)
        mask = torch.zeros(len(idx), n, dtype=torch.bool, device=u.device)
        mask[torch.arange(len(idx)), idx] = True
        mu, sd = head(u, mask)
        means.append(mu)
        sds.append(sd)
    return torch.cat(means), torch.cat(sds)


@torch.no_grad()
def unit_level_interval(e: dict, bin_centers, mask: torch.Tensor, ndraw: int, gen: torch.Generator,
                        temperature: float = 1.0):
    """CausalPFN's independent-draw aggregate interval for each subset (raw
    units), at the given temperature (1 = API default; the fitted temperature
    is what calibrate=True uses)."""
    def draws(logits, scale, shift):
        p = torch.softmax(logits / temperature, -1)
        idx = torch.multinomial(p, ndraw, replacement=True, generator=gen)   # [n, ndraw]
        return bin_centers[idx] * scale[:, None] + shift[:, None]
    tau_draws = draws(e["logits1"], e["scale1"], e["shift1"]) - draws(e["logits0"], e["scale0"], e["shift0"])
    m = mask.float()
    agg = (m @ tau_draws) / m.sum(1, keepdim=True)                            # [M, ndraw]
    return torch.quantile(agg, 0.025, dim=1), torch.quantile(agg, 0.975, dim=1)


class LatentSubsetHead(nn.Module):
    """Latent-NP head: z ~ N(mu(r_ctx), sigma(r_ctx)), 16-d; each unit's effect
    ~ N(m_i(z), s_i(z)^2) in task-standardized units, independent given z.

    Trained with the `combined` objective of the paper's latent NP: the
    per-unit log-likelihood averaged over K draws of z, plus lam times the log
    of the Monte Carlo average over draws of a Gaussian likelihood for theta_S,
    with the units of S aggregated within each draw. No KL term.
    """

    def __init__(self, d_in: int = 2 * EMB + 2, width: int = 256, latent: int = 16):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, width))
        self.prior = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * latent))
        self.z_proj = nn.Linear(latent, width)
        self.dec = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.latent = latent

    def _decode(self, u: torch.Tensor, K: int) -> torch.Tensor:
        """u [B, n, d_in] -> decoder output per draw of z and unit, [K, B, n, out]."""
        z_u = self.phi(u)                                  # [B, n, W]
        r = z_u.mean(1)                                    # [B, W]
        mu, raw = self.prior(r).chunk(2, -1)
        sig = F.softplus(raw) + 1e-4
        z = mu + sig * torch.randn(K, *mu.shape, device=u.device)          # [K, B, L]
        n = u.shape[1]
        h = torch.cat([z_u.expand(K, -1, -1, -1), r[None, :, None].expand(K, -1, n, -1),
                       self.z_proj(z)[:, :, None].expand(-1, -1, n, -1)], -1)
        return self.dec(h)

    def unit_params(self, u: torch.Tensor, K: int):
        """u [B, n, d_in] -> per-draw unit means and sds, each [K, B, n]."""
        out = self._decode(u, K)
        return out[..., 0], F.softplus(out[..., 1]) + 1e-4

    def loss(self, u, mask, tau, theta, K: int = 16, lam: float = 2.0):
        """u [B, n, d]; mask [B, M, n]; tau [B, n] and theta [B, M], standardized.

        The per-unit NLL averaged over the K draws (the paper's latent NP), plus
        lam times the negative log of the Monte Carlo average over draws of the
        Gaussian likelihood of theta_S."""
        m, s = self.unit_params(u, K)                                      # [K, B, n]
        unit = (torch.log(s) + 0.5 * ((tau - m) / s) ** 2).mean()
        mk = mask.float()
        k = mk.sum(-1)                                                     # [B, M]
        mu_s = torch.einsum("kbn,bmn->kbm", m, mk) / k
        var_s = torch.einsum("kbn,bmn->kbm", s ** 2, mk) / k ** 2 + 1e-6
        logp = -0.5 * torch.log(2 * math.pi * var_s) - 0.5 * (theta - mu_s) ** 2 / var_s
        func = -(torch.logsumexp(logp, 0) - math.log(K)).mean()
        return unit + lam * func + 0.5 * math.log(2 * math.pi), dict(unit=float(unit), func=float(func))

    @torch.no_grad()
    def scm_draws(self, u: torch.Tensor, mask: torch.Tensor, K: int = 500, chunk: int = 50) -> torch.Tensor:
        """K draws of theta_S / sd(y) under SCM-marginalized aggregation, [K, M]:
        the first half of `intervals`, kept separate so pooling draws across
        heads leaves `intervals` (and its random stream) unchanged."""
        mk = mask.float()
        k = mk.sum(-1)
        out = []
        for _ in range(K // chunk):
            m, s = self.unit_params(u[None], chunk)
            m, s = m[:, 0], s[:, 0]
            mu_s = (m @ mk.T) / k
            sd_s = torch.sqrt((s ** 2) @ mk.T) / k
            out.append(mu_s + sd_s * torch.randn_like(mu_s))
        return torch.cat(out)

    def intervals(self, u: torch.Tensor, mask: torch.Tensor, K: int = 500, chunk: int = 50):
        """Both aggregations from the same weights, for one task.
        u [n, d]; mask [M, n] -> (scm_lo, scm_hi, unit_lo, unit_hi), each [M]."""
        mk = mask.float()
        k = mk.sum(-1)
        scm, unit = [], []
        for _ in range(K // chunk):
            m, s = self.unit_params(u[None], chunk)                        # [c, 1, n]
            m, s = m[:, 0], s[:, 0]
            # SCM-marginalized: all units of S share the draw of z
            mu_s = (m @ mk.T) / k
            sd_s = torch.sqrt((s ** 2) @ mk.T) / k
            scm.append(mu_s + sd_s * torch.randn_like(mu_s))
            # unit-level: each unit gets its own draw of z, then average
            perm = torch.stack([torch.randperm(chunk, device=u.device) for _ in range(m.shape[1])], 1)
            mi = torch.gather(m, 0, perm)
            si = torch.gather(s, 0, perm)
            tau_d = mi + si * torch.randn_like(mi)
            unit.append((tau_d @ mk.T) / k)
        scm, unit = torch.cat(scm), torch.cat(unit)                        # [K, M]
        q = lambda a: (torch.quantile(a, 0.025, 0), torch.quantile(a, 0.975, 0))
        return (*q(scm), *q(unit))


def _paper_histogram():
    """The paper's histogram primitives (src/subsetcate/histogram.py), imported
    rather than copied so the definitions cannot drift."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
    if root not in sys.path:
        sys.path.insert(0, root)
    from subsetcate import histogram
    return histogram


class LatentHistSubsetHead(LatentSubsetHead):
    """The latent head with the paper's histogram per-unit output (its latent NP,
    `route_a` with `estimator_type: histogram`): for each draw of z the decoder
    gives each unit nbins logits over a fixed support.

    Loss, as the paper's `combined` objective (modules.py `_route_a_functional_ml`):
    the per-draw histogram NLL (cross-entropy against Gaussian-smoothed soft
    targets) averaged over the K draws, plus lam times the Monte Carlo
    functional likelihood, where each draw's subset mean has mean and variance
    from the units' histogram means and variances. Intervals as the paper's
    functional_sampling: per-unit inverse-CDF draws from the histograms,
    sharing each z across units (SCM-marginalized) or giving every unit its own
    z (unit-level).
    """

    def __init__(self, d_in: int = 2 * EMB + 2, width: int = 256, latent: int = 16, nbins: int = 64,
                 support: tuple[float, float] = (-4.0, 4.0), smoothing_bins: float = 1.0):
        super().__init__(d_in, width, latent)
        self.dec[-1] = nn.Linear(width, nbins)
        self.register_buffer("edges", _paper_histogram().bin_edges(float(support[0]), float(support[1]), nbins))
        self.smoothing_bins = smoothing_bins

    def unit_logits(self, u: torch.Tensor, K: int) -> torch.Tensor:
        return self._decode(u, K)                                          # [K, B, n, nbins]

    def unit_params(self, u: torch.Tensor, K: int):
        """Per-draw histogram mean and sd, each [K, B, n]."""
        m, v = _paper_histogram().histogram_mean_var(torch.softmax(self.unit_logits(u, K), -1), self.edges)
        return m, torch.sqrt(v + 1e-12)

    def loss(self, u, mask, tau, theta, K: int = 16, lam: float = 2.0):
        H = _paper_histogram()
        logits = self.unit_logits(u, K)                                    # [K, B, n, nbins]
        unit = torch.stack([H.histogram_loss(logits[j], tau, self.edges, self.smoothing_bins) for j in range(K)]).mean()
        m, v = H.histogram_mean_var(torch.softmax(logits, -1), self.edges)  # [K, B, n]
        mk = mask.float()
        k = mk.sum(-1)
        mu_s = torch.einsum("kbn,bmn->kbm", m, mk) / k
        var_s = torch.einsum("kbn,bmn->kbm", v, mk) / k ** 2 + 1e-6
        logp = torch.distributions.Normal(mu_s, var_s.sqrt()).log_prob(theta)
        func = -(torch.logsumexp(logp, 0) - math.log(K)).mean()
        return unit + lam * func, dict(unit=float(unit), func=float(func),
                                       clamp=float(H.clamp_fraction(tau, self.edges)))

    def _unit_draws(self, u: torch.Tensor, chunk: int) -> torch.Tensor:
        """Per-unit values drawn from each draw's histograms: [chunk, n]."""
        probs = torch.softmax(self.unit_logits(u[None], chunk)[:, 0], -1)  # [c, n, nbins]
        q = torch.rand(*probs.shape[:-1], 1, device=u.device)
        return _paper_histogram().histogram_quantiles(probs, self.edges, q).squeeze(-1)

    @torch.no_grad()
    def scm_draws(self, u, mask, K: int = 500, chunk: int = 50):
        mk = mask.float()
        return torch.cat([(self._unit_draws(u, chunk) @ mk.T) / mk.sum(-1) for _ in range(K // chunk)])

    @torch.no_grad()
    def intervals(self, u, mask, K: int = 500, chunk: int = 50):
        mk = mask.float()
        k = mk.sum(-1)
        scm, unit = [], []
        for _ in range(K // chunk):
            d = self._unit_draws(u, chunk)                                 # [c, n]
            scm.append((d @ mk.T) / k)
            perm = torch.stack([torch.randperm(chunk, device=u.device) for _ in range(d.shape[1])], 1)
            unit.append((torch.gather(d, 0, perm) @ mk.T) / k)
        scm, unit = torch.cat(scm), torch.cat(unit)
        q = lambda a: (torch.quantile(a, 0.025, 0), torch.quantile(a, 0.975, 0))
        return (*q(scm), *q(unit))
