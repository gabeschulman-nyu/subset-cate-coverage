import warnings
from typing import Any

import numpy as np
import pytorch_lightning as pl
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from metadentify.baselines import get_baselines
from metadentify.histogram import (
    bin_edges,
    clamp_fraction,
    histogram_loss,
    histogram_mean_var,
    histogram_quantiles,
)
from metadentify.latent import GaussianLatentHeads
from metadentify.lotv import lotv_nll, lotv_split
from metadentify.mdn import mdn_nll, mdn_quantiles


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, embedding_dim: int, num_cosines: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU())
        self.register_buffer("i_pi", torch.arange(1, num_cosines + 1).float() * np.pi)

    def forward(self, taus):
        return self.net(torch.cos(taus * self.i_pi))


class MAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_out, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)
        self.ffn = nn.Sequential(
            nn.Linear(dim_out, dim_out * 4), nn.ReLU(), nn.Linear(dim_out * 4, dim_out)
        )

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor, y_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_out, _ = self.mha(query=X, key=Y, value=Y, key_padding_mask=y_mask)
        H = self.ln1(X + attn_out)
        return self.ln2(H + self.ffn(H))


class ISAB(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inducing_points: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inducing_points, dim_out))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(dim_out, dim_in, num_heads, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, num_heads, dropout=dropout)

    def forward(self, X: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.size(0)
        I_expanded = self.I.expand(B, -1, -1)
        H = self.mab0(I_expanded, X, y_mask=x_mask)
        return self.mab1(X, H, y_mask=None)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, num_heads, dropout=dropout)

    def forward(self, X: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.size(0)
        S_expanded = self.S.expand(B, -1, -1)
        return self.mab(S_expanded, X, y_mask=x_mask)


class CausalMetaModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        lambda_crossing_penalty: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_lr_scheduler: bool = False,
        run_baselines: bool = False,
        baseline_setting: str = "confounder",
        confidence_level: float = 0.95,
        val_metrics_normalized: bool = True,
        plot_results: bool = False,
        config: dict = None,
        custom_baselines: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "config", "custom_baselines"])
        self.backbone = backbone
        self.config = config or {}
        self.lambda_crossing_penalty = lambda_crossing_penalty
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler

        self.gaussian_nll = None
        if self.estimator_type == "gaussian":
            self.gaussian_nll = nn.GaussianNLLLoss(eps=1e-6)

        if self.estimator_type == "histogram":
            self.hist_smoothing_sigma_bins = float(
                self.config.get("hist_smoothing_sigma_bins", 1.0)
            )
            self.register_buffer(
                "hist_edges",
                bin_edges(
                    float(self.config.get("hist_support_min", -4.0)),
                    float(self.config.get("hist_support_max", 4.0)),
                    int(getattr(backbone, "nbins", self.config.get("hist_nbins", 64))),
                ),
            )

        if self.estimator_type in ("mixture", "lotv"):
            self.mdn_min_sigma = float(self.config.get("mdn_min_sigma", 1e-3))

        if getattr(backbone, "variant", "vanilla") == "route_a":
            self.kl_free_bits = float(self.config.get("kl_free_bits", 0.05))
            self.kl_warmup_steps = int(self.config.get("kl_warmup_steps", 1000))
            self.z_spread_draws = int(self.config.get("z_spread_draws", 8))
            self.route_a_objective = self.config.get("route_a_objective", "elbo")
            self.ml_train_draws = int(self.config.get("ml_train_draws", 16))
            self.fml_lambda = float(self.config.get("fml_lambda", 1.0))

        self.run_baselines = run_baselines
        self.baseline_setting = baseline_setting
        self.val_metrics_normalized = val_metrics_normalized
        self.confidence_level = confidence_level
        self.plot_results = plot_results

        self.baselines = None
        if self.run_baselines:
            if custom_baselines is not None:
                self.baselines = custom_baselines
            else:
                self.baselines = get_baselines(baseline_setting=self.baseline_setting)

        self.validation_step_outputs: list[dict[str, Any]] = []
        self.test_step_outputs: list[dict[str, Any]] = []

        self.conformal_q = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "config" in checkpoint:
            self.config = checkpoint["config"]

    @property
    def estimator_type(self) -> str:
        return getattr(self.backbone, "estimator_type", None)

    def forward(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        query_x: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        subset_mask: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> Any:
        return self.backbone(x_features, x_sources, query_x, taus, lengths, subset_mask, z)

    def _kl_beta(self, step: int) -> float:
        """Linear KL warmup 0 -> 1 over kl_warmup_steps."""
        if self.kl_warmup_steps <= 0:
            return 1.0
        return min(1.0, step / self.kl_warmup_steps)

    def _route_a_elbo(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Negative ELBO: per-unit histogram-NLL(θ_i | h_i, z~q) + β·KL_fb.

        Targets are the exact abducted ITEs of the context rows (query type
        context-ite). Unit-level supervision — aggregate functional
        calibration is M6d's inference-time claim, not this loss's.
        """
        bb = self.backbone
        targets = batch["query_value"]  # [B, n_ctx] exact abducted ITEs
        h, r_agg = bb.encode(
            batch["x_features"], batch["x_sources"], lengths=batch.get("lengths")
        )

        mu_q, sigma_q = bb.latent_heads.posterior(r_agg, targets.unsqueeze(-1))
        mu_p, sigma_p = bb.latent_heads.prior(r_agg)
        z = bb.latent_heads.sample(mu_q, sigma_q)
        logits = bb.decode_units(h, r_agg, z)  # [B, n_ctx, nbins]

        nll = histogram_loss(
            logits, targets, self.hist_edges, self.hist_smoothing_sigma_bins
        )
        kl_dims = bb.latent_heads.kl_divergence(mu_q, sigma_q, mu_p, sigma_p)
        kl_raw = kl_dims.sum(dim=-1).mean()
        kl_fb = kl_dims.clamp(min=self.kl_free_bits).sum(dim=-1).mean()
        beta = self._kl_beta(self.global_step)
        loss = nll + beta * kl_fb

        active_units = (kl_dims.mean(dim=0) > 0.01).sum()
        self.log_dict(
            {
                "train/nll": nll,
                "train/kl": kl_raw,
                "train/kl_dim_mean": kl_dims.mean(),
                "train/kl_beta": beta,
                "train/active_units": active_units.float(),
                "train/hist_clamp_frac": clamp_fraction(targets, self.hist_edges),
            }
        )
        return loss

    @torch.no_grad()
    def _z_spread_ratio(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Output-side collapse diagnostic: across-z variance of decoded
        per-unit effect means (K prior draws) over mean within-histogram
        variance. Near-zero with healthy per-dim KL = quiet collapse; the
        early M6d go/no-go read."""
        bb = self.backbone
        h, r_agg = bb.encode(
            batch["x_features"], batch["x_sources"], lengths=batch.get("lengths")
        )
        mu_p, sigma_p = bb.latent_heads.prior(r_agg)
        means, variances = [], []
        for _ in range(self.z_spread_draws):
            z = bb.latent_heads.sample(mu_p, sigma_p)
            probs = torch.softmax(bb.decode_units(h, r_agg, z), dim=-1)
            m, v = histogram_mean_var(probs, self.hist_edges)
            means.append(m)
            variances.append(v)
        means = torch.stack(means)  # [K, B, n]
        across_z = means.var(dim=0, unbiased=False).mean()
        within = torch.stack(variances).mean()
        return across_z / within.clamp(min=1e-12)

    def _route_a_functional_ml(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """A' objective: ConvNP-style Monte-Carlo maximum likelihood
        specialized to FUNCTIONAL targets.

            loss = -log[(1/K) Σ_k p(θ_S | within-draw-aggregate(z_k))],
            z_k ~ p(z|ctx)

        Gradient structure is the whole point: the
        per-unit decodes are aggregated WITHIN each z-draw, the likelihood is
        evaluated on that within-draw aggregate, and draws combine by
        log-mean-exp. z is thereby the only path to getting the aggregate
        right — the pressure the unit-level ELBO cannot supply. In-loss
        functional curriculum: per-task random subset, k ~ log-uniform[1, n],
        θ_S = masked mean of the exact per-unit ITE targets. Within-draw
        aggregate likelihood by moment matching (Gaussian; CLT-exactish for
        moderate k, a smoothing approximation at k=1). No KL term; KL and
        z-spread remain logged as collapse diagnostics.
        """
        bb = self.backbone
        targets = batch["query_value"]  # [B, n] exact abducted per-unit ITEs
        B, n = targets.shape
        device = targets.device

        # per-task k ~ round(log-uniform[1, n]); mask = k random rows
        k = torch.exp(
            torch.rand(B, device=device) * torch.log(torch.tensor(float(n), device=device))
        ).round().clamp(1, n)
        ranks = torch.rand(B, n, device=device).argsort(dim=1).argsort(dim=1)
        mask = (ranks < k.unsqueeze(1)).to(targets.dtype)  # [B, n]
        theta_s = (targets * mask).sum(dim=1) / k  # exact functional target [B]

        h, r_agg = bb.encode(
            batch["x_features"], batch["x_sources"], lengths=batch.get("lengths")
        )
        mu_p, sigma_p = bb.latent_heads.prior(r_agg)

        combined = self.route_a_objective == "combined"
        log_ps, unit_nlls = [], []
        for _ in range(self.ml_train_draws):
            z = bb.latent_heads.sample(mu_p, sigma_p)
            logits = bb.decode_units(h, r_agg, z)
            probs = torch.softmax(logits, dim=-1)
            m, v = histogram_mean_var(probs, self.hist_edges)  # [B, n] each
            mu_s = (m * mask).sum(dim=1) / k
            var_s = (v * mask).sum(dim=1) / k**2 + 1e-6
            log_ps.append(
                torch.distributions.Normal(mu_s, var_s.sqrt()).log_prob(theta_s)
            )
            if combined:
                # unit term: per-DRAW histogram NLL pins each draw's marginals
                # sharp (kills A''s marginal bloat); the functional term below
                # keeps the joint variance across draws (kills A-ELBO's
                # z-starvation). Two terms, one z-inference scheme.
                unit_nlls.append(
                    histogram_loss(
                        logits, targets, self.hist_edges,
                        self.hist_smoothing_sigma_bins,
                    )
                )
        log_mix = torch.stack(log_ps).logsumexp(dim=0) - torch.log(
            torch.tensor(float(self.ml_train_draws), device=device)
        )
        fml_loss = -log_mix.mean()
        if combined:
            unit_loss = torch.stack(unit_nlls).mean()
            loss = unit_loss + self.fml_lambda * fml_loss
            self.log_dict({"train/unit_nll": unit_loss, "train/fml_term": fml_loss})
        else:
            loss = fml_loss

        with torch.no_grad():
            mu_q, sigma_q = bb.latent_heads.posterior(r_agg, targets.unsqueeze(-1))
            kl_dims = bb.latent_heads.kl_divergence(mu_q, sigma_q, mu_p, sigma_p)
        self.log_dict(
            {
                "train/fml_nll": loss,
                "train/fml_k_mean": k.mean(),
                "train/kl": kl_dims.sum(dim=-1).mean(),  # diagnostic only
                "train/active_units": (kl_dims.mean(dim=0) > 0.01).sum().float(),
            }
        )
        return loss

    def _route_c2_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """C v2 multi-task loss: token-wise unit NLL (exact per-unit ITEs)
        + fml_lambda x functional NLL on an in-loss log-uniform-k subset
        (theta_S = masked mean of the exact per-unit targets)."""
        bb = self.backbone
        targets = batch["query_value"]  # [B, n] exact abducted ITEs
        B, n = targets.shape
        device = targets.device

        unit_logits = bb(
            batch["x_features"], batch["x_sources"], lengths=batch.get("lengths")
        )  # [B, n, nbins] (no mask -> per-unit head)
        unit_loss = histogram_loss(
            unit_logits, targets, self.hist_edges, self.hist_smoothing_sigma_bins
        )

        k = torch.exp(
            torch.rand(B, device=device) * torch.log(torch.tensor(float(n), device=device))
        ).round().clamp(1, n)
        ranks = torch.rand(B, n, device=device).argsort(dim=1).argsort(dim=1)
        mask = ranks < k.unsqueeze(1)
        theta_s = (targets * mask).sum(dim=1) / k

        func_logits = bb(
            batch["x_features"], batch["x_sources"],
            lengths=batch.get("lengths"), subset_mask=mask,
        )  # [B, nbins]
        func_loss = histogram_loss(
            func_logits, theta_s, self.hist_edges, self.hist_smoothing_sigma_bins
        )
        lam = float(self.config.get("fml_lambda", 1.0))
        self.log_dict({"train/unit_nll": unit_loss, "train/func_nll": func_loss})
        return unit_loss + lam * func_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        lengths = batch.get("lengths", None)

        if getattr(self.backbone, "variant", "vanilla") == "route_c2":
            return self._route_c2_loss(batch)

        if getattr(self.backbone, "variant", "vanilla") == "route_a":
            if getattr(self, "route_a_objective", "elbo") in ("functional_ml", "combined"):
                return self._route_a_functional_ml(batch)
            return self._route_a_elbo(batch)

        if self.estimator_type == "gaussian":
            if self.backbone.predict_ate:
                mean, var = self(batch["x_features"], batch["x_sources"], lengths=lengths)
                target = batch["query_value"].view(mean.size(0), -1).mean(dim=1)
            else:
                mean, var = self(
                    batch["x_features"],
                    batch["x_sources"],
                    batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"]
            loss = self.gaussian_nll(mean, target, var)

        elif self.estimator_type == "point":
            if self.backbone.predict_ate:
                preds = self(batch["x_features"], batch["x_sources"], lengths=lengths)
                target = batch["query_value"].view(preds.size(0), -1).mean(dim=1)
            else:
                preds = self(
                    batch["x_features"],
                    batch["x_sources"],
                    batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"]
            loss = F.huber_loss(preds, target, delta=1.0)

        elif self.estimator_type == "histogram":
            if self.backbone.predict_ate:
                logits = self(
                    batch["x_features"],
                    batch["x_sources"],
                    lengths=lengths,
                    subset_mask=batch.get("subset_mask"),
                )
                target = batch["query_value"].view(logits.size(0), -1).mean(dim=1)
            else:
                logits = self(
                    batch["x_features"],
                    batch["x_sources"],
                    batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"]
            loss = histogram_loss(
                logits, target, self.hist_edges, self.hist_smoothing_sigma_bins
            )
            self.log("train/hist_clamp_frac", clamp_fraction(target, self.hist_edges))

        elif self.estimator_type == "mixture":
            # Direct-decode arms only (validated): a single functional NLL
            # under a continuous density — no smoothing, no width floor.
            params = self(
                batch["x_features"],
                batch["x_sources"],
                lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            target = batch["query_value"].view(params.size(0), -1).mean(dim=1)
            loss = mdn_nll(params, target, self.mdn_min_sigma)

        elif self.estimator_type == "lotv":
            params = self(
                batch["x_features"],
                batch["x_sources"],
                lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            target = batch["query_value"].view(params.size(0), -1).mean(dim=1)
            loss = lotv_nll(params, target, batch["subset_mask"], self.mdn_min_sigma)

        elif self.estimator_type == "quantile":
            if self.backbone.predict_ate:
                preds, taus = self(batch["x_features"], batch["x_sources"], lengths=lengths)
                target_val = batch["query_value"].view(preds.size(0), -1).mean(dim=1, keepdim=True)
                target = target_val.expand_as(preds)
            else:
                preds, taus = self(
                    batch["x_features"],
                    batch["x_sources"],
                    batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"].unsqueeze(2).expand_as(preds)

            error = target - preds
            huber_loss = F.huber_loss(preds, target, reduction="none", delta=1.0)
            tau_reshaped = (
                taus.squeeze(-1)
                if self.backbone.predict_ate
                else taus.view(batch["x_features"].size(0), 1, self.backbone.num_tau_samples)
            )

            base_loss = (torch.abs(tau_reshaped - (error.detach() < 0).float()) * huber_loss).mean()

            dim_to_sort = 1 if self.backbone.predict_ate else 2
            _, sort_idx = torch.sort(taus.squeeze(-1), dim=1)
            sort_idx_expanded = (
                sort_idx
                if self.backbone.predict_ate
                else sort_idx.unsqueeze(1).expand(-1, preds.size(1), -1)
            )
            sorted_preds = torch.gather(preds, dim=dim_to_sort, index=sort_idx_expanded)

            diffs = (
                sorted_preds[:, 1:] - sorted_preds[:, :-1]
                if self.backbone.predict_ate
                else sorted_preds[:, :, 1:] - sorted_preds[:, :, :-1]
            )
            crossing_penalty = F.relu(-diffs).mean()

            loss = base_loss + (self.lambda_crossing_penalty * crossing_penalty)

            self.log("train/base_loss", base_loss, prog_bar=True)
            self.log("train/crossing_penalty", crossing_penalty, prog_bar=True)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _evaluate_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, stage: str
    ) -> dict[str, torch.Tensor]:
        lengths = batch.get("lengths", None)
        x_features, x_sources = batch["x_features"], batch["x_sources"]
        query_x = batch.get("query_x", None)

        if self.backbone.predict_ate:
            y_true = batch["query_value"].view(x_features.size(0), -1).mean(dim=1)
        else:
            y_true = batch["query_value"]

        if getattr(self.backbone, "variant", "vanilla") == "route_c2":
            probs = torch.softmax(
                self.backbone(x_features, x_sources, lengths=lengths), dim=-1
            )
            lower_tau = (1 - self.confidence_level) / 2
            eval_qs = torch.tensor([lower_tau, 0.50, 1 - lower_tau], device=self.device)
            preds = histogram_quantiles(probs, self.hist_edges, eval_qs)
            pred_lower, pred_median, pred_upper = (
                preds[..., 0], preds[..., 1], preds[..., 2],
            )

        elif getattr(self.backbone, "variant", "vanilla") == "route_a":
            # Unit-level deterministic eval at z = mu_p (prior mean). These
            # metrics are per-unit reconstruction quality — NOT evidence about
            # aggregate functional calibration (that is M6d's claim).
            bb = self.backbone
            h, r_agg = bb.encode(x_features, x_sources, lengths=lengths)
            mu_p, _ = bb.latent_heads.prior(r_agg)
            probs = torch.softmax(bb.decode_units(h, r_agg, mu_p), dim=-1)
            lower_tau = (1 - self.confidence_level) / 2
            eval_qs = torch.tensor([lower_tau, 0.50, 1 - lower_tau], device=self.device)
            preds = histogram_quantiles(probs, self.hist_edges, eval_qs)
            pred_lower, pred_median, pred_upper = (
                preds[..., 0],
                preds[..., 1],
                preds[..., 2],
            )
            if stage == "val" and batch_idx == 0:
                self.log("val/z_spread_ratio", self._z_spread_ratio(batch))

        elif self.estimator_type == "point":
            if self.backbone.predict_ate:
                preds = self(x_features, x_sources, lengths=lengths)
            else:
                preds = self(x_features, x_sources, query_x, lengths=lengths)

            pred_median = preds
            pred_lower = preds
            pred_upper = preds

        elif self.estimator_type == "gaussian":
            if self.backbone.predict_ate:
                mean, var = self(x_features, x_sources, lengths=lengths)
            else:
                mean, var = self(x_features, x_sources, query_x, lengths=lengths)

            std = torch.sqrt(var)
            pred_median = mean
            z = sp.stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            pred_lower = mean - z * std
            pred_upper = mean + z * std

        elif self.estimator_type == "histogram":
            lower_tau = (1 - self.confidence_level) / 2
            eval_qs = torch.tensor(
                [lower_tau, 0.50, 1 - lower_tau], device=self.device
            )
            if self.backbone.predict_ate:
                logits = self(
                    x_features, x_sources, lengths=lengths,
                    subset_mask=batch.get("subset_mask"),
                )
            else:
                logits = self(x_features, x_sources, query_x, lengths=lengths)
            probs = torch.softmax(logits, dim=-1)
            preds = histogram_quantiles(probs, self.hist_edges, eval_qs)
            pred_lower, pred_median, pred_upper = (
                preds[..., 0],
                preds[..., 1],
                preds[..., 2],
            )

        elif self.estimator_type == "mixture":
            params = self(
                x_features, x_sources, lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            lower_tau = (1 - self.confidence_level) / 2
            eval_qs = torch.tensor(
                [lower_tau, 0.50, 1 - lower_tau], device=self.device
            )
            preds = mdn_quantiles(params, eval_qs, self.mdn_min_sigma)
            pred_lower, pred_median, pred_upper = (
                preds[..., 0],
                preds[..., 1],
                preds[..., 2],
            )

        elif self.estimator_type == "lotv":
            params = self(
                x_features, x_sources, lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            mu, sigma, c, v = lotv_split(params, batch["subset_mask"], self.mdn_min_sigma)
            z = sp.stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            pred_median = mu
            pred_lower = mu - z * sigma
            pred_upper = mu + z * sigma
            if stage == "val" and batch_idx == 0:
                # mechanism scalars: the pre-registered additive-vs-hetero check
                self.log("val/lotv_c_mean", c.mean())
                self.log("val/lotv_v_mean", v.mean())

        elif self.estimator_type == "quantile":
            lower_tau = (1 - self.confidence_level) / 2
            upper_tau = 1 - lower_tau
            eval_taus = torch.tensor([lower_tau, 0.50, upper_tau], device=self.device).view(1, 3, 1)
            eval_taus = eval_taus.expand(x_features.size(0), -1, -1)

            if self.backbone.predict_ate:
                preds, _ = self(x_features, x_sources, taus=eval_taus, lengths=lengths)
                pred_lower, pred_median, pred_upper = (
                    preds[:, 0],
                    preds[:, 1],
                    preds[:, 2],
                )
            else:
                preds, _ = self(x_features, x_sources, query_x, taus=eval_taus, lengths=lengths)
                pred_lower, pred_median, pred_upper = (
                    preds[:, :, 0],
                    preds[:, :, 1],
                    preds[:, :, 2],
                )

        if self.conformal_q and self.estimator_type in [
            "quantile",
            "gaussian",
        ]:
            pred_lower = pred_lower - self.conformal_q
            pred_upper = pred_upper + self.conformal_q

            pred_lower = torch.min(pred_lower, pred_median)
            pred_upper = torch.max(pred_upper, pred_median)

        if batch.get("standardized", torch.tensor([False])).any():
            if stage.startswith("test") or (stage == "val" and not self.val_metrics_normalized):
                y_std = batch.get("y_std", torch.ones(x_features.size(0), 1, device=self.device))
                t_std = batch.get("t_std", torch.ones(x_features.size(0), 1, device=self.device))
                if self.backbone.predict_ate:
                    y_std = y_std.squeeze(-1)
                    t_std = t_std.squeeze(-1)

                inverse_scale = y_std / t_std
                y_true = y_true * inverse_scale
                pred_median = pred_median * inverse_scale
                pred_lower = pred_lower * inverse_scale
                pred_upper = pred_upper * inverse_scale

        return {
            "y_true": y_true.detach().cpu(),
            "pred_lower": pred_lower.detach().cpu(),
            "pred_median": pred_median.detach().cpu(),
            "pred_upper": pred_upper.detach().cpu(),
        }

    @torch.no_grad()
    def _nll(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Proper-score validation NLL — the calibration-aware checkpoint
        criterion (gate 2026-07-14). val/rmse is calibration-blind: with the
        mixture head it demonstrably selects along a dispersion-decay
        trajectory. The NLL is the training objective scored on val, so it
        penalizes under- and over-dispersion alike. Defined for the heads
        whose training loss is a per-task density (histogram direct-decode /
        per-unit, mixture); ELBO arms (route_a, route_c2) return None and
        keep their legacy monitor."""
        lengths = batch.get("lengths", None)
        variant = getattr(self.backbone, "variant", "vanilla")
        if variant in ("route_a", "route_c2"):
            return None
        if self.estimator_type == "mixture":
            params = self(
                batch["x_features"], batch["x_sources"], lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            target = batch["query_value"].view(params.size(0), -1).mean(dim=1)
            return mdn_nll(params, target, self.mdn_min_sigma)
        if self.estimator_type == "lotv":
            params = self(
                batch["x_features"], batch["x_sources"], lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            target = batch["query_value"].view(params.size(0), -1).mean(dim=1)
            return lotv_nll(params, target, batch["subset_mask"], self.mdn_min_sigma)
        if self.estimator_type == "histogram":
            if self.backbone.predict_ate:
                logits = self(
                    batch["x_features"], batch["x_sources"], lengths=lengths,
                    subset_mask=batch.get("subset_mask"),
                )
                target = batch["query_value"].view(logits.size(0), -1).mean(dim=1)
            else:
                logits = self(
                    batch["x_features"], batch["x_sources"], batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"]
            return histogram_loss(
                logits, target, self.hist_edges, self.hist_smoothing_sigma_bins
            )
        return None

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self._evaluate_step(batch, batch_idx, "val")
        nll = self._nll(batch)
        if nll is not None:
            self.log("val/nll", nll, on_epoch=True)
        self.validation_step_outputs.append(out)
        return out

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        out = self._evaluate_step(batch, batch_idx, f"test_{dataloader_idx}")

        out_dict = {str(k): v for k, v in out.items()}
        out_dict["dataloader_idx"] = dataloader_idx

        if self.run_baselines and self.baselines:
            warnings.filterwarnings("ignore")

            required_keys = ["x_index", "t_index", "y_index"]
            if not all(k in batch for k in required_keys):
                raise KeyError(f"Missing indices for baselines: {required_keys}")

            x_features_np = batch["x_features"].cpu().numpy()
            x_sources_np = batch["x_sources"].cpu().numpy()
            batch_size = x_features_np.shape[0]

            def _parse_indices(idx_data):
                if (
                    isinstance(idx_data, (list, tuple))
                    and len(idx_data) == 1
                    and isinstance(idx_data[0], torch.Tensor)
                ):
                    return idx_data[0].cpu().tolist()

                elif isinstance(idx_data, torch.Tensor):
                    return idx_data.cpu().tolist()

                elif isinstance(idx_data, (list, tuple)):
                    if not isinstance(idx_data[0], int):
                        raise ValueError(f"Unsupported index type: {type(idx_data[0])}")
                    return list(idx_data)

                return [idx_data] * batch_size

            x_idx_list = _parse_indices(batch["x_index"])
            t_idx_list = _parse_indices(batch["t_index"])
            y_idx_list = _parse_indices(batch["y_index"])
            query_x_idx_list = _parse_indices(batch["query_x_index"])

            query_x_np = None
            if "query_x" in batch and batch["query_x"] is not None:
                query_x_np = batch["query_x"].cpu().numpy()
                if np.isnan(query_x_np).all():
                    query_x_np = None

            batch_y_baselines = {b["name"]: [] for b in self.baselines}

            for i in range(batch_size):
                xi = x_idx_list[i]
                ti = t_idx_list[i]
                yi = y_idx_list[i]
                qxi = query_x_idx_list[i] if query_x_np is not None else None

                x_i = x_features_np[i, ..., xi].T
                t_i = x_features_np[i, ..., ti].T
                y_i = x_features_np[i, ..., yi].T
                qx_i = query_x_np[i, ..., qxi].T if query_x_np is not None else None
                if self.backbone.predict_ate:
                    qx_i = None

                x_sources_i = x_sources_np[i]

                for b in self.baselines:
                    pred = b["model"](x=x_i, t=t_i, y=y_i, query_x=qx_i, x_sources=x_sources_i)
                    batch_y_baselines[b["name"]].append(pred)

            warnings.filterwarnings("default")

            for b in self.baselines:
                y_baseline = np.array(batch_y_baselines[b["name"]])
                if self.backbone.predict_ate:
                    y_baseline = y_baseline.reshape(batch_size)
                else:
                    y_baseline = y_baseline.reshape(batch_size, -1)

                if batch.get("standardized", torch.tensor([False])).any():
                    y_std_np = batch.get("y_std", torch.ones(batch_size, 1)).cpu().numpy()
                    t_std_np = batch.get("t_std", torch.ones(batch_size, 1)).cpu().numpy()
                    if self.backbone.predict_ate:
                        y_std_np = y_std_np.squeeze(-1)
                        t_std_np = t_std_np.squeeze(-1)
                    y_baseline = (y_baseline * y_std_np) / t_std_np

                out_dict[f"y_{b['name']}"] = y_baseline

        self.test_step_outputs.append(out_dict)
        return out_dict

    def _log_epoch_metrics_and_plots(
        self,
        outputs: list[dict[str, Any]],
        stage: str,
        dataloader_idx: int | None = None,
        with_plot: bool = False,
    ) -> None:
        if not outputs:
            return

        if dataloader_idx is not None:
            outputs = [o for o in outputs if o.get("dataloader_idx") == dataloader_idx]
            if not outputs:
                return
            stage_prefix = (
                f"{stage}/empirical"
                if dataloader_idx > 0
                else f"{stage}/simulated"
                if stage.startswith("test")
                else stage
            )
        else:
            stage_prefix = stage

        y_true = torch.cat([x["y_true"].flatten() for x in outputs]).numpy()
        y_lower = torch.cat([x["pred_lower"].flatten() for x in outputs]).numpy()
        y_median = torch.cat([x["pred_median"].flatten() for x in outputs]).numpy()
        y_upper = torch.cat([x["pred_upper"].flatten() for x in outputs]).numpy()

        mae = np.mean(np.abs(y_median - y_true))
        rmse = np.sqrt(np.mean((y_median - y_true) ** 2))
        r2 = r2_score(y_true, y_median)

        metrics = {
            f"{stage_prefix}/mae": mae,
            f"{stage_prefix}/rmse": rmse,
            f"{stage_prefix}/r2": r2,
        }

        if self.estimator_type in ["quantile", "gaussian", "histogram"]:
            coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
            mpiw = np.mean(y_upper - y_lower)
            inversion_rate = np.mean(y_lower > y_upper)

            metrics.update(
                {
                    f"{stage_prefix}/coverage_95": coverage,
                    f"{stage_prefix}/interval_width": mpiw,
                    f"{stage_prefix}/inversion_rate": inversion_rate,
                }
            )

        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics_and_plots(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        dl_indices = set(x.get("dataloader_idx", 0) for x in self.test_step_outputs)
        for idx in dl_indices:
            self._log_epoch_metrics_and_plots(
                self.test_step_outputs, "test", dataloader_idx=idx, with_plot=True
            )

        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # lr_schedule (config) overrides the legacy use_lr_scheduler flag:
        # "cosine" = per-step cosine decay over total_steps (the fairness-pass
        # schedule — both arms must train under the identical decayed schedule
        # before any final A-vs-B number); "none" = constant LR;
        # unset = legacy behavior (use_lr_scheduler -> ReduceLROnPlateau).
        lr_schedule = self.config.get("lr_schedule")
        if lr_schedule == "cosine":
            t_max = int(self.config.get("total_steps") or 10000)
            warmup = int(self.config.get("lr_warmup_steps") or 0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, t_max - warmup), eta_min=self.lr * 0.01
            )
            if warmup > 0:
                # Linear LR warmup into the cosine decay — the standard cure
                # for post-LN attention cold-start collapse (route_c3 E256).
                warm = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warm, scheduler], milestones=[warmup]
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        if lr_schedule == "none":
            return optimizer
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mae",
                    "frequency": 1,
                },
            }
        return optimizer

    @torch.no_grad()
    def calibrate(self, dataloader: Any, alpha: float | None = None) -> None:
        self.eval()
        device = self.device
        scores = []

        if alpha is None:
            alpha = 1 - self.confidence_level

        for batch in dataloader:
            lengths = batch.get("lengths", None)
            x_features = batch["x_features"].to(device)
            x_sources = batch["x_sources"].to(device)
            query_x = batch.get("query_x", None)
            if query_x is not None:
                query_x = query_x.to(device)

            if self.estimator_type == "gaussian":
                mean, var = self(x_features, x_sources, query_x, lengths=lengths)
                std = torch.sqrt(var)
                z_score = sp.stats.norm.ppf(1 - alpha / 2)
                pred_lower = mean - z_score * std
                pred_upper = mean + z_score * std

            elif self.estimator_type == "quantile":
                quantiles = [alpha / 2, 0.5, 1.0 - (alpha / 2)]
                eval_taus = (
                    torch.tensor(quantiles, device=device)
                    .view(1, 3, 1)
                    .expand(x_features.size(0), -1, -1)
                )
                preds, _ = self(x_features, x_sources, query_x, taus=eval_taus, lengths=lengths)

                dim_to_sort = 1 if self.backbone.predict_ate else 2
                preds, _ = torch.sort(preds, dim=dim_to_sort)
                pred_lower, pred_upper = preds[..., 0], preds[..., 2]
            else:
                raise ValueError("Conformal prediction requires an interval estimator")

            if self.backbone.predict_ate:
                y_true = batch["query_value"].view(x_features.size(0), -1).mean(dim=1).to(device)
            else:
                y_true = batch["query_value"].to(device)
                y_true = y_true.view_as(pred_lower)

            batch_scores = torch.max(pred_lower - y_true, y_true - pred_upper)
            scores.append(batch_scores.flatten())

        scores = torch.cat(scores).cpu().numpy()
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)

        self.conformal_q = np.quantile(scores, q_level)

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
        inverse_transform: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        lengths = batch.get("lengths", None)
        x_features, x_sources = batch["x_features"], batch["x_sources"]
        query_x = batch.get("query_x", None)
        B = x_features.size(0)

        if self.backbone.predict_ate:
            y_true = batch["query_value"].view(B, -1).mean(dim=1).cpu().numpy()
        else:
            y_true = batch["query_value"].view(-1).cpu().numpy()

        if getattr(self.backbone, "variant", "vanilla") == "route_a":
            bb = self.backbone
            h, r_agg = bb.encode(x_features, x_sources, lengths=lengths)
            mu_p, _ = bb.latent_heads.prior(r_agg)
            probs = torch.softmax(bb.decode_units(h, r_agg, mu_p), dim=-1)
            lq = (1 - self.confidence_level) / 2
            eval_qs = torch.tensor([lq, 0.5, 1.0 - lq], device=self.device)
            sorted_preds = histogram_quantiles(probs, self.hist_edges, eval_qs)

        elif self.estimator_type == "point":
            point_estimates = self(x_features, x_sources, query_x, lengths=lengths)
            if self.backbone.predict_ate:
                dummy_intervals = point_estimates.unsqueeze(1).expand(-1, 3)
            else:
                dummy_intervals = point_estimates.view(-1).unsqueeze(1).expand(-1, 3)

            if inverse_transform and batch.get("standardized", torch.tensor([False])).any():
                y_std = batch.get("y_std", torch.ones(B, 1, device=self.device))
                t_std = batch.get("t_std", torch.ones(B, 1, device=self.device))
                if self.backbone.predict_ate:
                    y_std = y_std.squeeze(-1)
                    t_std = t_std.squeeze(-1)

                scale_factor = y_std / t_std
                dummy_intervals = (
                    dummy_intervals * scale_factor.unsqueeze(-1)
                    if self.backbone.predict_ate
                    else dummy_intervals * scale_factor
                )
                y_true = y_true * scale_factor.cpu().numpy().squeeze()
            return {"y_pred": dummy_intervals.cpu().numpy().squeeze(), "y_true": y_true}

        elif self.estimator_type == "gaussian":
            if self.backbone.predict_ate:
                mean, var = self(x_features, x_sources, lengths=lengths)
            else:
                mean, var = self(x_features, x_sources, query_x, lengths=lengths)
            std = torch.sqrt(var)
            z_score = sp.stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            lower = mean - z_score * std
            upper = mean + z_score * std
            sorted_preds = torch.stack([lower, mean, upper], dim=-1)

        elif self.estimator_type == "histogram":
            lower_quantile = (1 - self.confidence_level) / 2
            eval_qs = torch.tensor(
                [lower_quantile, 0.5, 1.0 - lower_quantile], device=self.device
            )
            if self.backbone.predict_ate:
                logits = self(
                    x_features, x_sources, lengths=lengths,
                    subset_mask=batch.get("subset_mask"),
                )
            else:
                logits = self(x_features, x_sources, query_x, lengths=lengths)
            probs = torch.softmax(logits, dim=-1)
            sorted_preds = histogram_quantiles(probs, self.hist_edges, eval_qs)

        elif self.estimator_type == "quantile":
            lower_quantile = (1 - self.confidence_level) / 2
            upper_quantile = 1.0 - lower_quantile
            quantiles = [lower_quantile, 0.5, upper_quantile]
            eval_taus = torch.tensor(quantiles, device=self.device).view(1, 3, 1).expand(B, -1, -1)
            preds, taus = self(x_features, x_sources, query_x, taus=eval_taus, lengths=lengths)

            dim_to_sort = 1 if self.backbone.predict_ate else 2
            _, sort_idx = torch.sort(taus.squeeze(-1), dim=1)
            sort_idx_expanded = (
                sort_idx
                if self.backbone.predict_ate
                else sort_idx.unsqueeze(1).expand(-1, preds.size(1), -1)
            )
            sorted_preds = torch.gather(preds, dim=dim_to_sort, index=sort_idx_expanded)

        if self.conformal_q:
            sorted_preds[..., 0] -= self.conformal_q
            sorted_preds[..., 2] += self.conformal_q
            sorted_preds[..., 0] = torch.min(sorted_preds[..., 0], sorted_preds[..., 1])
            sorted_preds[..., 2] = torch.max(sorted_preds[..., 2], sorted_preds[..., 1])

        if inverse_transform and batch.get("standardized", torch.tensor([False])).any():
            y_std = batch.get("y_std", torch.ones(B, 1, device=self.device))
            t_std = batch.get("t_std", torch.ones(B, 1, device=self.device))
            transpose_dim = -1 if self.backbone.predict_ate else 1
            if self.backbone.predict_ate:
                y_std = y_std.squeeze(-1)
                t_std = t_std.squeeze(-1)

            scale_factor = y_std / t_std
            sorted_preds = sorted_preds * scale_factor.unsqueeze(transpose_dim)
            y_true = y_true * scale_factor.cpu().numpy().squeeze()

        if not self.backbone.predict_ate:
            sorted_preds = sorted_preds.view(-1, 3)

        return {"y_pred": sorted_preds.cpu().numpy().squeeze(), "y_true": y_true}


def _validate_variant(variant: str, predict_ate: bool, estimator_type: str) -> None:
    if variant not in ("vanilla", "route_b", "route_a", "route_c", "route_c2", "route_c3"):
        raise ValueError(
            f"variant must be one of vanilla|route_b|route_a|route_c|route_c2|route_c3, got {variant!r}"
        )
    if variant == "route_c2" and not (not predict_ate and estimator_type == "histogram"):
        raise ValueError(
            "route_c2 trains on per-unit data (predict_ate=False) with the "
            "histogram head; the functional head reads the subset mask"
        )
    if variant in ("route_b", "route_c", "route_c3") and not (
        predict_ate and estimator_type in ("histogram", "mixture", "lotv")
    ):
        raise ValueError(
            f"{variant} requires an aggregate decode (predict_ate=True) with "
            "a functional density head (estimator_type='histogram' or "
            "'mixture'); mixture is direct-decode only — route_a's ELBO "
            "still requires the histogram likelihood"
        )
    if variant == "route_a" and not (not predict_ate and estimator_type == "histogram"):
        raise ValueError(
            "route_a is a per-unit decode (predict_ate=False) with the "
            "histogram likelihood head (estimator_type='histogram'); the "
            "subset enters only at inference-time aggregation"
        )


def _masked_mean(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool encoded context [B, n, D] over a boolean mask [B, n]."""
    m = mask.unsqueeze(-1).to(encoded.dtype)
    return (encoded * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _decoder_mlp(in_dim: int, embed_dim: int, out_dim: int, num_layers: int, dropout: float):
    layers = [nn.Linear(in_dim, embed_dim if num_layers > 1 else out_dim)]
    for i in range(num_layers - 1):
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(embed_dim, embed_dim if i < num_layers - 2 else out_dim))
    return nn.Sequential(*layers)


class QCNP(nn.Module):
    def __init__(
        self,
        predict_ate: bool,
        dim_context_features: int,
        dim_query_features: int,
        num_source_types: int = 4,
        source_embed_dim: int = 16,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_tau_samples: int = 8,
        dropout: float = 0.0,
        output_dim: int = 1,
        estimator_type: str = "quantile",
        nbins: int = 64,
        variant: str = "vanilla",
        latent_dim: int = 16,
        c3_summary_concat: bool = False,
        mdn_components: int = 5,
    ) -> None:
        super().__init__()
        if estimator_type not in ("quantile", "histogram", "mixture", "lotv"):
            raise ValueError(
                "QCNP estimator_type must be quantile|histogram|mixture|lotv, "
                f"got {estimator_type!r}"
            )
        _validate_variant(variant, predict_ate, estimator_type)
        self.num_tau_samples = num_tau_samples
        self.predict_ate = predict_ate
        self.estimator_type = estimator_type
        self.nbins = nbins
        self.variant = variant
        self.c3_summary_concat = c3_summary_concat
        self.mdn_components = mdn_components

        self.source_embedding = nn.Embedding(num_source_types, source_embed_dim)

        ctx_layers = [nn.Linear(dim_context_features + source_embed_dim, embed_dim)]
        for _ in range(num_layers - 1):
            if dropout > 0:
                ctx_layers.append(nn.Dropout(dropout))
            ctx_layers.append(nn.ReLU())
            ctx_layers.append(nn.Linear(embed_dim, embed_dim))
        self.context_encoder = nn.Sequential(*ctx_layers)

        if estimator_type in ("histogram", "mixture", "lotv"):
            # Likelihood head (M4): nbins logits over the fixed effect
            # support; no tau conditioning — the histogram is the predictive.
            # Mixture (direct-decode arms only) reuses the same routing with
            # a 3K-parameter continuous head instead of nbins logits.
            if estimator_type == "histogram":
                head_out = nbins
            elif estimator_type == "mixture":
                head_out = 3 * mdn_components
            else:  # lotv: (mu, raw_v, raw_c)
                head_out = 3
            # lotv: +1 decoder input for the N-feature (log(n/500); constant
            # and inert under the fixed-N curriculum, recorded as such)
            n_feat = 1 if estimator_type == "lotv" else 0
            if variant == "route_c2":
                # C v2 (user design): context-aware tokens (full self-attn)
                # + two heads — functional (PMA over refined subset tokens)
                # and per-unit (token-wise ITE histogram). Fixes v1's
                # context starvation.
                self.sa_layers = nn.ModuleList(
                    [MAB(embed_dim, embed_dim, num_heads=4) for _ in range(2)]
                )
                self.subset_pma = PMA(embed_dim, num_heads=4, num_seeds=1)
                self.func_decoder = _decoder_mlp(
                    embed_dim * 2, embed_dim, nbins, num_layers, dropout
                )
                self.unit_head = _decoder_mlp(
                    embed_dim * 2, embed_dim, nbins, num_layers, dropout
                )
            elif variant == "route_a":
                # Route A (M6b): per-unit decode off the CONTEXT encoding h_i
                # (carries y_i — counterfactual-carrier requirement), plus the
                # deterministic r_agg path, plus z projected to embed_dim for
                # capacity parity with the deterministic summaries.
                self.latent_heads = GaussianLatentHeads(
                    embed_dim, latent_dim=latent_dim, target_dim=1
                )
                self.z_proj = nn.Linear(latent_dim, embed_dim)
                self.unit_decoder = _decoder_mlp(
                    embed_dim * 3, embed_dim, nbins, num_layers, dropout
                )
                self.z_injection = (
                    "concat(proj(z)->embed_dim) @ per-unit decoder input "
                    "[h_i, r_agg, proj(z)]"
                )
            elif self.predict_ate:
                # Route B (M5): decode conditions on the subset summary next
                # to the full-context summary — the full dataset informs the
                # mechanism, the subset specifies which functional.
                # Route C (aggregate-token prototype): same as B EXCEPT the
                # subset tokens pass through target-target self-attention
                # (the deliberate M3-invariant relaxation — the only added
                # mechanism) before the subset pool.
                head_in = embed_dim * 2 if variant in ("route_b", "route_c") else embed_dim
                self.ate_decoder = _decoder_mlp(head_in + n_feat, embed_dim, head_out, num_layers, dropout)
                if variant == "route_c":
                    self.tt_attention = nn.ModuleList(
                        [MAB(embed_dim, embed_dim, num_heads=4) for _ in range(2)]
                    )
                    self.subset_pma = PMA(embed_dim, num_heads=4, num_seeds=1)
                if variant == "route_c3":
                    # C v3 (user spec): two-stream token-resolution attention.
                    # Context stream self-attends; query stream (the subset)
                    # self-attends AND cross-attends to the full context —
                    # no pooling until a single readout at the end; no r_agg
                    # shortcut (context reaches the decode ONLY through
                    # cross-attention). Supervision identical to B.
                    self.ctx_sa = nn.ModuleList(
                        [MAB(embed_dim, embed_dim, num_heads=4) for _ in range(2)]
                    )
                    self.q_self = nn.ModuleList(
                        [MAB(embed_dim, embed_dim, num_heads=4) for _ in range(2)]
                    )
                    self.q_cross = nn.ModuleList(
                        [MAB(embed_dim, embed_dim, num_heads=4) for _ in range(2)]
                    )
                    self.readout = PMA(embed_dim, num_heads=4, num_seeds=1)
                    # c3_summary_concat: diagnostic escape from the marginal
                    # basin — decoder also sees [r_agg, r_S] (B's summaries),
                    # so route_c3 becomes B + an attention residual. The pure
                    # form (context only via cross-attention) keeps decoder
                    # input at embed_dim.
                    c3_dec_in = embed_dim * 3 if c3_summary_concat else embed_dim
                    self.func_decoder = _decoder_mlp(
                        c3_dec_in + n_feat, embed_dim, head_out, num_layers, dropout
                    )
            else:
                self.query_encoder = _decoder_mlp(
                    dim_query_features, embed_dim, embed_dim, num_layers, dropout
                )
                self.cate_decoder = _decoder_mlp(
                    embed_dim * 2, embed_dim, nbins, num_layers, dropout
                )
            return

        self.quantile_embedding = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

        if self.predict_ate:
            ate_layers = [nn.Linear(embed_dim * 2, embed_dim if num_layers > 1 else output_dim)]
            for i in range(num_layers - 1):
                if dropout > 0:
                    ate_layers.append(nn.Dropout(dropout))
                ate_layers.append(nn.ReLU())
                ate_layers.append(
                    nn.Linear(embed_dim, embed_dim if i < num_layers - 2 else output_dim)
                )
            self.ate_decoder = nn.Sequential(*ate_layers)
        else:
            query_layers = [nn.Linear(dim_query_features, embed_dim)]
            for _ in range(num_layers - 1):
                if dropout > 0:
                    query_layers.append(nn.Dropout(dropout))
                query_layers.append(nn.ReLU())
                query_layers.append(nn.Linear(embed_dim, embed_dim))
            self.query_encoder = nn.Sequential(*query_layers)

            cate_layers = [nn.Linear(embed_dim * 3, embed_dim if num_layers > 1 else output_dim)]
            for i in range(num_layers - 1):
                if dropout > 0:
                    cate_layers.append(nn.Dropout(dropout))
                cate_layers.append(nn.ReLU())
                cate_layers.append(
                    nn.Linear(embed_dim, embed_dim if i < num_layers - 2 else output_dim)
                )
            self.cate_decoder = nn.Sequential(*cate_layers)

    def encode(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row context encodings h_i [B, n, E] and pooled r_agg [B, E]."""
        device = x_features.device
        source_emb = self.source_embedding(x_sources)
        combined_context = torch.cat([x_features, source_emb], dim=-1)
        encoded_ctx = self.context_encoder(combined_context)

        if lengths is not None:
            max_len = encoded_ctx.size(1)
            valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            masked_ctx = encoded_ctx * valid_mask.unsqueeze(-1)
            r_agg = masked_ctx.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)
        else:
            r_agg = torch.mean(encoded_ctx, dim=1)
        return encoded_ctx, r_agg

    def decode_units(
        self, h: torch.Tensor, r_agg: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Route A per-unit decode: [h_i, r_agg, proj(z)] -> nbins logits."""
        n = h.size(1)
        z_exp = self.z_proj(z).unsqueeze(1).expand(-1, n, -1)
        r_exp = r_agg.unsqueeze(1).expand(-1, n, -1)
        return self.unit_decoder(torch.cat([h, r_exp, z_exp], dim=-1))

    def _n_feature(self, x_features: torch.Tensor) -> torch.Tensor:
        """log(N/500) context-size feature (lotv head): the 'token for N'.
        Constant under the fixed-N=500 curriculum — inert until variable-N
        training exists."""
        import math

        n = x_features.size(1)
        return torch.full(
            (x_features.size(0), 1), math.log(n / 500.0),
            device=x_features.device, dtype=x_features.dtype,
        )

    def forward(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        query_x: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        subset_mask: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        B = x_features.size(0)
        device = x_features.device

        encoded_ctx, r_agg = self.encode(x_features, x_sources, lengths=lengths)

        if self.estimator_type in ("histogram", "mixture", "lotv"):
            if self.variant == "route_c2":
                tokens = encoded_ctx
                for layer in self.sa_layers:
                    tokens = layer(tokens, tokens)  # full self-attention
                r_ctx = tokens.mean(dim=1)
                if subset_mask is not None:
                    # functional head: PMA over refined SUBSET tokens
                    r_s = self.subset_pma(tokens, x_mask=~subset_mask).squeeze(1)
                    return self.func_decoder(torch.cat([r_ctx, r_s], dim=-1))
                # per-unit head over all refined tokens: [B, n_ctx, nbins]
                n = tokens.size(1)
                r_exp = r_ctx.unsqueeze(1).expand(-1, n, -1)
                return self.unit_head(torch.cat([tokens, r_exp], dim=-1))
            if self.variant == "route_a":
                if z is None:
                    mu_p, sigma_p = self.latent_heads.prior(r_agg)
                    z = self.latent_heads.sample(mu_p, sigma_p)
                return self.decode_units(encoded_ctx, r_agg, z)  # [B, n_ctx, nbins]
            if self.predict_ate:
                if self.variant == "route_c3":
                    if subset_mask is None:
                        raise ValueError("route_c3 forward requires subset_mask")
                    ctx = encoded_ctx
                    for layer in self.ctx_sa:
                        ctx = layer(ctx, ctx)  # context self-attention
                    # Query stream runs on the SUBSET rows only, gathered to a
                    # padded [B, k_max, E] tensor. Exactness invariant: subset
                    # rows never read non-subset q-rows (self-attn keys are
                    # subset-only, cross-attn keys are ctx, MAB LN/FFN are
                    # per-token), so dropping the non-subset rows — which the
                    # final PMA masked out anyway — changes nothing.
                    k = subset_mask.sum(dim=1)
                    k_max = max(int(k.max()), 1)
                    order = torch.sort(
                        subset_mask.int(), dim=1, descending=True, stable=True
                    ).indices[:, :k_max]
                    pad = torch.arange(k_max, device=device).unsqueeze(0) >= k.unsqueeze(1)
                    q = torch.gather(ctx, 1, order.unsqueeze(-1).expand(-1, -1, ctx.size(-1)))
                    for sa, ca in zip(self.q_self, self.q_cross, strict=True):
                        q = sa(q, q, y_mask=pad)  # query self-attn (subset keys)
                        q = ca(q, ctx)            # cross-attn to full context
                    r = self.readout(q, x_mask=pad).squeeze(1)  # single end readout
                    if self.c3_summary_concat:
                        r_s = _masked_mean(encoded_ctx, subset_mask)
                        r = torch.cat([r, r_agg, r_s], dim=-1)
                    if self.estimator_type == "lotv":
                        r = torch.cat([r, self._n_feature(x_features)], dim=-1)
                    return self.func_decoder(r)
                if self.variant == "route_b":
                    if subset_mask is None:
                        raise ValueError("route_b forward requires subset_mask")
                    r_s = _masked_mean(encoded_ctx, subset_mask)
                    rb = torch.cat([r_agg, r_s], dim=-1)
                    if self.estimator_type == "lotv":
                        rb = torch.cat([rb, self._n_feature(x_features)], dim=-1)
                    return self.ate_decoder(rb)
                if self.variant == "route_c":
                    if subset_mask is None:
                        raise ValueError("route_c forward requires subset_mask")
                    # target-target self-attention among subset tokens: keys
                    # restricted to the subset; PMA pool over the subset.
                    exclude = ~subset_mask  # True = masked out of attention
                    attended = encoded_ctx
                    for layer in self.tt_attention:
                        attended = layer(attended, attended, y_mask=exclude)
                    r_s = self.subset_pma(attended, x_mask=exclude).squeeze(1)
                    return self.ate_decoder(torch.cat([r_agg, r_s], dim=-1))
                return self.ate_decoder(r_agg)  # [B, nbins]
            q_i = self.query_encoder(query_x)
            r_agg_exp = r_agg.unsqueeze(1).expand(-1, query_x.size(1), -1)
            return self.cate_decoder(torch.cat([r_agg_exp, q_i], dim=-1))  # [B, N_q, nbins]

        if taus is None:
            taus = torch.rand(B, self.num_tau_samples, 1, device=device)
        N_tau = taus.size(1)

        tau_embed = self.quantile_embedding(taus)

        if self.predict_ate:
            r_agg_exp = r_agg.unsqueeze(1).expand(-1, N_tau, -1)
            decoder_input = torch.cat([r_agg_exp, tau_embed], dim=-1)
            return self.ate_decoder(decoder_input).squeeze(-1), taus
        else:
            N_q = query_x.size(1)
            q_i = self.query_encoder(query_x)
            r_agg_exp = r_agg.view(B, 1, 1, -1).expand(-1, N_q, N_tau, -1)
            q_i_exp = q_i.unsqueeze(2).expand(-1, -1, N_tau, -1)
            tau_emb_exp = tau_embed.unsqueeze(1).expand(-1, N_q, -1, -1)

            decoder_input = torch.cat([r_agg_exp, q_i_exp, tau_emb_exp], dim=-1)
            return self.cate_decoder(decoder_input).squeeze(-1), taus


class QTNP(nn.Module):
    def __init__(
        self,
        predict_ate: bool,
        dim_context_features: int,
        dim_query_features: int,
        num_source_types: int = 4,
        source_embed_dim: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing_points: int = 32,
        num_tau_samples: int = 8,
        dropout: float = 0.0,
        estimator_type: str = "quantile",
        nbins: int = 64,
        variant: str = "vanilla",
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        if estimator_type not in ("quantile", "histogram"):
            raise ValueError(
                f"QTNP estimator_type must be quantile|histogram, got {estimator_type!r}"
            )
        _validate_variant(variant, predict_ate, estimator_type)
        self.num_tau_samples = num_tau_samples
        self.predict_ate = predict_ate
        self.estimator_type = estimator_type
        self.nbins = nbins
        self.variant = variant

        self.source_embedding = nn.Embedding(num_source_types, source_embed_dim)
        self.ctx_embedding = nn.Sequential(
            nn.Linear(dim_context_features + source_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.context_encoder = nn.ModuleList(
            [
                ISAB(
                    embed_dim,
                    embed_dim,
                    num_heads,
                    num_inducing_points,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.pma = PMA(embed_dim, num_heads, num_seeds=1, dropout=dropout)

        if estimator_type == "histogram":
            # Likelihood head (M4): nbins logits, no tau conditioning; the
            # query->context cross-attention deterministic path is preserved.
            if variant == "route_a":
                # Route A (M6b): per-unit decode off ISAB-mixed h_i (carries
                # y_i and cross-row context), r_agg, and capacity-matched z.
                self.latent_heads = GaussianLatentHeads(
                    embed_dim, latent_dim=latent_dim, target_dim=1
                )
                self.z_proj = nn.Linear(latent_dim, embed_dim)
                self.unit_decoder = nn.Sequential(
                    nn.Linear(embed_dim * 3, embed_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(embed_dim, nbins),
                )
                self.z_injection = (
                    "concat(proj(z)->embed_dim) @ per-unit decoder input "
                    "[h_i(ISAB), r_agg, proj(z)]"
                )
            elif self.predict_ate:
                head_in = embed_dim * 2 if variant == "route_b" else embed_dim
                self.ate_decoder = nn.Sequential(
                    nn.Linear(head_in, embed_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(embed_dim, nbins),
                )
            else:
                self.query_embedding = nn.Sequential(
                    nn.Linear(dim_query_features, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.cate_decoder = nn.Sequential(
                    nn.Linear(embed_dim * 3, embed_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(embed_dim, nbins),
                )
            return

        self.quantile_embedding = CosineEmbeddingNetwork(embedding_dim=embed_dim)

        if self.predict_ate:
            self.ate_decoder = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.query_embedding = nn.Sequential(
                nn.Linear(dim_query_features, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cate_decoder = nn.Sequential(
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )

    def _encode_full(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        device = x_features.device
        max_len = x_features.size(1)

        padding_mask = None
        if lengths is not None:
            seq_range = torch.arange(max_len, device=device).unsqueeze(0)
            padding_mask = seq_range >= lengths.unsqueeze(1)

        source_emb = self.source_embedding(x_sources)
        combined_context = torch.cat([x_features, source_emb], dim=-1)
        ctx_mapped = self.ctx_embedding(combined_context)

        enc_out = ctx_mapped
        for layer in self.context_encoder:
            enc_out = layer(enc_out, x_mask=padding_mask)

        r_agg = self.pma(enc_out, x_mask=padding_mask).squeeze(1)
        return ctx_mapped, enc_out, r_agg, padding_mask

    def encode(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row (ISAB-mixed) encodings h_i [B, n, E] and r_agg [B, E]."""
        _, enc_out, r_agg, _ = self._encode_full(x_features, x_sources, lengths)
        return enc_out, r_agg

    def decode_units(
        self, h: torch.Tensor, r_agg: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Route A per-unit decode: [h_i, r_agg, proj(z)] -> nbins logits."""
        n = h.size(1)
        z_exp = self.z_proj(z).unsqueeze(1).expand(-1, n, -1)
        r_exp = r_agg.unsqueeze(1).expand(-1, n, -1)
        return self.unit_decoder(torch.cat([h, r_exp, z_exp], dim=-1))

    def forward(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        query_x: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        subset_mask: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, max_len, _ = x_features.shape
        device = x_features.device

        ctx_mapped, enc_out, r_agg, padding_mask = self._encode_full(
            x_features, x_sources, lengths=lengths
        )

        if self.estimator_type == "histogram":
            if self.variant == "route_a":
                if z is None:
                    mu_p, sigma_p = self.latent_heads.prior(r_agg)
                    z = self.latent_heads.sample(mu_p, sigma_p)
                return self.decode_units(enc_out, r_agg, z)  # [B, n_ctx, nbins]
            if self.predict_ate:
                if self.variant == "route_b":
                    if subset_mask is None:
                        raise ValueError("route_b forward requires subset_mask")
                    # PMA x_mask excludes True positions: exclude non-subset
                    # rows (and padding) so the pool attends only to S.
                    subset_exclude = ~subset_mask
                    if padding_mask is not None:
                        subset_exclude = subset_exclude | padding_mask
                    r_s = self.pma(enc_out, x_mask=subset_exclude).squeeze(1)
                    return self.ate_decoder(torch.cat([r_agg, r_s], dim=-1))
                return self.ate_decoder(r_agg)  # [B, nbins]
            queries = self.query_embedding(query_x)
            query_features, _ = self.cross_attention(
                query=queries, key=ctx_mapped, value=ctx_mapped,
                key_padding_mask=padding_mask,
            )
            N_q = query_x.size(1)
            r_agg_exp = r_agg.unsqueeze(1).expand(-1, N_q, -1)
            decoder_input = torch.cat([r_agg_exp, queries, query_features], dim=-1)
            return self.cate_decoder(decoder_input)  # [B, N_q, nbins]

        if taus is None:
            taus = torch.rand(B, self.num_tau_samples, 1, device=device)
        N_tau = taus.size(1)
        tau_embed = self.quantile_embedding(taus)

        if self.predict_ate:
            r_agg_exp = r_agg.unsqueeze(1).expand(-1, N_tau, -1)
            decoder_input = torch.cat([r_agg_exp, tau_embed], dim=-1)
            preds = self.ate_decoder(decoder_input).squeeze(-1)
            return preds, taus
        else:
            N_q = query_x.size(1)
            queries = self.query_embedding(query_x)

            query_features, _ = self.cross_attention(
                query=queries,
                key=ctx_mapped,
                value=ctx_mapped,
                key_padding_mask=padding_mask,
            )

            r_agg_exp = r_agg.view(B, 1, 1, -1).expand(-1, N_q, N_tau, -1)
            q_exp = query_features.unsqueeze(2).expand(-1, -1, N_tau, -1)
            tau_exp = tau_embed.unsqueeze(1).expand(-1, N_q, -1, -1)

            query_orig_exp = queries.unsqueeze(2).expand(-1, -1, N_tau, -1)
            decoder_input = torch.cat([r_agg_exp, query_orig_exp, q_exp, tau_exp], dim=-1)
            preds = self.cate_decoder(decoder_input).squeeze(-1)
            return preds, taus


def build_backbone(
    config: dict[str, Any],
    backbone_type: str,
    predict_ate: bool,
    dim_context_features: int,
    dim_query_features: int,
    num_source_types: int,
) -> nn.Module:
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    num_tau_samples = config["num_tau_samples"]
    source_embed_dim = config["source_embed_dim"]
    output_dim = config["output_dim"]
    num_inducing_points = config["num_inducing_points"]
    dropout = config["dropout"]
    estimator_type = config.get("estimator_type", "quantile")
    nbins = config.get("hist_nbins", 64)
    variant = config.get("model_variant", "vanilla")
    latent_dim = config.get("latent_dim", 16)
    c3_summary_concat = bool(config.get("c3_summary_concat", False))
    mdn_components = int(config.get("mdn_components", 5))

    if backbone_type == "q-cnp":
        return QCNP(
            predict_ate=predict_ate,
            dim_context_features=dim_context_features,
            dim_query_features=dim_query_features,
            num_source_types=num_source_types,
            source_embed_dim=source_embed_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_tau_samples=num_tau_samples,
            dropout=dropout,
            output_dim=output_dim,
            estimator_type=estimator_type,
            nbins=nbins,
            variant=variant,
            latent_dim=latent_dim,
            c3_summary_concat=c3_summary_concat,
            mdn_components=mdn_components,
        )

    elif backbone_type == "q-tnp":
        return QTNP(
            predict_ate=predict_ate,
            dim_context_features=dim_context_features,
            dim_query_features=dim_query_features,
            num_source_types=num_source_types,
            source_embed_dim=source_embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_inducing_points=num_inducing_points,
            num_tau_samples=num_tau_samples,
            dropout=dropout,
            estimator_type=estimator_type,
            nbins=nbins,
            variant=variant,
            latent_dim=latent_dim,
        )

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


BACKBONE_REGISTRY = {
    "q-cnp": QCNP,
    "q-tnp": QTNP,
}
