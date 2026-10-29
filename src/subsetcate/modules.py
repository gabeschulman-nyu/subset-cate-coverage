from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from subsetcate.histogram import (
    bin_edges,
    clamp_fraction,
    histogram_loss,
    histogram_mean_var,
    histogram_quantiles,
)
from subsetcate.latent import GaussianLatentHeads
from subsetcate.mdn import mdn_nll, mdn_quantiles


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

        if self.estimator_type == "mixture":
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

        if self.run_baselines:
            raise ValueError("classical baselines are not part of this release")

        self.validation_step_outputs: list[dict[str, Any]] = []
        self.test_step_outputs: list[dict[str, Any]] = []


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


    @torch.no_grad()
    def _z_spread_ratio(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Variance of decoded per-unit means across K prior draws of z, over
        the mean within-histogram variance. Near zero means z is ignored."""
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
        """Latent NP loss, a Monte Carlo likelihood of the subset-CATE (Appendix B).

            loss = -log[(1/K) sum_k p(theta_S | z_k)],  z_k ~ p(z | ctx)

        Within each draw the per-unit histograms over S are moment-matched to
        a Gaussian (mean sum_i m_i / k, variance sum_i v_i / k^2), so units are
        independent given z and their error correlation comes from z alone.
        One subset per task, k ~ round(log-uniform[1, n]). 'combined' adds the
        per-draw unit histogram NLL and weights the subset term by fml_lambda.
        No KL term; KL is logged as a diagnostic.
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
                # Per-draw unit NLL sharpens each draw's per-unit marginals;
                # the subset term fits the variance across draws.
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


    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        lengths = batch.get("lengths", None)

        if getattr(self.backbone, "variant", "vanilla") == "route_a":
            if getattr(self, "route_a_objective", "elbo") in ("functional_ml", "combined"):
                return self._route_a_functional_ml(batch)
            raise ValueError("route_a_objective must be 'combined' or 'functional_ml'")

        if self.estimator_type == "histogram":
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
            # Direct-decode CNP: NLL of the subset-CATE under the mixture.
            params = self(
                batch["x_features"],
                batch["x_sources"],
                lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            target = batch["query_value"].view(params.size(0), -1).mean(dim=1)
            loss = mdn_nll(params, target, self.mdn_min_sigma)

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

        if getattr(self.backbone, "variant", "vanilla") == "route_a":
            # Per-unit predictions at the prior mean z = mu_p.
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
        """Training loss on a validation batch, for checkpoint selection
        (penalizes under- and over-dispersion; None for route_a)."""
        lengths = batch.get("lengths", None)
        variant = getattr(self.backbone, "variant", "vanilla")
        if variant == "route_a":
            return None
        if self.estimator_type == "mixture":
            params = self(
                batch["x_features"], batch["x_sources"], lengths=lengths,
                subset_mask=batch.get("subset_mask"),
            )
            target = batch["query_value"].view(params.size(0), -1).mean(dim=1)
            return mdn_nll(params, target, self.mdn_min_sigma)
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

        if self.estimator_type == "histogram":
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
        # lr_schedule overrides use_lr_scheduler. "cosine" = per-step cosine
        # decay over total_steps; "none" = constant LR; unset = ReduceLROnPlateau
        # if use_lr_scheduler.
        lr_schedule = self.config.get("lr_schedule")
        if lr_schedule == "cosine":
            t_max = int(self.config.get("total_steps") or 10000)
            warmup = int(self.config.get("lr_warmup_steps") or 0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, t_max - warmup), eta_min=self.lr * 0.01
            )
            if warmup > 0:
                # Linear warmup into the cosine decay (0 steps for all trained models).
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


def _validate_variant(variant: str, predict_ate: bool, estimator_type: str) -> None:
    if variant not in ("vanilla", "route_b", "route_a"):
        raise ValueError(f"variant must be one of vanilla|route_b|route_a, got {variant!r}")
    if variant == "route_b" and not (predict_ate and estimator_type in ("histogram", "mixture")):
        raise ValueError(
            "route_b requires an aggregate decode (predict_ate=True) with a "
            "functional density head (estimator_type='histogram' or 'mixture')"
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
        if estimator_type not in ("histogram", "mixture"):
            raise ValueError(
                "QCNP estimator_type must be histogram|mixture, "
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

        if estimator_type in ("histogram", "mixture"):
            # Histogram head: nbins logits over a fixed effect support.
            # Mixture head: 3K parameters (pi logits, mu, raw sigma).
            if estimator_type == "histogram":
                head_out = nbins
            else:  # mixture
                head_out = 3 * mdn_components
            if variant == "route_a":
                # route_a (latent NP): per-unit decode from [h_i, r_agg, proj(z)].
                # h_i encodes row i including y_i, on which the abducted ITE
                # depends. z is projected to embed_dim to match h_i and r_agg.
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
                # route_b (CNP): decode from [r_agg, r_S], the mean encodings
                # of the full context and of the subset.
                head_in = embed_dim * 2 if variant == "route_b" else embed_dim
                self.ate_decoder = _decoder_mlp(head_in, embed_dim, head_out, num_layers, dropout)
            else:
                self.query_encoder = _decoder_mlp(
                    dim_query_features, embed_dim, embed_dim, num_layers, dropout
                )
                self.cate_decoder = _decoder_mlp(
                    embed_dim * 2, embed_dim, nbins, num_layers, dropout
                )
            return


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
        """route_a per-unit decode: [h_i, r_agg, proj(z)] -> nbins logits."""
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
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:

        encoded_ctx, r_agg = self.encode(x_features, x_sources, lengths=lengths)

        if self.estimator_type in ("histogram", "mixture"):
            if self.variant == "route_a":
                if z is None:
                    mu_p, sigma_p = self.latent_heads.prior(r_agg)
                    z = self.latent_heads.sample(mu_p, sigma_p)
                return self.decode_units(encoded_ctx, r_agg, z)  # [B, n_ctx, nbins]
            if self.predict_ate:
                if self.variant == "route_b":
                    if subset_mask is None:
                        raise ValueError("route_b forward requires subset_mask")
                    r_s = _masked_mean(encoded_ctx, subset_mask)
                    rb = torch.cat([r_agg, r_s], dim=-1)
                    return self.ate_decoder(rb)
                return self.ate_decoder(r_agg)  # [B, nbins]
            q_i = self.query_encoder(query_x)
            r_agg_exp = r_agg.unsqueeze(1).expand(-1, query_x.size(1), -1)
            return self.cate_decoder(torch.cat([r_agg_exp, q_i], dim=-1))  # [B, N_q, nbins]


def build_backbone(
    config: dict[str, Any],
    backbone_type: str,
    predict_ate: bool,
    dim_context_features: int,
    dim_query_features: int,
    num_source_types: int,
) -> nn.Module:
    embed_dim = config["embed_dim"]
    num_layers = config["num_layers"]
    num_tau_samples = config["num_tau_samples"]
    source_embed_dim = config["source_embed_dim"]
    output_dim = config["output_dim"]
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

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


BACKBONE_REGISTRY = {
    "q-cnp": QCNP,
}
