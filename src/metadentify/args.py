import argparse


def get_base_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_setup_path", type=str, default="")
    parser.add_argument("--config_path", type=str, default="config.py")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--savelast", action="store_true")
    parser.add_argument("--overwrite_data", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable_wandb", action="store_true")

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--query_name", type=str, default="PATE")
    parser.add_argument("--subset_backend", type=str, default="uniform_random")
    parser.add_argument("--gate_k_min", type=int, default=1)
    parser.add_argument("--gate_k_max", type=int, default=None)
    parser.add_argument("--baseline_setting", type=str, default=None)

    parser.add_argument("--backbone_type", type=str, default="q-cnp")
    parser.add_argument("--model_variant", type=str, default="vanilla")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument(
        "--use_lr_scheduler", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--lr_schedule", type=str, default=None,
                        choices=["cosine", "none"])
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--route_a_objective", type=str, default="elbo",
                        choices=["elbo", "functional_ml", "combined"])
    parser.add_argument("--ml_train_draws", type=int, default=16)
    parser.add_argument("--fml_lambda", type=float, default=1.0)
    parser.add_argument(
        "--c3_summary_concat", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--kl_free_bits", type=float, default=0.05)
    parser.add_argument("--kl_warmup_steps", type=int, default=1000)
    parser.add_argument("--z_spread_draws", type=int, default=8)
    parser.add_argument("--estimator_type", type=str, default="quantile")
    parser.add_argument("--hist_nbins", type=int, default=64)
    parser.add_argument("--hist_support_min", type=float, default=-4.0)
    parser.add_argument("--hist_support_max", type=float, default=4.0)
    parser.add_argument("--hist_smoothing_sigma_bins", type=float, default=1.0)
    parser.add_argument("--mdn_components", type=int, default=5)
    parser.add_argument("--mdn_min_sigma", type=float, default=1e-3)
    parser.add_argument("--ckpt_monitor", type=str, default=None)
    # Post-hoc conformal rescale for gaussian/quantile heads. On by default to
    # preserve prior behaviour; turn off for like-for-like head comparisons,
    # since no density head receives it.
    parser.add_argument("--post_hoc_calibrate", type=lambda v: str(v).lower()
                        not in ("0", "false", "no"), default=True)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_tau_samples", type=int, default=8)
    parser.add_argument("--source_embed_dim", type=int, default=16)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--num_inducing_points", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--standardize", action="store_true")
    parser.add_argument(
        "--deterministic_datagen", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--num_pytorch_workers", type=int, default=0)
    parser.add_argument("--num_datagen_workers", type=int, default=-2)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--num_query_points", type=int, default=10)
    parser.add_argument("--num_scms", type=int, default=None)
    parser.add_argument("--num_train_tasks", type=int, default=10000)
    parser.add_argument("--num_val_tasks", type=int, default=1000)
    parser.add_argument("--num_test_tasks", type=int, default=1000)
    parser.add_argument("--tasks_per_file", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--online_data", action="store_true")
    parser.add_argument("--total_steps", type=int, default=None)

    parser.add_argument(
        "--val_metrics_normalized", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_crossing_penalty", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--plot_diagnostics", action="store_true")

    return parser
