"""The training recipe shared by the trained models.

FROZEN holds the settings every trained model used. Some differ from the
parser defaults (patience, online_data, deterministic_datagen), and a run
that omits them trains without error but is not comparable to the others.
``assert_matches`` raises on any such departure.
"""

from __future__ import annotations

FROZEN: dict[str, object] = {
    "experiment_setup_path": "experiments/exact_polynomial.py",
    "experiment_name": "polynomial-confounder",
    "query_name": "GATE",
    "backbone_type": "q-cnp",
    "embed_dim": 128,
    "source_embed_dim": 16,
    "dataset_size": 500,
    "gate_k_min": 1,
    "gate_k_max": None,
    "batch_size": 64,
    "lr": 3e-4,
    "lr_schedule": "cosine",
    "lr_warmup_steps": 0,
    "use_lr_scheduler": True,
    "total_steps": 30000,
    # patience=100000 disables early stopping; online_data makes total_steps
    # set the run length; deterministic_datagen fixes the task stream.
    "patience": 100000,
    "online_data": True,
    "deterministic_datagen": True,
    "num_train_tasks": 8192,
    "num_val_tasks": 256,
    "num_test_tasks": 128,
    "num_query_points": 10,
    "tasks_per_file": 500,
}

# Set per model or per experiment.
FREE = frozenset({
    "model_variant", "estimator_type", "mdn_components", "seed",
    "checkpoint_dir", "ckpt_monitor", "effect_scale", "overlap_floor",
    "overlap_sharpness", "unobs_conf_coef", "unobs_noise_coef",
    "covariate_corr", "outcome_noise_scale",
    # GATE for the CNP; ITE for the latent NP (train.py rejects GATE for it).
    "query_name",
    # n varies by experiment and is recorded as "n" in every result file.
    "dataset_size",
})


def assert_matches(config: dict, *, ignore: set[str] | None = None) -> None:
    """Raise if a checkpoint config differs from FROZEN on any key not in FREE.

    Call on a loaded checkpoint before using it in a comparison.
    """
    ignore = ignore or set()
    bad = []
    for key, want in FROZEN.items():
        if key in FREE or key in ignore or key == "experiment_setup_path":
            continue
        got = config.get(key)
        if got != want:
            bad.append(f"{key}: recipe={want!r} checkpoint={got!r}")
    if bad:
        raise ValueError(
            "checkpoint departs from the frozen recipe:\n  " + "\n  ".join(bad)
        )
