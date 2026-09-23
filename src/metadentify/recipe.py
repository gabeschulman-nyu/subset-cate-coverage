"""The frozen training recipe, in one executable place.

Every claim-carrying cell in the paper was trained under one recipe, but that
recipe only existed as shell history. Reconstructing it by hand for a new arm
went wrong three times in a row, each time silently:

  * ``--patience`` defaults to 30, while every prior cell used 100000. A new
    run early-stopped at step 5120 of 30000 and reported COMPLETED.
  * ``total_steps`` only binds when ``--online_data`` is set; otherwise
    ``max_epochs=100`` binds instead. Without it a run trained ~12.8k offline
    steps against the recipe's 30k online ones.
  * ``--deterministic_datagen`` governs whether tasks are regenerated per
    epoch, so omitting it changes the effective training distribution.

None of these raise. They produce a checkpoint that looks fine and is not
comparable to the others, which is the worst failure mode for a head or arm
comparison. FROZEN is the single source of truth; ``assert_matches`` is the
guard that turns a silent deviation into an error.
"""

from __future__ import annotations

# Values shared by every claim-carrying cell. Read off the knob-battery
# checkpoints (sweep_kn_*), which define the recipe.
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
    # The three that bit us. patience effectively disables early stopping;
    # online_data is what makes total_steps bind instead of max_epochs;
    # deterministic_datagen fixes the per-epoch task stream.
    "patience": 100000,
    "online_data": True,
    "deterministic_datagen": True,
    "num_train_tasks": 8192,
    "num_val_tasks": 256,
    "num_test_tasks": 128,
    "num_query_points": 10,
    "tasks_per_file": 500,
}

# Set per cell rather than fixed by the recipe.
FREE = frozenset({
    "model_variant", "estimator_type", "mdn_components", "seed",
    "checkpoint_dir", "ckpt_monitor", "effect_scale", "overlap_floor",
    "overlap_sharpness", "unobs_conf_coef", "unobs_noise_coef",
    "covariate_corr", "outcome_noise_scale", "disable_wandb",
    # Structural per arm, not a free choice: route_a trains on per-unit ITE
    # targets and train.py rejects a GATE query for it, so the subset-decode
    # arms and the latent arm necessarily differ here.
    "query_name",
    # n is a deliberate design dimension, not a recipe constant. It is
    # recorded as "n" in every result artifact, so a mismatch between cells
    # shows up in the data rather than silently; figures that compare across
    # cells should assert on it themselves.
    "dataset_size",
})


def cli_args(**overrides: object) -> list[str]:
    """Recipe as a CLI argument list. Store-true flags are emitted bare, and
    None-valued keys are dropped so the parser default applies."""
    cfg = dict(FROZEN)
    cfg.update(overrides)
    out: list[str] = []
    for key, val in cfg.items():
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                out.append(f"--{key}")
            continue
        out += [f"--{key}", str(val)]
    return out


def assert_matches(config: dict, *, ignore: set[str] | None = None) -> None:
    """Raise if a trained config departs from the recipe on any pinned key.

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
