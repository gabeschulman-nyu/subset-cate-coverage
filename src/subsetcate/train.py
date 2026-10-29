import hashlib
import json
import os
from pathlib import Path
from typing import Any

import filelock
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from subsetcate.mixture import (
    OfflineCausalDataGenerator,
    OnlineCausalDataModule,
)
from subsetcate.modules import CausalMetaModel, build_backbone
from subsetcate.utils import get_available_cpus


def run_meta_training_pipeline(
    config: dict[str, Any],
    setup_dict: dict[str, Any],
    logger: Any | None = None,
    run_id: str = "default_run",
) -> tuple[CausalMetaModel, pl.LightningDataModule, pl.Trainer]:
    exp_name = config["experiment_name"]
    query_name = config["query_name"]
    backbone_type = config["backbone_type"]
    baseline_setting = config["baseline_setting"]
    if baseline_setting is None:
        baseline_setting = exp_name

    online_data = config["online_data"]

    available_cpus = get_available_cpus()

    cfg_pytorch_workers = config["num_pytorch_workers"]
    cfg_datagen_workers = config["num_datagen_workers"]

    if cfg_pytorch_workers < 0:
        num_pytorch_workers = max(0, available_cpus + cfg_pytorch_workers)
    else:
        num_pytorch_workers = min(cfg_pytorch_workers, available_cpus)

    if cfg_datagen_workers < 0:
        num_datagen_workers = max(1, available_cpus + cfg_datagen_workers)
    else:
        num_datagen_workers = min(cfg_datagen_workers, available_cpus)

    if num_datagen_workers == 0 and config["num_datagen_workers"] != 0:
        num_datagen_workers = 1

    query_fn = setup_dict["queries"][query_name]
    observed_covariates = setup_dict["observed_covariates"]
    outcome_name = setup_dict["outcome_name"]
    treatment_name = setup_dict["treatment_name"]
    dataset_size = config["dataset_size"]

    predict_ate = query_name in ["PATE", "STATE", "GATE"]
    dim_context_features = len(observed_covariates)

    if config.get("model_variant") == "route_a":
        # route_a trains on the exact abducted ITE of every context row;
        # subsets enter only at inference. CATE targets match ITEs only for
        # additive families, so the target must be the abduction primitive.
        from subsetcate.exact import require_counterfactual_target

        require_counterfactual_target(query_fn)
        if predict_ate:
            raise ValueError(
                "route_a is per-unit; use a per-unit query (e.g. ITE), "
                f"not {query_name}"
            )
        query_x_type = "context-ite"
        dim_query_features = 1
        query_feature_names = None
    elif predict_ate:
        if query_name == "STATE":
            query_x_type = "full-context-dummy"
        elif query_name == "GATE":
            query_x_type = "subset-gate"
        else:
            query_x_type = "none"
        dim_query_features = 1
        query_feature_names = None
    else:
        query_x_type = "sample-from-context"
        dim_query_features = len(observed_covariates)
        query_feature_names = observed_covariates

    subset_sampler = None
    if query_name == "GATE":
        from subsetcate.subsets import make_subset_sampler

        subset_sampler = make_subset_sampler(
            config["subset_backend"],
            training=True,
            k_min=config["gate_k_min"],
            k_max=config["gate_k_max"],
        )

    data_keys = [
        "dataset_size",
        "num_query_points",
        "num_train_tasks",
        "num_val_tasks",
        "num_test_tasks",
        "standardize",
        "num_scms",
        "experiment_name",
        "query_name",
    ]

    data_keys.extend(setup_dict.get("data_generation_keys", []))

    if query_name == "GATE":
        data_keys.extend(["subset_backend", "gate_k_min", "gate_k_max"])

    if config.get("model_variant") == "route_a":
        # context-ite data differs from the same query_name's default type
        data_keys.append("model_variant")

    if config.get("deterministic_datagen"):
        # Cached val/test data is keyed by seed, so models trained with the
        # same seed are evaluated on the same tasks.
        data_keys.extend(["deterministic_datagen", "seed"])

    data_params = {k: config[k] for k in data_keys if k in config}
    params_str = json.dumps(data_params, sort_keys=True)
    data_hash = hashlib.sha256(params_str.encode("utf-8")).hexdigest()[:8]

    save_dir = config["save_dir"]
    base_data_dir = Path(
        config.get(
            "base_data_dir",
            f"{save_dir}/{exp_name}/{query_name.lower()}/size_{dataset_size}_{data_hash}",
        )
    )

    custom_datamodule = setup_dict.get("custom_datamodule")

    if custom_datamodule is not None:
        datamodule = custom_datamodule
    else:
        base_data_dir.mkdir(parents=True, exist_ok=True)

        lock_file_path = base_data_dir / ".generation_lock"
        lock = filelock.FileLock(lock_file_path)

        with lock:
            splits_to_check = ["val", "test"] if online_data else ["train", "val", "test"]
            all_complete = all(
                (base_data_dir / split / "_GENERATION_COMPLETE").exists()
                for split in splits_to_check
            )

            if not all_complete or config["overwrite_data"]:
                generator = OfflineCausalDataGenerator(
                    causal_model_prior_fn=setup_dict["causal_model_prior_fn"],
                    mixture_builder_fn=setup_dict["mixture_builder_fn"],
                    causal_query_fn=query_fn,
                    num_context_points=dataset_size,
                    num_query_points=config["num_query_points"],
                    outcome_name=outcome_name,
                    treatment_name=treatment_name,
                    observed_feature_names=observed_covariates,
                    query_feature_names=query_feature_names,
                    query_x_type=query_x_type,
                    standardize=config["standardize"],
                    subset_sampler=subset_sampler,
                    deterministic_seed=config["seed"]
                    if config.get("deterministic_datagen")
                    else None,
                )

                tasks_per_file = config["tasks_per_file"]
                num_train_tasks = config["num_train_tasks"]
                num_val_tasks = config["num_val_tasks"]
                num_test_tasks = config["num_test_tasks"]

                if not online_data:
                    generator.generate_and_save(
                        total_tasks=num_train_tasks,
                        tasks_per_file=min(num_train_tasks, tasks_per_file),
                        save_dir=base_data_dir / "train",
                        n_workers=num_datagen_workers,
                    )
                    (base_data_dir / "train" / "_GENERATION_COMPLETE").touch()

                generator.generate_and_save(
                    total_tasks=num_val_tasks,
                    tasks_per_file=min(num_val_tasks, tasks_per_file),
                    save_dir=base_data_dir / "val",
                    n_workers=num_datagen_workers,
                )
                (base_data_dir / "val" / "_GENERATION_COMPLETE").touch()

                generator.generate_and_save(
                    total_tasks=num_test_tasks,
                    tasks_per_file=min(num_test_tasks, tasks_per_file),
                    save_dir=base_data_dir / "test",
                    n_workers=num_datagen_workers,
                )
                (base_data_dir / "test" / "_GENERATION_COMPLETE").touch()

        empirical_parser_fns = setup_dict.get("empirical_parser_fns", {})
        empirical_parser_fn = empirical_parser_fns.get(query_name)

        if online_data:
            datamodule = OnlineCausalDataModule(
                causal_model_prior_fn=setup_dict["causal_model_prior_fn"],
                mixture_builder_fn=setup_dict["mixture_builder_fn"],
                causal_query_fn=query_fn,
                num_context_points=dataset_size,
                num_query_points=config["num_query_points"],
                outcome_name=outcome_name,
                treatment_name=treatment_name,
                observed_feature_names=observed_covariates,
                query_feature_names=query_feature_names,
                query_x_type=query_x_type,
                standardize=config["standardize"],
                subset_sampler=subset_sampler,
                deterministic=bool(config.get("deterministic_datagen")),
                batch_size=config["batch_size"],
                num_workers=num_pytorch_workers,
                prefetch_factor=config["prefetch_factor"],
                data_dir=base_data_dir,
                empirical_data_dir=setup_dict["empirical_data_dir"]
                if empirical_parser_fn
                else None,
                empirical_parser_fn=empirical_parser_fn,
                empirical_max_len=setup_dict["empirical_max_len"],
            )
        else:
            raise ValueError("this release trains with --online_data only")

    backbone = build_backbone(
        config=config,
        backbone_type=backbone_type,
        predict_ate=predict_ate,
        dim_context_features=dim_context_features,
        dim_query_features=dim_query_features,
        num_source_types=len(setup_dict.get("source_names", ["Observational"])),
    )

    config["empirical_data_dir"] = setup_dict.get("empirical_data_dir", "N/A")
    config["empirical_max_len"] = setup_dict.get("empirical_max_len", "N/A")

    custom_baselines = setup_dict.get("custom_baselines")

    model = CausalMetaModel(
        backbone=backbone,
        lr=config["lr"],
        lambda_crossing_penalty=config["lambda_crossing_penalty"],
        weight_decay=config["weight_decay"],
        use_lr_scheduler=config.get("use_lr_scheduler", True),
        run_baselines=config["run_baselines"],
        baseline_setting=baseline_setting,
        val_metrics_normalized=config["val_metrics_normalized"],
        plot_results=config["plot_results"],
        config=config,
        custom_baselines=custom_baselines,
    )

    ckpt_dir = os.path.join(config["checkpoint_dir"], run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Select checkpoints on val/nll for the density heads, val/rmse for route_a.
    density_head = config.get("estimator_type") in (
        "histogram", "mixture"
    )
    elbo_arm = config.get("model_variant") == "route_a"
    ckpt_monitor = config.get("ckpt_monitor") or (
        "val/nll" if density_head and not elbo_arm else "val/rmse"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=ckpt_monitor,
        save_top_k=1,
        mode="min",
        save_last=config["savelast"],
    )
    early_stop_callback = EarlyStopping(
        monitor=ckpt_monitor, patience=config["patience"], mode="min", verbose=True
    )

    trainer_kwargs = {
        "max_epochs": config["max_epochs"],
        "logger": logger,
        "callbacks": [checkpoint_callback, early_stop_callback],
        "accelerator": "auto",
        "devices": "auto",
        "enable_progress_bar": config["progress_bar"],
    }

    if online_data:
        num_train_tasks = config["num_train_tasks"]
        batch_size = config["batch_size"]
        trainer_kwargs["limit_train_batches"] = max(1, num_train_tasks // batch_size)

        if config.get("total_steps"):
            trainer_kwargs["max_steps"] = config["total_steps"]
            trainer_kwargs["max_epochs"] = -1

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, datamodule)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = CausalMetaModel.load_from_checkpoint(
            best_model_path,
            backbone=backbone,
            custom_baselines=custom_baselines,
        )

    return model, datamodule, trainer
