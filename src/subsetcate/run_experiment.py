import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from subsetcate.args import get_base_parser
from subsetcate.train import run_meta_training_pipeline


def load_user_module(filepath):
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find experiment setup file at: {path}")

    module_dir = str(path.parent)
    module_name = path.stem

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{module_dir}{os.pathsep}{current_pythonpath}"

    import importlib

    importlib.invalidate_caches()
    module = importlib.import_module(module_name)

    return module


def main():
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument("--experiment_setup_path", type=str)
    peek_args, _ = peek_parser.parse_known_args()

    parser = get_base_parser(description="")

    module = None
    if peek_args.experiment_setup_path:
        try:
            module = load_user_module(peek_args.experiment_setup_path)
            if hasattr(module, "add_args"):
                module.add_args(parser)
        except Exception:
            pass

    args = parser.parse_args()

    if module is None:
        module = load_user_module(args.experiment_setup_path)

    experiment_setup = module.experiment_setup

    config = dict(vars(args))

    pl.seed_everything(config["seed"], workers=True)

    exp_name = config["experiment_name"]

    if exp_name not in experiment_setup:
        raise ValueError(f"Experiment {exp_name} not found in setups.")

    setup_entry = experiment_setup[exp_name]

    if callable(setup_entry):
        setup_dict = setup_entry(config)
    else:
        setup_dict = setup_entry

    # Metrics are written as CSV under the checkpoint directory.
    logger = CSVLogger(save_dir=config["checkpoint_dir"], name="metrics")

    model, datamodule, trainer = run_meta_training_pipeline(
        config=config, setup_dict=setup_dict, logger=logger, run_id="run"
    )

    trainer.test(model, dataloaders=datamodule.test_dataloader(num_workers=0))



if __name__ == "__main__":
    main()
