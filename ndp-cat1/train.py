"""Train the PGML model on FIRETEC runs stored as Zarr.

    python ndp-cat1/train.py --config ndp-cat1/configs/firetec_256x256.yaml

Any key of the configuration except ``storage_options`` can be overridden on
the command line, for example ``--max-epochs 5``; an option that is not a key
stops the run before it starts. Each run writes its own folder under the
configuration's ``output_dir``: the resolved config.yaml, training.log,
history.csv and best_model.pth.
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from firetec_pgml.datasets import WindowSpec, build_window_datasets
from firetec_pgml.features import common_geometry, load_run
from firetec_pgml.modeling import build_loss, build_model
from firetec_pgml.training import CONFIG_NAME, Trainer

from common_utils.config import Config
from common_utils.early_stopping import EarlyStopping
from common_utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def train(config: Config) -> Path:
    """Train as ``config`` describes and return the run's folder."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{timestamp}_{config.experiment_name}"
    run_dir.mkdir(parents=True)
    configure_logger(timestamp, log_dir=run_dir)
    config.save(run_dir / CONFIG_NAME)

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s; results in %s", device, run_dir)

    runs = [
        load_run(
            uri, config.storage_options, config.wind_speed_scale, config.get("zarr_format")
        )
        for uri in config.stores
    ]
    geometry = common_geometry(runs)
    train_set, val_set = build_window_datasets(
        runs, WindowSpec.from_config(config), config.val_fraction
    )
    logger.info(
        "%d run(s) on a %s grid: %d training and %d validation samples",
        len(runs), geometry.grid_shape, len(train_set), len(val_set),
    )

    model = build_model(config, geometry).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=config.min_lr
    )
    trainer = Trainer(
        model=model,
        loss_fn=build_loss(config, geometry),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_pred_frames=config.num_pred_frames,
        amp_dtype=getattr(torch, config.mixed_prec_dtype) if config.enable_amp else None,
    )
    trainer.fit(
        train_loader=DataLoader(train_set, batch_size=config.batch_size, shuffle=True),
        val_loader=DataLoader(val_set, batch_size=config.batch_size),
        max_epochs=config.max_epochs,
        early_stopping=EarlyStopping(
            patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta
        ),
        run_dir=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path of the YAML configuration")
    args, overrides = parser.parse_known_args()
    unknown = unknown_overrides(args.config, overrides)
    if unknown:
        parser.error(
            f"unknown arguments {' '.join(unknown)}: an option is --<key> for a key of "
            f"{args.config}, spelled in full with - for _; storage_options can be set "
            "only in the file"
        )
    # Config reads the overrides itself. Its environment overrides (TRAIN_<KEY>)
    # stay off: common_utils/config.py cannot convert their values.
    train(Config(args.config, override_from_env=False))


def unknown_overrides(config_path: str, arguments: list[str]) -> list[str]:
    """The arguments that are not ``--<key> [value ...]`` for a key of the file.

    Config takes ``--<key>`` for each key, with - for _, except keys whose value
    is a mapping, such as storage_options, and silently ignores anything else.
    Options must be spelled in full.
    """
    with open(config_path) as file:
        settings = yaml.safe_load(file)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for key, value in settings.items():
        if not isinstance(value, dict):
            nargs = "+" if isinstance(value, list) else None
            parser.add_argument(f"--{key.replace('_', '-')}", nargs=nargs)
    _, unknown = parser.parse_known_args(arguments)
    return unknown


if __name__ == "__main__":
    main()
