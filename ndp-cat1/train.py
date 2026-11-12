"""Train the PGML model on FIRETEC runs stored as Zarr.

    python ndp-cat1/train.py --config ndp-cat1/configs/firetec_256x256.yaml

Any key of the configuration can be overridden on the command line, for
example ``--max-epochs 5``. Each run writes its own folder under the
configuration's ``output_dir``: the resolved config.yaml, training.log,
history.csv and best_model.pth.
"""

import argparse
import logging
import time
from pathlib import Path

import torch
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

    runs = [load_run(uri, config.storage_options, config.wind_speed_scale) for uri in config.stores]
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
    # Config reads the remaining arguments itself, as overrides of its keys.
    args, _ = parser.parse_known_args()
    train(Config(args.config))


if __name__ == "__main__":
    main()
