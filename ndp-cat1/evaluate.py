"""Compare a trained model's forecasts with the FIRETEC runs.

    python ndp-cat1/evaluate.py --run-dir ndp-cat1/runs/<run folder>

Forecasts every frame of each run the model was trained on and writes, under
``<run folder>/evaluation/<store name>/``:

    forecast.gif          FIRETEC, the forecast and their difference, frame by frame
    forecast_metrics.csv  each frame's error, next to the error of predicting no change
    store.txt             the store the results are for

``--store`` evaluates other stores instead, for example a run the model has not
seen. It accepts paths and S3 or Pelican URLs like the configuration's
``stores``, and reads them with the run's ``storage_options`` and ``zarr_format``.
A store is refused when another store's results already have its name.
"""

import argparse
import logging
import os
from pathlib import Path

import torch

from firetec_pgml.datasets import WindowSpec, split_frames
from firetec_pgml.evaluation import (
    Forecast,
    forecast_run,
    root_mean_square_error,
    save_forecast_animation,
    write_forecast_metrics,
)
from firetec_pgml.features import FiretecRun, load_run
from firetec_pgml.modeling import build_model
from firetec_pgml.store import canonical_uri, url_scheme
from firetec_pgml.training import CHECKPOINT_NAME, CONFIG_NAME

from common_utils.config import Config

logger = logging.getLogger(__name__)

STORE_RECORD_NAME = "store.txt"


def evaluate(run_dir: Path, stores: list[str] | None = None) -> None:
    """Forecast ``stores`` (default: the training runs) with the model in ``run_dir``."""
    config = Config(str(run_dir / CONFIG_NAME), override_from_env=False, override_from_args=False)
    windows = WindowSpec.from_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / CHECKPOINT_NAME, map_location=device, weights_only=True)

    stores = stores or config.stores
    for uri, output_dir in zip(stores, output_folders(run_dir, stores)):
        run = load_run(
            uri, config.storage_options, config.wind_speed_scale, config.get("zarr_format")
        )
        model = build_model(config, run.geometry).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        forecast = forecast_run(model, run, windows, device, config.batch_size)
        frame_labels = label_frames(forecast, run, windows, config)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / STORE_RECORD_NAME).write_text(canonical_uri(uri) + "\n")
        write_forecast_metrics(forecast, frame_labels, output_dir / "forecast_metrics.csv")
        save_forecast_animation(
            forecast, frame_labels, run.geometry.cell_size_m, output_dir / "forecast.gif"
        )
        log_summary(forecast, frame_labels, output_dir)


def output_folders(run_dir: Path, stores: list[str]) -> list[Path]:
    """The folder for each store's results: ``evaluation/<store name>``.

    Two stores with the same name would overwrite each other's results, so a
    folder that holds, or is about to hold, another store's results is refused.
    Folders written before store.txt existed name no store and are not checked.
    """
    folders, owners = [], {}
    for uri in stores:
        folder = run_dir / "evaluation" / store_name(uri)
        record = folder / STORE_RECORD_NAME
        owner = owners.get(folder) or (record.read_text().strip() if record.exists() else None)
        if owner is not None and owner != canonical_uri(uri):
            raise ValueError(
                f"{folder} is for the results of {owner}, and {uri} has the same name; "
                "give one of the stores another name"
            )
        owners[folder] = canonical_uri(uri)
        folders.append(folder)
    return folders


def label_frames(
    forecast: Forecast, run: FiretecRun, windows: WindowSpec, config: Config
) -> list[str]:
    """Whether each forecast frame lies in the part of its run that training used,
    in the held-out part, or in a run the model was not trained on."""
    if canonical_uri(run.source) not in {canonical_uri(uri) for uri in config.stores}:
        return ["unseen run"] * len(forecast.frames)
    training_frames, _ = split_frames(run.num_frames, config.val_fraction, windows)
    return [
        "training" if frame < training_frames.stop else "validation" for frame in forecast.frames
    ]


def store_name(uri: str) -> str:
    """The last part of a store's path or URL, without the .zarr ending."""
    # An absolute path, so that "." and ".." name the store they stand for.
    location = uri if url_scheme(uri) else os.path.abspath(uri)
    return location.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")


def log_summary(forecast: Forecast, frame_labels: list[str], output_dir: Path) -> None:
    model_rmse = root_mean_square_error(forecast.predicted, forecast.observed)
    persistence_rmse = root_mean_square_error(forecast.persisted, forecast.observed)
    for label in dict.fromkeys(frame_labels):
        selected = [frame_label == label for frame_label in frame_labels]
        logger.info(
            "%s frames (%d): mean RMSE %.5f kg/m3, predicting no change %.5f kg/m3",
            label, sum(selected), model_rmse[selected].mean(), persistence_rmse[selected].mean(),
        )
    logger.info("Wrote %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path, help="folder written by train.py")
    parser.add_argument("--store", nargs="+", help="stores to evaluate instead of the training runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate(args.run_dir, args.store)


if __name__ == "__main__":
    main()
