"""Compare a trained model's forecasts with the FIRETEC runs.

    python ndp-cat1/evaluate.py --run-dir ndp-cat1/runs/<run folder>

Forecasts every frame of each run the model was trained on and writes, under
``<run folder>/evaluation/<store name>/``:

    forecast.gif          FIRETEC, the forecast and their difference, frame by frame
    forecast_metrics.csv  each frame's error, next to the error of predicting no change

``--store`` evaluates other stores instead, for example a run the model has not
seen. It accepts paths and Pelican URLs like the configuration's ``stores``.
"""

import argparse
import logging
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
from firetec_pgml.training import CHECKPOINT_NAME, CONFIG_NAME

from common_utils.config import Config

logger = logging.getLogger(__name__)


def evaluate(run_dir: Path, stores: list[str] | None = None) -> None:
    """Forecast ``stores`` (default: the training runs) with the model in ``run_dir``."""
    config = Config(str(run_dir / CONFIG_NAME), override_from_env=False, override_from_args=False)
    windows = WindowSpec.from_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / CHECKPOINT_NAME, map_location=device, weights_only=True)

    for uri in stores or config.stores:
        run = load_run(uri, config.storage_options, config.wind_speed_scale)
        model = build_model(config, run.geometry).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        forecast = forecast_run(model, run, windows, device, config.batch_size)
        frame_labels = label_frames(forecast, run, windows, config)

        output_dir = run_dir / "evaluation" / store_name(uri)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_forecast_metrics(forecast, frame_labels, output_dir / "forecast_metrics.csv")
        save_forecast_animation(
            forecast, frame_labels, run.geometry.cell_size_m, output_dir / "forecast.gif"
        )
        log_summary(forecast, frame_labels, output_dir)


def label_frames(
    forecast: Forecast, run: FiretecRun, windows: WindowSpec, config: Config
) -> list[str]:
    """Whether each forecast frame was a training target, held out, or from an unseen run."""
    if run.source not in config.stores:
        return ["unseen run"] * len(forecast.frames)
    training_frames, _ = split_frames(run.num_frames, config.val_fraction, windows)
    return [
        "training" if frame < training_frames.stop else "validation" for frame in forecast.frames
    ]


def store_name(uri: str) -> str:
    """The last part of a store's path or URL, without the .zarr ending."""
    return uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")


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
