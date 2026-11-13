"""Compares a trained model's forecasts of a FIRETEC run with the run itself."""

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
from torch.utils.data import DataLoader

from common_utils.dataset import SequentialDatasetLite
from firetec_pgml.datasets import WindowSpec
from firetec_pgml.features import INPUT_CHANNELS, FiretecRun

# Fuel density is a magnitude: one hue, light where little fuel is left.
FUEL_COLORS = LinearSegmentedColormap.from_list("fuel", ["#cde2fb", "#3987e5", "#0d366b"])
# The error has a sign: red where the model burned too much, blue where it left
# too much fuel, neutral gray where it agrees with FIRETEC.
ERROR_COLORS = LinearSegmentedColormap.from_list("error", ["#e34948", "#f0efec", "#2a78d6"])
SURFACE_COLOR = "#fcfcfb"
INK_COLOR = "#0b0b0b"
MUTED_INK_COLOR = "#52514e"
FRAMES_PER_SECOND = 4


@dataclass(frozen=True)
class Forecast:
    """Forecasts of a run's fuel density, one per frame that can be forecast.

    Each frame is forecast from the input frames that end before it, exactly as
    the first forecast frame of a training sample. ``persisted`` is the forecast
    of a model that predicts no change: the fuel in the last input frame.
    """

    frames: np.ndarray  # index in the run of each forecast frame
    times_s: np.ndarray  # simulated time of each forecast frame
    predicted: np.ndarray  # (frame, y, x) model forecast, kg m-3
    observed: np.ndarray  # (frame, y, x) FIRETEC, kg m-3
    persisted: np.ndarray  # (frame, y, x) last input frame, kg m-3


def forecast_run(
    model: nn.Module, run: FiretecRun, windows: WindowSpec, device: torch.device, batch_size: int
) -> Forecast:
    """Forecast every frame of ``run`` that has enough frames before it."""
    samples = SequentialDatasetLite(
        inputs=run.inputs.unsqueeze(0),
        targets=run.targets.unsqueeze(0),
        context_len=windows.context_len,
        num_pred_frames=1,
        future=windows.future,
    )
    fuel_channel = INPUT_CHANNELS.index("fuel_density")
    predicted, observed, persisted = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in DataLoader(samples, batch_size=batch_size):
            forecasts = model(inputs.to(device))
            predicted.append(forecasts[:, 0, 0].cpu())
            observed.append(targets[:, 0, 0])
            persisted.append(inputs[:, -1, fuel_channel])
    frames = windows.future + np.arange(len(samples))
    return Forecast(
        frames=frames,
        times_s=run.times_s[frames],
        predicted=torch.cat(predicted).numpy(),
        observed=torch.cat(observed).numpy(),
        persisted=torch.cat(persisted).numpy(),
    )


def root_mean_square_error(forecast: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Error of each frame over the grid, in the units of the fields."""
    return np.sqrt(np.mean((forecast - observed) ** 2, axis=(1, 2)))


def write_forecast_metrics(forecast: Forecast, frame_labels: list[str], path: Path) -> None:
    """Write each frame's forecast error, next to the error of predicting no change."""
    model_rmse = root_mean_square_error(forecast.predicted, forecast.observed)
    persistence_rmse = root_mean_square_error(forecast.persisted, forecast.observed)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "time_s", "split", "model_rmse", "persistence_rmse"])
        writer.writerows(
            zip(forecast.frames, forecast.times_s, frame_labels, model_rmse, persistence_rmse)
        )


def save_forecast_animation(
    forecast: Forecast, frame_labels: list[str], cell_size_m: tuple[float, float], path: Path
) -> None:
    """Animate FIRETEC's fuel density, the model's forecast and their difference."""
    error = forecast.predicted - forecast.observed
    fuel_scale = {"vmin": 0.0, "vmax": float(forecast.observed.max()), "cmap": FUEL_COLORS}
    error_limit = float(np.abs(error).max())
    error_scale = {"vmin": -error_limit, "vmax": error_limit, "cmap": ERROR_COLORS}
    rows, columns = forecast.observed.shape[1:]
    extent = (0, columns * cell_size_m[0], 0, rows * cell_size_m[1])

    figure, axes = plt.subplots(
        1, 3, figsize=(12, 4.2), sharex=True, sharey=True, layout="constrained",
        facecolor=SURFACE_COLOR,
    )
    panels = [
        ("FIRETEC", forecast.observed, fuel_scale),
        ("Model forecast", forecast.predicted, fuel_scale),
        ("Model − FIRETEC", error, error_scale),
    ]
    images = []
    for axis, (title, fields, scale) in zip(axes, panels):
        images.append(axis.imshow(fields[0], origin="lower", extent=extent, **scale))
        axis.set_title(title, color=INK_COLOR)
        axis.set_xlabel("x (m)", color=MUTED_INK_COLOR)
        axis.tick_params(colors=MUTED_INK_COLOR)
    axes[0].set_ylabel("y (m)", color=MUTED_INK_COLOR)
    figure.colorbar(images[1], ax=axes[:2], label="fuel density (kg m$^{-3}$)")
    figure.colorbar(images[2], ax=axes[2], label="difference (kg m$^{-3}$)")
    heading = figure.suptitle("", color=INK_COLOR)

    def draw(index: int) -> list:
        for image, (_, fields, _) in zip(images, panels):
            image.set_data(fields[index])
        heading.set_text(
            f"t = {forecast.times_s[index]:g} s  ·  frame {forecast.frames[index]}"
            f"  ·  {frame_labels[index]}"
        )
        return images

    animation = FuncAnimation(figure, draw, frames=len(forecast.frames))
    animation.save(path, writer="pillow", fps=FRAMES_PER_SECOND)
    plt.close(figure)
