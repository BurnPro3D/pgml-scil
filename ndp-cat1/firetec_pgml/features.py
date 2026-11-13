"""Turns a FIRETEC run into the input channels and the target of the PGML model.

The model sees one frame per FIRETEC output time. Each frame has the channels
of the repository's QUIC-Fire data, filled from FIRETEC's fields:

    fuel_density  bulk density of dry fuel in the ground layer, kg m-3
    wind_x        ambient wind along x, in units of ``wind_speed_scale``
    wind_y        ambient wind along y, in units of ``wind_speed_scale``
    ignition      1 until a cell is ignited, 0 from then on

The target is the fuel density of later frames.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import xarray as xr

from firetec_pgml.store import open_firetec_store

INPUT_CHANNELS = ("fuel_density", "wind_x", "wind_y", "ignition")
REQUIRED_VARIABLES = ("rhoFuel", "u", "v", "ignition_i", "ignition_j", "ignition_time")


@dataclass(frozen=True)
class RunGeometry:
    """Grid and output cadence of a run."""

    grid_shape: tuple[int, int]  # cells along (y, x)
    cell_size_m: tuple[float, float]  # (dx, dy)
    time_step_s: float  # time between frames


@dataclass(frozen=True)
class FiretecRun:
    """One FIRETEC run as model frames."""

    source: str  # where the store was read from
    times_s: np.ndarray  # (time,) simulated time of each frame
    inputs: torch.Tensor  # (time, channel, y, x), channels as in INPUT_CHANNELS
    targets: torch.Tensor  # (time, 1, y, x), fuel density
    geometry: RunGeometry

    @property
    def num_frames(self) -> int:
        return self.inputs.shape[0]


def load_run(
    uri: str,
    storage_options: Mapping[str, Any] | None = None,
    wind_speed_scale: float = 10.0,
    zarr_format: int | str | None = None,
) -> FiretecRun:
    """Read the store at ``uri`` and build the model frames of its run.

    ``storage_options`` and ``zarr_format`` are `open_firetec_store`'s.
    """
    with open_firetec_store(uri, storage_options, zarr_format) as store:
        return run_from_store(store, source=uri, wind_speed_scale=wind_speed_scale)


def run_from_store(store: xr.Dataset, source: str, wind_speed_scale: float) -> FiretecRun:
    """Build the model frames of the run held in an open FIRETEC store."""
    _check_store(store, source)
    fuel = surface_fuel_density(store)
    if not np.isfinite(fuel).all():
        raise ValueError(f"{source} has missing fuel density values")
    wind_x, wind_y = ambient_wind(store)
    channels = {
        "fuel_density": fuel,
        "wind_x": np.full_like(fuel, wind_x / wind_speed_scale),
        "wind_y": np.full_like(fuel, wind_y / wind_speed_scale),
        "ignition": ~ignited_cells(store),
    }
    inputs = np.stack([channels[name] for name in INPUT_CHANNELS], axis=1)
    return FiretecRun(
        source=source,
        times_s=store["time"].values,
        inputs=torch.from_numpy(inputs.astype(np.float32)),
        targets=torch.from_numpy(fuel[:, np.newaxis].astype(np.float32)),
        geometry=run_geometry(store),
    )


def surface_fuel_density(store: xr.Dataset) -> np.ndarray:
    """Dry fuel bulk density in the ground layer, all fuel types together: (time, y, x)."""
    ground_layer = store["rhoFuel"].isel(z_fuel=0)
    return ground_layer.sum("fuel_type", skipna=False).transpose("time", "y", "x").values


def ambient_wind(store: xr.Dataset) -> tuple[float, float]:
    """Wind (x, y) in m/s that the run was driven with.

    FIRETEC sets its ambient wind at the reference height ``zu``. This is the
    horizontal mean of the wind at the model level nearest that height, in the
    first output.
    """
    reference_height = store.attrs["firetec_parameters"]["zu"]["value"]
    level = store[["u", "v"]].isel(time=0).sel(z=reference_height, method="nearest")
    return float(level["u"].mean()), float(level["v"].mean())


def ignited_cells(store: xr.Dataset) -> np.ndarray:
    """Cells an ignition was applied to at or before each frame: bool (time, y, x).

    The ignition file numbers cells from zero in x and y: FIRETEC's ignite.F90
    adds one to reach its own 1-based index. The stored indices therefore
    address the arrays directly. Points outside the grid ignite nothing in
    FIRETEC and are skipped.
    """
    rows, columns = store["ignition_j"].values, store["ignition_i"].values
    num_rows, num_columns = store.sizes["y"], store.sizes["x"]
    inside = (rows >= 0) & (rows < num_rows) & (columns >= 0) & (columns < num_columns)
    applied = store["ignition_time"].values[np.newaxis, :] <= store["time"].values[:, np.newaxis]
    frames, points = np.nonzero(applied & inside)
    ignited = np.zeros((store.sizes["time"], num_rows, num_columns), dtype=bool)
    ignited[frames, rows[points], columns[points]] = True
    return ignited


def run_geometry(store: xr.Dataset) -> RunGeometry:
    """Grid shape, cell size and output interval of a store."""
    x, y, time = store["x"].values, store["y"].values, store["time"].values
    return RunGeometry(
        grid_shape=(y.size, x.size),
        cell_size_m=(float(x[1] - x[0]), float(y[1] - y[0])),
        time_step_s=float(time[1] - time[0]),
    )


def common_geometry(runs: Sequence[FiretecRun]) -> RunGeometry:
    """The geometry all ``runs`` share; one model cannot train on differing grids."""
    geometry = runs[0].geometry
    for run in runs[1:]:
        if run.geometry != geometry:
            raise ValueError(
                f"{run.source} has {run.geometry}, but {runs[0].source} has {geometry}"
            )
    return geometry


def _check_store(store: xr.Dataset, source: str) -> None:
    missing = [name for name in REQUIRED_VARIABLES if name not in store]
    if missing:
        raise ValueError(f"{source} lacks {missing}; is it a FIRETEC store with ignition data?")
    if store.sizes["time"] < 2:
        raise ValueError(f"{source} holds fewer than two output frames")
    intervals = np.diff(store["time"].values)
    if not np.allclose(intervals, intervals[0]):
        raise ValueError(f"{source} has unevenly spaced output times; is an output step missing?")
