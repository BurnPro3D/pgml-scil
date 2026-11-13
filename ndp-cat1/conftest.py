"""Shared test fixtures. Its place in ndp-cat1/ also lets pytest import firetec_pgml."""

from collections.abc import Iterator

import fsspec
import numpy as np
import pytest
import torch
import xarray as xr
import yaml
from fsspec.implementations.memory import MemoryFileSystem

from firetec_pgml.features import FiretecRun, RunGeometry

from common_utils.config import Config

NUM_FRAMES, NUM_LEVELS, NUM_ROWS, NUM_COLUMNS = 12, 3, 8, 16
INITIAL_FUEL = 0.5
WIND = (3.0, -1.0)  # (x, y) in m/s at the reference height
REFERENCE_HEIGHT = 25.0
# (i, j, time): cells (x index, y index) and when they are ignited
IGNITIONS = [(2, 5, 10.0), (3, 5, 10.0), (9, 1, 20.0)]

MAX_EPOCHS = 12  # of the training runs that the `config` fixture describes


def make_firetec_dataset() -> xr.Dataset:
    """A small dataset laid out like firetec_to_zarr.py's output.

    Frames are 5 s apart from t = 5 s. Fuel is uniform, and halves in an
    ignited cell from the first frame after its ignition.
    """
    time = 5.0 * np.arange(1, NUM_FRAMES + 1)
    heights = np.array([1.0, 24.0, 60.0])
    ignition_i, ignition_j, ignition_time = (np.array(column) for column in zip(*IGNITIONS))

    fuel = np.full((NUM_FRAMES, 1, 1, NUM_ROWS, NUM_COLUMNS), INITIAL_FUEL, dtype=np.float32)
    for i, j, start in IGNITIONS:
        fuel[time > start, 0, 0, j, i] = INITIAL_FUEL / 2

    def wind_field(at_reference_height: float) -> np.ndarray:
        """Wind that equals the given value only at the level nearest the reference height."""
        field = np.zeros((NUM_FRAMES, NUM_LEVELS, NUM_ROWS, NUM_COLUMNS), dtype=np.float32)
        field[:, 1] = at_reference_height
        return field

    return xr.Dataset(
        data_vars={
            "rhoFuel": (("time", "fuel_type", "z_fuel", "y", "x"), fuel),
            "u": (("time", "z", "y", "x"), wind_field(WIND[0])),
            "v": (("time", "z", "y", "x"), wind_field(WIND[1])),
            "ignition_i": ("ignition_point", ignition_i.astype(np.int32)),
            "ignition_j": ("ignition_point", ignition_j.astype(np.int32)),
            "ignition_time": ("ignition_point", ignition_time.astype(np.float64)),
        },
        coords={
            "time": time,
            "z": heights,
            "y": 1.0 + 2.0 * np.arange(NUM_ROWS),
            "x": 1.0 + 2.0 * np.arange(NUM_COLUMNS),
            "fuel_type": [1],
            "z_fuel": heights[:1],
        },
        attrs={"firetec_parameters": {"zu": {"value": REFERENCE_HEIGHT}}},
    )


@pytest.fixture
def firetec_dataset() -> xr.Dataset:
    return make_firetec_dataset()


@pytest.fixture
def firetec_store(tmp_path) -> str:
    """Path of a small FIRETEC-like Zarr store on disk."""
    path = tmp_path / "run.zarr"
    make_firetec_dataset().to_zarr(path, zarr_format=2)
    return str(path)


class ListingDeniedFileSystem(MemoryFileSystem):
    """An in-memory bucket that answers like an S3 bucket to a client that may
    not list it: "access denied", not "not found", for a missing object."""

    protocol = "listingdenied"

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path.replace("listingdenied://", "memory://", 1))

    def cat_file(self, path, start=None, end=None, **kwargs):
        try:
            return super().cat_file(path, start=start, end=end, **kwargs)
        except FileNotFoundError as error:
            raise PermissionError(f"Forbidden: {path}") from error


fsspec.register_implementation("listingdenied", ListingDeniedFileSystem, clobber=True)


@pytest.fixture
def listing_denied_store(tmp_path) -> Iterator[str]:
    """URL of the small FIRETEC-like store, in Zarr format 2, in a bucket that
    answers "access denied" for missing objects. It opens only when the format
    is named: detection would ask for format 3 metadata."""
    path = f"{tmp_path.name}/run.zarr"
    make_firetec_dataset().to_zarr(f"memory://{path}", zarr_format=2)
    yield f"listingdenied://{path}"
    MemoryFileSystem().rm(f"memory://{path}", recursive=True)


def make_numbered_run(num_frames: int) -> FiretecRun:
    """A run whose every frame holds its own frame number, so samples show where they came from."""
    frame_numbers = torch.arange(num_frames, dtype=torch.float32).reshape(num_frames, 1, 1, 1)
    return FiretecRun(
        source="numbered",
        times_s=2.5 * np.arange(1, num_frames + 1),
        inputs=frame_numbers.expand(num_frames, 4, 2, 2).clone(),
        targets=frame_numbers.expand(num_frames, 1, 2, 2).clone(),
        geometry=RunGeometry(grid_shape=(2, 2), cell_size_m=(2.0, 2.0), time_step_s=2.5),
    )


@pytest.fixture
def config(firetec_store, tmp_path) -> Config:
    """A configuration that trains a very small model on the test store.

    It has no zarr_format, like the configurations of runs from before that key
    existed, so the store's format is detected.
    """
    settings = {
        "stores": [firetec_store],
        "storage_options": {},
        "val_fraction": 0.25,
        "wind_speed_scale": 10.0,
        "seed": 0,
        "patch_size": 4,
        "out_channels": 1,
        "context_len": 6,
        "future": 6,
        "num_pred_frames": 3,
        "temporal_patch_size": 3,
        "temporal_stride": 1,
        "embed_dim": 16,
        "depth": 1,
        "num_blocks": 4,
        "physics_loss": True,
        "mse_weight": 1.0,
        "fuel_transport_weight": 0.01,
        "burned_weight": 1.0,
        "unburned_weight": 1.0,
        "fire_metrics_weight": 0.01,
        "ros_method": "horizontal_average",
        "max_epochs": MAX_EPOCHS,
        "batch_size": 2,
        "enable_amp": False,
        "mixed_prec_dtype": "bfloat16",
        "max_lr": 1.0e-2,
        "min_lr": 1.0e-4,
        "weight_decay": 1.0e-2,
        "early_stopping_patience": MAX_EPOCHS,
        "early_stopping_min_delta": 0.0,
        "output_dir": str(tmp_path / "runs"),
        "experiment_name": "test",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    return Config(str(path), override_from_env=False, override_from_args=False)
