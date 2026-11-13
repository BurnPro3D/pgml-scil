import numpy as np
import pytest
import torch
import xarray as xr

from conftest import IGNITIONS, INITIAL_FUEL, NUM_COLUMNS, NUM_FRAMES, NUM_ROWS, WIND
from firetec_pgml.features import (
    INPUT_CHANNELS,
    common_geometry,
    load_run,
    run_from_store,
    surface_fuel_density,
)

WIND_SPEED_SCALE = 10.0


def channel(run, name: str) -> torch.Tensor:
    return run.inputs[:, INPUT_CHANNELS.index(name)]


@pytest.fixture
def run(firetec_dataset):
    return run_from_store(firetec_dataset, source="test", wind_speed_scale=WIND_SPEED_SCALE)


def test_frames_have_one_channel_per_model_input(run):
    assert run.inputs.shape == (NUM_FRAMES, len(INPUT_CHANNELS), NUM_ROWS, NUM_COLUMNS)
    assert run.targets.shape == (NUM_FRAMES, 1, NUM_ROWS, NUM_COLUMNS)
    assert run.inputs.dtype == run.targets.dtype == torch.float32


def test_target_is_the_fuel_density_channel(run):
    assert torch.equal(run.targets[:, 0], channel(run, "fuel_density"))
    assert channel(run, "fuel_density").max().item() == pytest.approx(INITIAL_FUEL)


def test_wind_channels_hold_the_scaled_wind_at_the_reference_height(run):
    assert torch.allclose(channel(run, "wind_x"), torch.tensor(WIND[0] / WIND_SPEED_SCALE))
    assert torch.allclose(channel(run, "wind_y"), torch.tensor(WIND[1] / WIND_SPEED_SCALE))


def test_ignition_channel_drops_to_zero_from_the_ignition_time(run, firetec_dataset):
    ignition = channel(run, "ignition")
    times = firetec_dataset["time"].values
    for i, j, start in IGNITIONS:
        assert torch.all(ignition[times < start, j, i] == 1)
        assert torch.all(ignition[times >= start, j, i] == 0)
    assert int((ignition[-1] == 0).sum()) == len(IGNITIONS)


def test_ignited_cells_are_the_cells_whose_fuel_burns(run):
    burned = channel(run, "fuel_density")[-1] < INITIAL_FUEL
    assert torch.equal(channel(run, "ignition")[-1] == 0, burned)


def test_ignition_points_outside_the_grid_are_skipped(firetec_dataset):
    firetec_dataset["ignition_i"][:] = [2, NUM_COLUMNS, -1]
    run = run_from_store(firetec_dataset, source="test", wind_speed_scale=WIND_SPEED_SCALE)
    ignited = channel(run, "ignition")[-1] == 0
    assert ignited.nonzero().tolist() == [[5, 2]]


def test_fuel_density_adds_the_fuel_types_of_the_ground_layer():
    fuel = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2)
    store = xr.Dataset({"rhoFuel": (("time", "fuel_type", "z_fuel", "y", "x"), fuel)})
    np.testing.assert_array_equal(surface_fuel_density(store), fuel[:, :, 0].sum(axis=1))


def test_geometry_comes_from_the_coordinates(run):
    assert run.geometry.grid_shape == (NUM_ROWS, NUM_COLUMNS)
    assert run.geometry.cell_size_m == (2.0, 2.0)
    assert run.geometry.time_step_s == 5.0


def test_load_run_reads_a_store_from_disk(firetec_store, run):
    loaded = load_run(firetec_store, wind_speed_scale=WIND_SPEED_SCALE)
    assert torch.equal(loaded.inputs, run.inputs)
    assert loaded.geometry == run.geometry


def test_load_run_opens_the_store_in_the_zarr_format_it_is_given(listing_denied_store, run):
    loaded = load_run(listing_denied_store, wind_speed_scale=WIND_SPEED_SCALE, zarr_format=2)
    assert torch.equal(loaded.inputs, run.inputs)


def test_store_without_ignition_data_is_rejected(firetec_dataset):
    store = firetec_dataset.drop_vars(["ignition_i", "ignition_j", "ignition_time"])
    with pytest.raises(ValueError, match="ignition"):
        run_from_store(store, source="test", wind_speed_scale=WIND_SPEED_SCALE)


def test_store_with_a_missing_output_step_is_rejected(firetec_dataset):
    store = firetec_dataset.drop_isel(time=3)
    with pytest.raises(ValueError, match="unevenly spaced"):
        run_from_store(store, source="test", wind_speed_scale=WIND_SPEED_SCALE)


def test_store_with_missing_fuel_values_is_rejected(firetec_dataset):
    firetec_dataset["rhoFuel"][2, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="missing fuel density"):
        run_from_store(firetec_dataset, source="test", wind_speed_scale=WIND_SPEED_SCALE)


def test_runs_on_different_grids_cannot_be_combined(run, firetec_dataset):
    shorter = run_from_store(
        firetec_dataset.isel(y=slice(0, 6)), source="short", wind_speed_scale=WIND_SPEED_SCALE
    )
    assert common_geometry([run, run]) == run.geometry
    with pytest.raises(ValueError, match="short"):
        common_geometry([run, shorter])
