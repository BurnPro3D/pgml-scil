import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from conftest import make_firetec_dataset, make_numbered_run
from evaluate import STORE_RECORD_NAME, evaluate, store_name
from firetec_pgml.datasets import WindowSpec
from firetec_pgml.evaluation import forecast_run, root_mean_square_error
from firetec_pgml.features import INPUT_CHANNELS, FiretecRun
from train import train

WINDOWS = WindowSpec(context_len=6, future=6, num_pred_frames=3)
NUM_FRAMES = 10
FORECAST_OFFSET = 100.0


class AddsAnOffset(nn.Module):
    """Stands in for the model: returns the fuel channel of its input frames, plus an offset."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fuel = inputs[:, :, INPUT_CHANNELS.index("fuel_density")]
        return fuel.unsqueeze(2) + FORECAST_OFFSET


@pytest.fixture
def numbered_run() -> FiretecRun:
    return make_numbered_run(NUM_FRAMES)


def test_every_frame_with_enough_frames_before_it_is_forecast(numbered_run):
    forecast = forecast_run(AddsAnOffset(), numbered_run, WINDOWS, torch.device("cpu"), batch_size=3)
    assert forecast.frames.tolist() == [6, 7, 8, 9]
    assert forecast.times_s.tolist() == [17.5, 20.0, 22.5, 25.0]
    assert forecast.observed[:, 0, 0].tolist() == [6, 7, 8, 9]


def test_forecast_is_the_first_frame_the_model_returns(numbered_run):
    forecast = forecast_run(AddsAnOffset(), numbered_run, WINDOWS, torch.device("cpu"), batch_size=3)
    first_input_frames = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(forecast.predicted[:, 0, 0], first_input_frames + FORECAST_OFFSET)


def test_persistence_forecast_is_the_last_input_frame(numbered_run):
    forecast = forecast_run(AddsAnOffset(), numbered_run, WINDOWS, torch.device("cpu"), batch_size=3)
    assert forecast.persisted[:, 0, 0].tolist() == [5, 6, 7, 8]


def test_a_run_too_short_to_forecast_is_refused_with_its_length():
    with pytest.raises(ValueError, match="has 6 frames"):
        forecast_run(AddsAnOffset(), make_numbered_run(6), WINDOWS, torch.device("cpu"), batch_size=3)


def test_error_is_computed_per_frame():
    observed = np.zeros((2, 2, 2))
    forecast = np.stack([np.full((2, 2), 3.0), np.array([[0.0, 0.0], [0.0, 2.0]])])
    np.testing.assert_allclose(root_mean_square_error(forecast, observed), [3.0, 1.0])


@pytest.mark.parametrize(
    "uri",
    ["/data/TestRun.zarr", "/data/TestRun.zarr/", "osdf:///namespace/TestRun.zarr"],
)
def test_store_name_is_the_last_part_of_the_uri(uri):
    assert store_name(uri) == "TestRun"


def test_the_current_folder_is_named_after_itself(tmp_path, monkeypatch):
    store = tmp_path / "TestRun.zarr"
    (store / "evaluation").mkdir(parents=True)
    monkeypatch.chdir(store / "evaluation")
    assert store_name("..") == store_name("../") == "TestRun"


def test_evaluation_writes_metrics_and_an_animation_for_each_training_run(config):
    run_dir = train(config)
    evaluate(run_dir)

    output_dir = run_dir / "evaluation" / "run"
    with open(output_dir / "forecast_metrics.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    # The test store has 12 frames, of which the last 3 are held out.
    assert [int(row["frame"]) for row in rows] == [6, 7, 8, 9, 10, 11]
    assert [row["split"] for row in rows] == ["training"] * 3 + ["validation"] * 3
    assert all(float(row["model_rmse"]) >= 0 for row in rows)
    with Image.open(output_dir / "forecast.gif") as animation:
        assert animation.n_frames == len(rows)


def test_training_and_evaluation_open_the_stores_in_the_configured_zarr_format(
    config, listing_denied_store
):
    config.stores = [listing_denied_store]
    config.zarr_format = 2
    run_dir = train(config)
    evaluate(run_dir)
    assert (run_dir / "evaluation" / "run" / "forecast_metrics.csv").exists()


def write_store(path) -> str:
    make_firetec_dataset().to_zarr(path, zarr_format=2)
    return str(path)


def splits(output_dir) -> set[str]:
    with open(output_dir / "forecast_metrics.csv", newline="") as file:
        return {row["split"] for row in csv.DictReader(file)}


def test_a_store_the_model_was_not_trained_on_is_labelled_unseen(config, tmp_path):
    run_dir = train(config)
    evaluate(run_dir, stores=[write_store(tmp_path / "other.zarr")])
    assert splits(run_dir / "evaluation" / "other") == {"unseen run"}


@pytest.mark.parametrize("spelling", ["trailing slash", "symbolic link"])
def test_a_training_store_spelled_differently_keeps_its_labels(config, tmp_path, spelling):
    run_dir = train(config)
    store = config.stores[0]
    if spelling == "trailing slash":
        respelled = store + "/"
    else:
        respelled = str(tmp_path / "alias.zarr")
        Path(respelled).symlink_to(store, target_is_directory=True)
    evaluate(run_dir, stores=[respelled])
    assert splits(run_dir / "evaluation" / store_name(respelled)) == {"training", "validation"}


def test_the_results_folder_names_its_store(config):
    run_dir = train(config)
    evaluate(run_dir)
    record = run_dir / "evaluation" / "run" / STORE_RECORD_NAME
    assert record.read_text().strip() == str(Path(config.stores[0]).resolve())


def test_a_store_named_like_another_stores_results_is_refused(config, tmp_path):
    run_dir = train(config)
    evaluate(run_dir)
    namesake = write_store(tmp_path / "elsewhere" / "run.zarr")
    with pytest.raises(ValueError, match="same name"):
        evaluate(run_dir, stores=[namesake])
    assert splits(run_dir / "evaluation" / "run") == {"training", "validation"}


def test_two_stores_with_one_name_are_refused_before_anything_is_written(config, tmp_path):
    run_dir = train(config)
    namesake = write_store(tmp_path / "elsewhere" / "run.zarr")
    with pytest.raises(ValueError, match="same name"):
        evaluate(run_dir, stores=[config.stores[0], namesake])
    assert not (run_dir / "evaluation").exists()
