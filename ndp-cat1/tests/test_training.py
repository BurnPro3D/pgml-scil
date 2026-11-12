import csv
import math

import pytest
import torch

from conftest import MAX_EPOCHS
from firetec_pgml.features import RunGeometry, common_geometry, load_run
from firetec_pgml.modeling import build_loss, build_model
from firetec_pgml.training import CHECKPOINT_NAME, CONFIG_NAME, HISTORY_NAME
from train import train

from common_utils.config import Config


def read_history(run_dir) -> list[dict[str, float]]:
    with open(run_dir / HISTORY_NAME, newline="") as file:
        return [{name: float(value) for name, value in row.items()} for row in csv.DictReader(file)]


def test_training_writes_the_run_folder_and_lowers_the_loss(config):
    run_dir = train(config)

    assert {path.name for path in run_dir.iterdir()} == {
        CONFIG_NAME, CHECKPOINT_NAME, HISTORY_NAME, "training.log",
    }
    history = read_history(run_dir)
    assert [row["epoch"] for row in history] == list(range(1, MAX_EPOCHS + 1))
    assert all(math.isfinite(value) for row in history for value in row.values())
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_best_checkpoint_loads_into_a_model_built_from_the_saved_config(config):
    run_dir = train(config)

    saved_config = Config(
        str(run_dir / CONFIG_NAME), override_from_env=False, override_from_args=False
    )
    geometry = common_geometry([load_run(uri) for uri in saved_config.stores])
    checkpoint = torch.load(run_dir / CHECKPOINT_NAME, weights_only=True)
    build_model(saved_config, geometry).load_state_dict(checkpoint["state_dict"])

    best = min(read_history(run_dir), key=lambda row: row["val_loss"])
    assert checkpoint["epoch"] == best["epoch"]
    assert checkpoint["val_loss"] == pytest.approx(best["val_loss"])


def test_loss_uses_the_cell_size_and_frame_interval_of_the_runs(config):
    geometry = RunGeometry(grid_shape=(8, 16), cell_size_m=(4.0, 3.0), time_step_s=2.5)
    physics_loss = build_loss(config, geometry).physics_loss_fn
    assert physics_loss.cell_size == (4.0, 3.0)
    assert physics_loss.temporal_resolution == 2.5


@pytest.mark.parametrize(
    "setting, value, message",
    [
        ("patch_size", 3, "patch_size"),
        ("temporal_patch_size", 4, "temporal_patch_size"),
        ("num_pred_frames", 7, "num_pred_frames"),
    ],
)
def test_model_settings_that_do_not_fit_are_rejected(config, setting, value, message):
    setattr(config, setting, value)
    geometry = RunGeometry(grid_shape=(8, 16), cell_size_m=(2.0, 2.0), time_step_s=5.0)
    with pytest.raises(ValueError, match=message):
        build_model(config, geometry)
