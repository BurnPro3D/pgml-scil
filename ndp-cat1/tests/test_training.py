import csv
import logging
import math
import sys

import pytest
import torch
from torch import nn

import train as train_script
from conftest import MAX_EPOCHS
from firetec_pgml.features import INPUT_CHANNELS, RunGeometry, common_geometry, load_run
from firetec_pgml.modeling import ChangeFromLastFrame, build_loss, build_model
from firetec_pgml.training import CHECKPOINT_NAME, CONFIG_NAME, HISTORY_NAME, Trainer
from train import train, unknown_overrides

from common_utils.config import Config
from common_utils.early_stopping import EarlyStopping


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


@pytest.mark.parametrize("predict_change", [False, True])
def test_best_checkpoint_loads_into_a_model_built_from_the_saved_config(config, predict_change):
    config.predict_change = predict_change
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


class Zeros(nn.Module):
    """Stands in for the model: returns no change for every input frame."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs[:, :, :1])


def test_a_model_that_predicts_the_change_adds_it_to_the_last_input_frame():
    inputs = torch.rand(2, 6, len(INPUT_CHANNELS), 4, 4)
    fuel = INPUT_CHANNELS.index("fuel_density")
    forecast = ChangeFromLastFrame(Zeros())(inputs)
    assert forecast.shape == (2, 6, 1, 4, 4)
    assert torch.equal(forecast, inputs[:, -1:, fuel : fuel + 1].expand_as(forecast))


# The stubbed epochs take no optimizer step, which the scheduler warns about.
@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)`")
def test_early_stopping_names_the_epoch_of_the_best_model(tmp_path, caplog):
    """The saved model is the one with the lowest validation loss, even when early
    stopping only counts improvements larger than its min_delta."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model, loss_fn=None, optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        device=torch.device("cpu"), num_pred_frames=1,
    )
    val_losses = iter([1.0, 0.99, 0.98])
    trainer._run_epoch = lambda loader, training: {"loss": 0.5 if training else next(val_losses)}
    with caplog.at_level(logging.INFO):
        trainer.fit(None, None, max_epochs=10, early_stopping=EarlyStopping(patience=2, min_delta=0.05),
                    run_dir=tmp_path)
    assert "Stopping early after epoch 3" in caplog.text
    assert "The best model is from epoch 3" in caplog.text


def config_file(tmp_path) -> str:
    """The file the `config` fixture was read from."""
    return str(tmp_path / "config.yaml")


def test_options_that_name_keys_of_the_file_are_known(config, tmp_path):
    arguments = [
        "--max-epochs", "2", "--embed-dim=8", "--stores", "a.zarr", "b.zarr",
        "--min-lr=-1e-5", "--seed", "-1", "--physics-loss", "false",
    ]
    assert unknown_overrides(config_file(tmp_path), arguments) == []


@pytest.mark.parametrize(
    "arguments, unknown",
    [
        (["--max_epochs", "3"], ["--max_epochs", "3"]),
        (["--epochs", "1"], ["--epochs", "1"]),
        (["--max-epoch", "2"], ["--max-epoch", "2"]),
        (["-max-epochs", "2"], ["-max-epochs", "2"]),
        (["max-epochs", "2"], ["max-epochs", "2"]),
        (["--max-epochs", "2", "3"], ["3"]),
        (["--storage-options", "{}"], ["--storage-options", "{}"]),
    ],
    ids=["underscore", "not a key", "abbreviated", "single dash", "no dash", "extra value", "mapping"],
)
def test_arguments_config_would_ignore_are_unknown(config, tmp_path, arguments, unknown):
    assert unknown_overrides(config_file(tmp_path), arguments) == unknown


def run_main(monkeypatch, *arguments) -> list[Config]:
    """Run train.py's main() with the given arguments, recording the configuration it trains with."""
    trained_with = []
    monkeypatch.setattr(sys, "argv", ["train.py", *arguments])
    monkeypatch.setattr(train_script, "train", trained_with.append)
    train_script.main()
    return trained_with


def test_an_option_overrides_the_file(config, tmp_path, monkeypatch):
    [trained_with] = run_main(monkeypatch, "--config", config_file(tmp_path), "--max-epochs", "2")
    assert trained_with.max_epochs == 2


def test_an_unknown_option_stops_the_run_before_training(config, tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as stopped:
        run_main(monkeypatch, "--config", config_file(tmp_path), "--max_epochs", "2")
    assert stopped.value.code == 2


def test_environment_overrides_are_ignored(config, tmp_path, monkeypatch):
    """common_utils/config.py cannot convert TRAIN_<KEY> values, so they stay off."""
    monkeypatch.setenv("TRAIN_MAX_EPOCHS", "5")
    [trained_with] = run_main(monkeypatch, "--config", config_file(tmp_path))
    assert trained_with.max_epochs == MAX_EPOCHS
