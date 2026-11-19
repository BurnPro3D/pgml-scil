import pytest
import torch

from conftest import make_numbered_run
from firetec_pgml.datasets import WindowSpec, build_window_datasets, split_frames
from firetec_pgml.features import FiretecRun

WINDOWS = WindowSpec(context_len=6, future=6, num_pred_frames=3)
NUM_FRAMES = 30


def numbered_run(num_frames: int = NUM_FRAMES) -> FiretecRun:
    return make_numbered_run(num_frames)


def frame_numbers(frames: torch.Tensor) -> list[int]:
    return [int(frame.flatten()[0]) for frame in frames]


def test_the_last_frames_are_held_out_for_validation():
    train_frames, val_frames = split_frames(NUM_FRAMES, val_fraction=0.2, windows=WINDOWS)
    assert train_frames == slice(0, 24)
    assert val_frames == slice(18, 30)


def test_sample_takes_context_frames_and_the_forecast_frames_after_them():
    train_set, _ = build_window_datasets([numbered_run()], WINDOWS, val_fraction=0.2)
    inputs, target, _ = train_set[2]
    assert inputs.shape == (6, 4, 2, 2)
    assert target.shape == (3, 1, 2, 2)
    assert frame_numbers(inputs) == [2, 3, 4, 5, 6, 7]
    assert frame_numbers(target) == [8, 9, 10]


def test_no_validation_target_is_a_training_target():
    train_set, val_set = build_window_datasets([numbered_run()], WINDOWS, val_fraction=0.2)
    train_targets = {n for _, target, _ in train_set for n in frame_numbers(target)}
    val_targets = {n for _, target, _ in val_set for n in frame_numbers(target)}
    assert train_targets == set(range(6, 24))
    assert val_targets == set(range(24, 30))


def test_every_run_contributes_samples():
    one_run = build_window_datasets([numbered_run()], WINDOWS, val_fraction=0.2)
    two_runs = build_window_datasets([numbered_run(), numbered_run()], WINDOWS, val_fraction=0.2)
    assert [len(split) for split in one_run] == [16, 4]
    assert [len(split) for split in two_runs] == [32, 8]


def test_at_least_one_forecast_is_held_out_when_the_fraction_rounds_to_nothing():
    _, val_set = build_window_datasets([numbered_run()], WINDOWS, val_fraction=0.0)
    assert len(val_set) == 1
    assert frame_numbers(val_set[0][1]) == [27, 28, 29]


def test_run_too_short_to_leave_a_training_sample_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        build_window_datasets([numbered_run(num_frames=11)], WINDOWS, val_fraction=0.2)
