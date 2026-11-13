"""Cuts FIRETEC runs into training and validation samples."""

from collections.abc import Sequence
from dataclasses import dataclass

from torch.utils.data import ConcatDataset, Dataset

from common_utils.config import Config
from common_utils.dataset import SequentialDatasetLite
from firetec_pgml.features import FiretecRun


@dataclass(frozen=True)
class WindowSpec:
    """How a sample is cut from a run's frames.

    The sample starting at frame ``start`` holds the ``context_len`` input
    frames from ``start`` and, as its target, the ``num_pred_frames`` fuel
    frames from ``start + future``.
    """

    context_len: int
    future: int
    num_pred_frames: int
    temporal_stride: int = 1

    @classmethod
    def from_config(cls, config: Config) -> "WindowSpec":
        return cls(
            context_len=config.context_len,
            future=config.future,
            num_pred_frames=config.num_pred_frames,
            temporal_stride=config.temporal_stride,
        )

    @property
    def span(self) -> int:
        """Number of consecutive frames one sample covers."""
        return self.future + self.num_pred_frames


def build_window_datasets(
    runs: Sequence[FiretecRun], windows: WindowSpec, val_fraction: float
) -> tuple[Dataset, Dataset]:
    """Training and validation samples, split in time within each run.

    Every sample is ``(inputs, target, sequence index)``, as in the
    repository's other datasets.
    """
    train_parts, val_parts = [], []
    for run in runs:
        train_frames, val_frames = split_frames(run.num_frames, val_fraction, windows)
        train_parts.append(_windows_of(run, train_frames, windows))
        val_parts.append(_windows_of(run, val_frames, windows))
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def split_frames(num_frames: int, val_fraction: float, windows: WindowSpec) -> tuple[slice, slice]:
    """Frame ranges of a run to cut training and validation samples from.

    The last ``val_fraction`` of the frames, and at least one sample's worth,
    are held out as validation targets. Validation samples take their input
    frames from just before them, which training has seen, but no held-out
    frame is ever a training target.
    """
    num_val_targets = max(windows.num_pred_frames, round(num_frames * val_fraction))
    first_val_target = num_frames - num_val_targets
    if first_val_target < windows.span:
        raise ValueError(
            f"a run of {num_frames} frames is too short: holding out {num_val_targets} "
            f"leaves {first_val_target} for training, and one sample spans {windows.span}"
        )
    return slice(0, first_val_target), slice(first_val_target - windows.future, num_frames)


def _windows_of(run: FiretecRun, frames: slice, windows: WindowSpec) -> SequentialDatasetLite:
    return SequentialDatasetLite(
        inputs=run.inputs[frames].unsqueeze(0),
        targets=run.targets[frames].unsqueeze(0),
        context_len=windows.context_len,
        temporal_stride=windows.temporal_stride,
        num_pred_frames=windows.num_pred_frames,
        future=windows.future,
    )
