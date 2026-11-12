"""Training loop for the PGML model."""

import csv
import logging
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from common_utils.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)

# Files of a run's folder
CONFIG_NAME = "config.yaml"
CHECKPOINT_NAME = "best_model.pth"
HISTORY_NAME = "history.csv"

GRADIENT_CLIP_NORM = 1.0  # as in FourCastNet/src/trainer.py


class Trainer:
    """Fits a model that maps input frames to future fuel density frames."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        num_pred_frames: int,
        amp_dtype: torch.dtype | None = None,
    ):
        """``loss_fn(predictions, targets)`` returns the loss and a dict of its parts.

        ``amp_dtype`` turns on mixed precision with that dtype; None trains in float32.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_pred_frames = num_pred_frames
        self.amp_dtype = amp_dtype
        # Gradient scaling guards float16 gradients against underflow; other dtypes do not need it.
        self.scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        early_stopping: EarlyStopping,
        run_dir: Path,
    ) -> list[dict[str, float]]:
        """Train, keeping the weights with the lowest validation loss in ``run_dir``.

        Returns one row of metrics per epoch, also written to ``run_dir``.
        """
        history = []
        best_val_loss = math.inf
        for epoch in range(1, max_epochs + 1):
            learning_rate = self.optimizer.param_groups[0]["lr"]
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)
            self.scheduler.step()

            history.append(
                {"epoch": epoch, "lr": learning_rate}
                | {f"train_{name}": value for name, value in train_metrics.items()}
                | {f"val_{name}": value for name, value in val_metrics.items()}
            )
            write_history(history, run_dir / HISTORY_NAME)
            logger.info(
                "epoch %d/%d | train loss %.6f | val loss %.6f | lr %.2e",
                epoch, max_epochs, train_metrics["loss"], val_metrics["loss"], learning_rate,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint(run_dir / CHECKPOINT_NAME, epoch, best_val_loss)
            if early_stopping(val_metrics["loss"]):
                logger.info("Stopping early: no better validation loss since epoch %d",
                            epoch - early_stopping.patience)
                break
        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        """One pass over ``loader``; returns the per-sample mean of the loss and its parts."""
        self.model.train(training)
        totals = defaultdict(float)
        num_samples = 0
        with torch.set_grad_enabled(training):
            for inputs, targets, _ in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                loss, parts = self._loss(inputs, targets)
                if training:
                    self._optimizer_step(loss)
                batch_size = inputs.shape[0]
                totals["loss"] += loss.item() * batch_size
                for name, value in parts.items():
                    totals[name] += value * batch_size
                num_samples += batch_size
        return {name: total / num_samples for name, total in totals.items()}

    def _loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        mixed_precision = torch.autocast(
            self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        )
        with mixed_precision:
            # The model returns as many frames as it is given; the first ones are the forecast.
            predictions = self.model(inputs)[:, : self.num_pred_frames]
            return self.loss_fn(predictions, targets)

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError(f"the training loss became {loss.item()}")
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_NORM)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        # 'epoch' and 'state_dict' are the keys of the repository's other checkpoints.
        torch.save(
            {"epoch": epoch, "state_dict": self.model.state_dict(), "val_loss": val_loss}, path
        )
        logger.info("Saved best model (epoch %d, val loss %.6f) to %s", epoch, val_loss, path)


def write_history(history: list[dict[str, float]], path: Path) -> None:
    """Write the per-epoch metrics as CSV."""
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
