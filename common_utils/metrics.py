from typing import Dict, Union
from collections import defaultdict
import json
from typing import Any, Dict
import numpy as np
import torch
import sklearn


class MetricsTracker:
    """Tracks and computes various metrics during training"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_metrics = defaultdict(list)  # Metrics updated during training
        self.val_metrics = defaultdict(list)    # Metrics updated during validation
        self.test_metrics = defaultdict(list)    # Metrics updated for final test

    def update(self, metrics: Dict[str, float], metric_type: str = 'train'):
        """Update metrics.
        Args:
            metrics: Dictionary of metric names and values.
            metric_type: 'train' for training metrics, 'val' for validation metrics.
        """
        if metric_type == 'train':
            for key, value in metrics.items():
                self.train_metrics[key].append(value)
        elif metric_type == 'val':
            for key, value in metrics.items():
                self.val_metrics[key].append(value)
        elif metric_type == 'test':
            for key, value in metrics.items():
                self.test_metrics[key].append(value)
        else:
            raise ValueError(f"Invalid metric_type: {metric_type}. Use 'train' or 'val'.")

    def get_metrics(self, metric_type: str = 'train') -> Dict[str, float]:
        """Compute summary metrics for the current epoch.
        Args:
            metric_type: 'train' for training metrics, 'val' for validation metrics.
        """
        summary_metrics = {}
        metrics_dict = self.train_metrics if metric_type == 'train' else self.val_metrics if metric_type == 'val' else self.test_metrics

        for key, values in metrics_dict.items():
            if values:
                summary_metrics[key] = values[-1]  # Latest value
                # summary_metrics[f'{key}_mean'] = np.mean(values)
                # summary_metrics[f'{key}_std'] = np.std(values)
                # summary_metrics[f'{key}_min'] = np.min(values)
                # summary_metrics[f'{key}_max'] = np.max(values)

        # Add elapsed time
        # summary_metrics['elapsed_time'] = time.time() - self.start_time

        return summary_metrics

    def log_metrics(self, metrics: dict, file_path: str, step: int, metric_type: str):
        """
        Log metrics to a .txt file in a structured format.
        Args:
            metrics: Dictionary of metrics to log.
            file_path: Path to the .txt file.
            step: Current step/epoch.
            metric_type: Type of metrics (e.g., 'train' or 'val').
        """
        log_entry = {
            "step": step,
            "metric_type": metric_type,
            **metrics
        }
        with open(file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def compute_additional_metrics(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """Compute additional metrics for model evaluation, supporting both Torch tensors and NumPy arrays."""
    
    # Ensure both inputs are NumPy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Compute metrics
    return {
        'mse': np.square(sklearn.metrics.root_mean_squared_error(target, pred)),
        'mae': sklearn.metrics.mean_absolute_error(target, pred),
        'max_error': np.max(np.abs(target - pred)),
        'mean_error': np.mean(target - pred),
        'std_error': np.std(target - pred),
    }