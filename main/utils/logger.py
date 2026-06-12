"""Experiment logger for training metrics, checkpoints, and results."""

import os
import json
import torch
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentLogger:
    """Simple experiment logger.

    Logs metrics to JSON file and console with timestamps.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{experiment_name}_log.json")
        self.metrics_history = []
        self.start_time = datetime.now()

    def info(self, msg: str):
        """Log an info message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Log metrics dict."""
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "prefix": prefix,
            **{f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()},
        }
        self.metrics_history.append(entry)

    def save(self):
        """Save all logged metrics to JSON."""
        with open(self.log_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

    def save_results_table(self, results: Dict[str, Dict[str, Any]], filename: str):
        """Save a results table (e.g., for experiment comparison)."""
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.info(f"Results table saved to {path}")

    @property
    def elapsed(self) -> float:
        """Elapsed time in hours since logger creation."""
        return (datetime.now() - self.start_time).total_seconds() / 3600
