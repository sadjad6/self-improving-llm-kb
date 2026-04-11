"""Experiment tracking using MLflow."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.utils.config import ExperimentConfig

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow-based experiment tracker for reproducible experiments.

    Tracks configurations, retrieval strategies, prompt versions,
    and evaluation results across experiment runs.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        if mlflow is None:
            raise ImportError("mlflow is required: pip install mlflow")
        self.config = config or ExperimentConfig()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        self._run_id: Optional[str] = None

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run.

        Args:
            run_name: Optional name for the run.

        Returns:
            The run ID.
        """
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        logger.info("Started MLflow run: %s (%s)", run_name, self._run_id)
        return self._run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the current run."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the current run."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_config(self, config: Any) -> None:
        """Log a full configuration object."""
        if hasattr(config, "__dataclass_fields__"):
            params = {}
            for field_name in config.__dataclass_fields__:
                value = getattr(config, field_name)
                if hasattr(value, "__dataclass_fields__"):
                    for sub_field in value.__dataclass_fields__:
                        params[f"{field_name}.{sub_field}"] = str(
                            getattr(value, sub_field)
                        )
                else:
                    params[field_name] = str(value)
            self.log_params(params)

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run: %s", self._run_id)
        self._run_id = None

