"""
MLflow utilities module for EMI-Predict AI.

This module handles:
- Experiment tracking and management
- Parameter and metric logging
- Artifact storage
- Model registry operations
- Run comparison and analysis
"""

from emipredict.mlflow_utils.tracker import (
    start_experiment,
    log_model_metrics,
    log_model_artifacts,
    register_model,
    get_best_run,
)

from emipredict.mlflow_utils.comparison import (
    ModelComparator,
    compare_classification_regression,
    generate_model_selection_report,
)

__all__ = [
    "start_experiment",
    "log_model_metrics",
    "log_model_artifacts",
    "register_model",
    "get_best_run",
    "ModelComparator",
    "compare_classification_regression",
    "generate_model_selection_report",
]

