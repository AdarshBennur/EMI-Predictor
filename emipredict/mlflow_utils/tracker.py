"""
MLflow tracking utilities for EMI-Predict AI.

This module provides functions for:
- Experiment management
- Parameter and metric logging
- Artifact storage
- Model registry operations
- Run comparison and analysis
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from emipredict.config.settings import Config

# Setup logging
logger = logging.getLogger(__name__)


def initialize_mlflow() -> None:
    """
    Initialize MLflow tracking.
    
    Sets the tracking URI and configures MLflow for the project.
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to: {Config.MLFLOW_TRACKING_URI}")


def start_experiment(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> mlflow.ActiveRun:
    """
    Start an MLflow experiment run.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Name for this specific run (optional)
        tags: Dictionary of tags to add to the run (optional)
        
    Returns:
        MLflow ActiveRun context manager
        
    Example:
        >>> with start_experiment("my_experiment", "run_1") as run:
        ...     mlflow.log_param("alpha", 0.5)
    """
    initialize_mlflow()
    
    # Set experiment (creates if doesn't exist)
    mlflow.set_experiment(experiment_name)
    
    # Start run
    run = mlflow.start_run(run_name=run_name)
    
    # Log tags
    if tags:
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    # Log default tags
    mlflow.set_tag("project", "EMI-Predict-AI")
    mlflow.set_tag("version", "1.0.0")
    
    logger.info(f"Started experiment: {experiment_name}, run: {run.info.run_id}")
    
    return run


def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameters to log
        
    Example:
        >>> log_params({"learning_rate": 0.1, "max_depth": 6})
    """
    for key, value in params.items():
        # MLflow has character limits, truncate if needed
        if isinstance(value, str) and len(value) > 500:
            value = value[:500]
        mlflow.log_param(key, value)
    
    logger.info(f"Logged {len(params)} parameters")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number for tracking metrics over time
        
    Example:
        >>> log_metrics({"accuracy": 0.92, "f1_score": 0.90})
    """
    for key, value in metrics.items():
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)
    
    logger.info(f"Logged {len(metrics)} metrics")


def log_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    task: str = "classification",
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate and log model performance metrics.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_pred_proba: Predicted probabilities (for classification)
        task: Task type ('classification' or 'regression')
        prefix: Prefix for metric names (e.g., 'train_', 'val_')
        
    Returns:
        Dictionary of calculated metrics
        
    Example:
        >>> metrics = log_model_metrics(y_true, y_pred, task='classification')
    """
    from emipredict.utils.helpers import (
        get_classification_metrics_summary,
        get_regression_metrics_summary
    )
    
    if task == "classification":
        metrics = get_classification_metrics_summary(y_true, y_pred, y_pred_proba)
    else:  # regression
        metrics = get_regression_metrics_summary(y_true, y_pred)
    
    # Add prefix to metric names
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    
    # Log to MLflow
    log_metrics(metrics)
    
    return metrics


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    filename: str = "confusion_matrix.png"
) -> None:
    """
    Create and log confusion matrix plot.
    
    Supports both binary and multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (e.g., ['Eligible', 'High_Risk', 'Not_Eligible'])
        filename: Name for saved plot file
        
    Example:
        >>> log_confusion_matrix(y_true, y_pred, labels=['Eligible', 'High_Risk', 'Not_Eligible'])
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine figure size based on number of classes
    n_classes = cm.shape[0]
    figsize = (8, 6) if n_classes == 2 else (10, 8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn for better visualization
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels if labels else range(n_classes),
        yticklabels=labels if labels else range(n_classes),
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_title(f'Confusion Matrix ({n_classes}-class Classification)', fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Rotate labels if multi-class
    if n_classes > 2:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save and log
    temp_path = Path(filename)
    fig.savefig(temp_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(temp_path))
    temp_path.unlink()  # Delete temp file
    
    plt.close(fig)
    logger.info(f"Logged confusion matrix ({n_classes}-class): {filename}")


def log_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    filename: str = "feature_importance.png"
) -> None:
    """
    Create and log feature importance plot.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to display
        filename: Name for saved plot file
        
    Example:
        >>> log_feature_importance(model, X_train.columns.tolist())
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    from emipredict.utils.helpers import create_feature_importance_plot
    
    fig = create_feature_importance_plot(
        feature_names,
        model.feature_importances_,
        top_n=top_n
    )
    
    # Save and log
    temp_path = Path(filename)
    fig.savefig(temp_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(temp_path))
    temp_path.unlink()  # Delete temp file
    
    plt.close(fig)
    logger.info(f"Logged feature importance: {filename}")


def log_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "residual_plot.png"
) -> None:
    """
    Create and log residual plot for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        filename: Name for saved plot file
        
    Example:
        >>> log_residual_plot(y_true, y_pred)
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and log
    temp_path = Path(filename)
    fig.savefig(temp_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(temp_path))
    temp_path.unlink()
    
    plt.close(fig)
    logger.info(f"Logged residual plot: {filename}")


def log_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "prediction_scatter.png"
) -> None:
    """
    Create and log prediction vs actual scatter plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        filename: Name for saved plot file
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predicted vs Actual Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and log
    temp_path = Path(filename)
    fig.savefig(temp_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(temp_path))
    temp_path.unlink()
    
    plt.close(fig)
    logger.info(f"Logged prediction scatter plot: {filename}")


def log_model_artifacts(
    model: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    task: str = "classification",
    class_names: Optional[List[str]] = None
) -> None:
    """
    Log comprehensive model artifacts.
    
    Args:
        model: Trained model
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_pred_proba: Predicted probabilities (for classification)
        feature_names: List of feature names
        task: Task type ('classification' or 'regression')
        class_names: Names of classes for classification
        
    Example:
        >>> log_model_artifacts(model, y_true, y_pred, feature_names=X.columns)
    """
    logger.info("Logging model artifacts")
    
    if task == "classification":
        # Confusion matrix with class names
        log_confusion_matrix(y_true, y_pred, labels=class_names)
        
        # ROC curve (if probabilities available)
        if y_pred_proba is not None:
            try:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                temp_path = Path("roc_curve.png")
                fig.savefig(temp_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(str(temp_path))
                temp_path.unlink()
                plt.close(fig)
                
                logger.info("Logged ROC curve")
            except Exception as e:
                logger.warning(f"Could not log ROC curve: {str(e)}")
    
    else:  # regression
        # Residual plot
        log_residual_plot(y_true, y_pred)
        
        # Prediction scatter
        log_prediction_scatter(y_true, y_pred)
    
    # Feature importance (if available)
    if feature_names is not None:
        try:
            log_feature_importance(model, feature_names)
        except Exception as e:
            logger.warning(f"Could not log feature importance: {str(e)}")


def register_model(
    model: Any,
    model_name: str,
    run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> None:
    """
    Register model in MLflow model registry.
    
    Args:
        model: Trained model to register
        model_name: Name for registered model
        run_id: MLflow run ID (uses current run if None)
        tags: Tags to add to registered model
        
    Example:
        >>> register_model(model, "emi_eligibility_classifier")
    """
    if run_id is None:
        run_id = mlflow.active_run().info.run_id
    
    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name
    )
    
    logger.info(f"Registered model: {model_name} (run_id: {run_id})")


def get_best_run(
    experiment_name: str,
    metric_name: str,
    ascending: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to optimize
        ascending: If True, lower is better (for RMSE, MAE)
        
    Returns:
        Tuple of (run_id, run_data)
        
    Example:
        >>> best_run_id, data = get_best_run("classification", "metrics.accuracy")
    """
    initialize_mlflow()
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    
    # Search runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment: {experiment_name}")
    
    best_run = runs.iloc[0]
    run_id = best_run.run_id
    
    logger.info(f"Best run: {run_id}, {metric_name}: {best_run[f'metrics.{metric_name}']}")
    
    return run_id, best_run.to_dict()


def compare_runs(
    experiment_name: str,
    metrics: List[str],
    top_n: int = 5
) -> pd.DataFrame:
    """
    Compare top N runs from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metrics: List of metrics to compare
        top_n: Number of top runs to compare
        
    Returns:
        DataFrame with run comparison
        
    Example:
        >>> comparison = compare_runs(
        ...     "classification",
        ...     ["accuracy", "precision", "recall"],
        ...     top_n=5
        ... )
    """
    initialize_mlflow()
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    
    # Search runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=top_n
    )
    
    # Select relevant columns
    columns = ['run_id', 'start_time', 'status']
    metric_cols = [f'metrics.{m}' for m in metrics]
    columns.extend(metric_cols)
    
    # Add parameter columns if they exist
    param_cols = [col for col in runs.columns if col.startswith('params.')]
    columns.extend(param_cols)
    
    # Filter columns that exist
    columns = [col for col in columns if col in runs.columns]
    
    comparison_df = runs[columns].copy()
    
    logger.info(f"Compared {len(comparison_df)} runs from {experiment_name}")
    
    return comparison_df


def load_model_from_run(run_id: str, artifact_path: str = "model") -> Any:
    """
    Load a model from a specific MLflow run.
    
    Args:
        run_id: MLflow run ID
        artifact_path: Path to model artifact
        
    Returns:
        Loaded model
        
    Example:
        >>> model = load_model_from_run("abc123", "model")
    """
    initialize_mlflow()
    
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.sklearn.load_model(model_uri)
    
    logger.info(f"Loaded model from run: {run_id}")
    
    return model


def log_dataset_info(df: pd.DataFrame, name: str = "dataset") -> None:
    """
    Log dataset information as parameters and artifacts.
    
    Args:
        df: Dataset DataFrame
        name: Name prefix for logged information
        
    Example:
        >>> log_dataset_info(df, "training_data")
    """
    # Log basic stats as parameters
    mlflow.log_param(f"{name}_samples", len(df))
    mlflow.log_param(f"{name}_features", len(df.columns))
    
    # Create and log dataset summary
    summary = df.describe().to_csv()
    temp_path = Path(f"{name}_summary.csv")
    temp_path.write_text(summary)
    mlflow.log_artifact(str(temp_path))
    temp_path.unlink()
    
    logger.info(f"Logged dataset info: {name}")


def end_run() -> None:
    """
    End the current MLflow run.
    
    Example:
        >>> start_experiment("my_experiment")
        >>> # ... log metrics and parameters ...
        >>> end_run()
    """
    mlflow.end_run()
    logger.info("MLflow run ended")

