"""
Utility functions for EMI-Predict AI.

This module provides helper functions for:
- Logging setup
- Data validation
- Model persistence
- Plotting utilities
- Common calculations
"""

import logging
import joblib
from pathlib import Path
from typing import Any, Optional, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from emipredict.config.settings import Config


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file
        
    Example:
        >>> setup_logging(level='INFO', log_file=Path('logs/app.log'))
    """
    if level is None:
        level = Config.LOG_LEVEL
    
    if log_file is None:
        log_file = Config.LOG_FILE
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    handlers = []
    
    # Console handler
    if Config.ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(Config.LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # File handler
    if Config.ENABLE_FILE_LOGGING:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(Config.LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    
    logging.info("Logging configured successfully")


def validate_input_data(
    data: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> None:
    """
    Validate input data structure and content.
    
    Args:
        data: DataFrame to validate
        expected_columns: List of expected column names
        min_rows: Minimum number of rows required
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> validate_input_data(df, expected_columns=['age', 'salary'])
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if len(data) < min_rows:
        raise ValueError(
            f"DataFrame has only {len(data)} rows. "
            f"Minimum required: {min_rows}"
        )
    
    if expected_columns:
        missing = set(expected_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Check for all NaN columns
    all_nan_cols = data.columns[data.isnull().all()].tolist()
    if all_nan_cols:
        raise ValueError(f"Columns with all NaN values: {all_nan_cols}")


def save_model(
    model: Any,
    file_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model to disk using joblib.
    
    Args:
        model: Model object to save
        file_path: Path to save the model
        metadata: Optional metadata to save with model
        
    Example:
        >>> save_model(trained_model, Path('models/xgboost_clf.pkl'))
    """
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    save_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    # Save using joblib
    joblib.dump(save_data, file_path)
    logging.info(f"Model saved to {file_path}")


def load_model(file_path: Path) -> tuple:
    """
    Load model from disk.
    
    Args:
        file_path: Path to model file
        
    Returns:
        Tuple of (model, metadata)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        
    Example:
        >>> model, metadata = load_model(Path('models/xgboost_clf.pkl'))
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    save_data = joblib.load(file_path)
    
    if isinstance(save_data, dict):
        model = save_data.get('model')
        metadata = save_data.get('metadata', {})
    else:
        # Old format - just the model
        model = save_data
        metadata = {}
    
    logging.info(f"Model loaded from {file_path}")
    return model, metadata


def save_plot(
    fig: plt.Figure,
    filename: str,
    directory: Optional[Path] = None,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> Path:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of file to save
        directory: Directory to save to (default: Config.PLOTS_DIR)
        dpi: DPI for saved image
        bbox_inches: Bounding box setting
        
    Returns:
        Path to saved file
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> save_plot(fig, 'my_plot.png')
    """
    if directory is None:
        directory = Config.PLOTS_DIR
    
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename
    
    fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
    logging.info(f"Plot saved to {file_path}")
    
    return file_path


def calculate_debt_to_income_ratio(
    monthly_debt: float,
    monthly_income: float
) -> float:
    """
    Calculate debt-to-income ratio.
    
    Args:
        monthly_debt: Total monthly debt payments
        monthly_income: Monthly gross income
        
    Returns:
        DTI ratio (0-1 scale)
        
    Example:
        >>> dti = calculate_debt_to_income_ratio(2000, 5000)
        >>> print(f"DTI: {dti:.2%}")
    """
    if monthly_income == 0:
        return 0.0
    return monthly_debt / monthly_income


def calculate_expense_ratio(
    monthly_expenses: float,
    monthly_income: float
) -> float:
    """
    Calculate expense-to-income ratio.
    
    Args:
        monthly_expenses: Total monthly expenses
        monthly_income: Monthly gross income
        
    Returns:
        Expense ratio (0-1 scale)
    """
    if monthly_income == 0:
        return 0.0
    return monthly_expenses / monthly_income


def calculate_savings_rate(
    monthly_income: float,
    monthly_expenses: float,
    monthly_debt: float
) -> float:
    """
    Calculate savings rate.
    
    Args:
        monthly_income: Monthly gross income
        monthly_expenses: Total monthly expenses
        monthly_debt: Total monthly debt payments
        
    Returns:
        Savings rate (0-1 scale)
    """
    if monthly_income == 0:
        return 0.0
    
    savings = monthly_income - monthly_expenses - monthly_debt
    return max(0, savings / monthly_income)


def format_currency(amount: float, currency: str = "INR") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        Formatted currency string
        
    Example:
        >>> format_currency(50000)
        '₹50,000'
    """
    if currency == "INR":
        return f"₹{amount:,.0f}"
    elif currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Create confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    return fig


def create_feature_importance_plot(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance"
) -> plt.Figure:
    """
    Create feature importance bar plot.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to show
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def create_distribution_plot(
    data: pd.Series,
    title: str = "Distribution",
    bins: int = 50
) -> plt.Figure:
    """
    Create distribution plot with histogram and KDE.
    
    Args:
        data: Data to plot
        title: Plot title
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', density=True)
    
    # Add KDE
    from scipy import stats
    kde = stats.gaussian_kde(data.dropna())
    x_range = np.linspace(data.min(), data.max(), 100)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax.set_title(title)
    ax.set_xlabel(data.name if hasattr(data, 'name') else 'Value')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    
    return fig


def get_classification_metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Supports both binary and multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Add macro averages for multi-class
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ROC-AUC for multi-class (one-vs-rest)
    if y_pred_proba is not None:
        try:
            n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 2
            if n_classes > 2:
                # Multi-class ROC-AUC (one-vs-rest)
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
            else:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            logging.warning(f"Could not calculate ROC-AUC: {str(e)}")
    
    return metrics


def get_regression_metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score
    )
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics


# Initialize logging on import
setup_logging()

