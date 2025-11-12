"""
Unit tests for model training modules.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def classification_data():
    """Create sample 3-class classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,  # 3-class classification
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split into train/val
    split_idx = int(0.8 * len(X_df))
    X_train = X_df[:split_idx]
    X_val = X_df[split_idx:]
    y_train = y_series[:split_idx]
    y_val = y_series[split_idx:]
    
    return X_train, X_val, y_train, y_val


@pytest.fixture
def regression_data():
    """Create sample regression dataset."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # Scale target to be in EMI range
    y = (y - y.min()) / (y.max() - y.min()) * 40000 + 5000
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split into train/val
    split_idx = int(0.8 * len(X_df))
    X_train = X_df[:split_idx]
    X_val = X_df[split_idx:]
    y_train = y_series[:split_idx]
    y_val = y_series[split_idx:]
    
    return X_train, X_val, y_train, y_val


def test_train_logistic_regression(classification_data):
    """Test logistic regression training."""
    from emipredict.models.classification import train_logistic_regression
    
    X_train, X_val, y_train, y_val = classification_data
    
    model, metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val,
        hyperparameter_tuning=False
    )
    
    # Check model is trained
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Check metrics
    assert 'val_accuracy' in metrics
    assert 0 <= metrics['val_accuracy'] <= 1


def test_train_random_forest_classifier(classification_data):
    """Test random forest classifier training."""
    from emipredict.models.classification import train_random_forest_classifier
    
    X_train, X_val, y_train, y_val = classification_data
    
    model, metrics = train_random_forest_classifier(
        X_train, y_train, X_val, y_val,
        hyperparameter_tuning=False
    )
    
    # Check model is trained
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'feature_importances_')
    
    # Check metrics
    assert 'val_accuracy' in metrics
    assert 'val_precision' in metrics
    assert 'val_recall' in metrics
    assert 'val_f1_score' in metrics


def test_train_xgboost_classifier(classification_data):
    """Test XGBoost classifier training."""
    from emipredict.models.classification import train_xgboost_classifier
    
    X_train, X_val, y_train, y_val = classification_data
    
    model, metrics = train_xgboost_classifier(
        X_train, y_train, X_val, y_val,
        hyperparameter_tuning=False
    )
    
    # Check model is trained
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'feature_importances_')
    
    # Check metrics
    assert 'val_accuracy' in metrics
    assert metrics['val_accuracy'] > 0.5  # Should be better than random


def test_train_linear_regression(regression_data):
    """Test linear regression training."""
    from emipredict.models.regression import train_linear_regression
    
    X_train, X_val, y_train, y_val = regression_data
    
    model, metrics = train_linear_regression(
        X_train, y_train, X_val, y_val
    )
    
    # Check model is trained
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Check metrics
    assert 'val_rmse' in metrics
    assert 'val_mae' in metrics
    assert 'val_r2_score' in metrics
    assert metrics['val_rmse'] > 0


def test_train_random_forest_regressor(regression_data):
    """Test random forest regressor training."""
    from emipredict.models.regression import train_random_forest_regressor
    
    X_train, X_val, y_train, y_val = regression_data
    
    model, metrics = train_random_forest_regressor(
        X_train, y_train, X_val, y_val,
        hyperparameter_tuning=False
    )
    
    # Check model is trained
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'feature_importances_')
    
    # Check metrics
    assert 'val_rmse' in metrics
    assert 'val_mae' in metrics
    assert metrics['val_r2_score'] <= 1.0


def test_train_xgboost_regressor(regression_data):
    """Test XGBoost regressor training."""
    from emipredict.models.regression import train_xgboost_regressor
    
    X_train, X_val, y_train, y_val = regression_data
    
    model, metrics = train_xgboost_regressor(
        X_train, y_train, X_val, y_val,
        hyperparameter_tuning=False
    )
    
    # Check model is trained
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Check metrics are reasonable
    assert 'val_rmse' in metrics
    assert metrics['val_rmse'] > 0
    assert metrics['val_r2_score'] > 0  # Should have some predictive power


def test_model_prediction_shape(classification_data):
    """Test that model predictions have correct shape."""
    from emipredict.models.classification import train_logistic_regression
    
    X_train, X_val, y_train, y_val = classification_data
    
    model, _ = train_logistic_regression(
        X_train, y_train, X_val, y_val,
        hyperparameter_tuning=False
    )
    
    predictions = model.predict(X_val)
    
    # Check prediction shape matches input
    assert len(predictions) == len(X_val)
    
    # Check prediction values are 0, 1, or 2 (3-class)
    assert set(predictions).issubset({0, 1, 2})


def test_regression_prediction_positive(regression_data):
    """Test that regression predictions are in reasonable range."""
    from emipredict.models.regression import train_linear_regression
    
    X_train, X_val, y_train, y_val = regression_data
    
    model, _ = train_linear_regression(
        X_train, y_train, X_val, y_val
    )
    
    predictions = model.predict(X_val)
    
    # Check prediction shape
    assert len(predictions) == len(X_val)
    
    # Check predictions are numerical
    assert np.all(np.isfinite(predictions))

