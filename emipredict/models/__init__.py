"""
Machine learning models module for EMI-Predict AI.

This module handles:
- Classification model training (EMI eligibility)
- Regression model training (EMI amount prediction)
- Hyperparameter tuning
- Model evaluation and comparison
- Model persistence
"""

from emipredict.models.classification import (
    train_classification_models,
    train_logistic_regression,
    train_random_forest_classifier,
    train_xgboost_classifier,
)
from emipredict.models.regression import (
    train_regression_models,
    train_linear_regression,
    train_random_forest_regressor,
    train_xgboost_regressor,
)

__all__ = [
    "train_classification_models",
    "train_logistic_regression",
    "train_random_forest_classifier",
    "train_xgboost_classifier",
    "train_regression_models",
    "train_linear_regression",
    "train_random_forest_regressor",
    "train_xgboost_regressor",
]

