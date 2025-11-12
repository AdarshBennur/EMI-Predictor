"""
Feature engineering module for EMI-Predict AI.

This module handles:
- Derived feature creation (DTI ratio, expense ratio, etc.)
- Feature selection and importance analysis
- Correlation analysis
- Feature transformations
"""

from emipredict.features.engineering import (
    create_features,
    create_financial_features,
    select_features,
)

__all__ = [
    "create_features",
    "create_financial_features",
    "select_features",
]

