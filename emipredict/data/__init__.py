"""
Data loading and preprocessing module for EMI-Predict AI.

This module handles:
- Dataset loading and validation
- Missing value imputation
- Outlier detection and treatment
- Categorical encoding
- Feature scaling
- Train/validation/test splitting
"""

from emipredict.data.loader import (
    load_data,
    preprocess_data,
    split_data,
    load_and_preprocess_data,
)

__all__ = [
    "load_data",
    "preprocess_data",
    "split_data",
    "load_and_preprocess_data",
]

