"""
EMI-Predict AI - Intelligent Financial Risk Assessment Platform

This package provides machine learning models and utilities for:
- EMI eligibility classification
- EMI amount prediction
- Model experiment tracking with MLflow
- Web-based prediction interface with Streamlit

Author: EMI-Predict AI Team
License: [Specify License]
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "EMI-Predict AI Team"

# Expose key components for easier imports
from emipredict.config.settings import Config

__all__ = [
    "Config",
    "__version__",
    "__author__",
]

