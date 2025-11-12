"""
Unit tests for data loading and preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from emipredict.data.loader import (
    load_data,
    validate_data,
    handle_missing_values,
    handle_outliers,
    encode_categorical_features,
    split_data,
    preprocess_data
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    return pd.DataFrame({
        'age': [25, 30, 35, np.nan, 45],
        'monthly_salary': [50000, 60000, 70000, 80000, 90000],
        'credit_score': [650, 700, 750, 800, 850],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'education': ['Graduate', 'Professional', 'Graduate', 'High School', 'Professional'],
        'existing_loans': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'emi_eligibility': ['Eligible', 'Eligible', 'Not_Eligible', 'Eligible', 'Not_Eligible'],
        'max_monthly_emi': [10000, 15000, 5000, 12000, 8000]
    })


def test_validate_data_success(sample_data):
    """Test data validation with valid data."""
    # Should not raise any exception
    validate_data(sample_data)


def test_validate_data_empty():
    """Test data validation with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_data(empty_df)


def test_validate_data_missing_target():
    """Test data validation with missing target column."""
    df = pd.DataFrame({
        'age': [25, 30],
        'salary': [50000, 60000]
    })
    
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df)


def test_handle_missing_values(sample_data):
    """Test missing value imputation."""
    # Add some missing values
    df = sample_data.copy()
    df.loc[0, 'credit_score'] = np.nan
    
    # Handle missing values
    df_clean = handle_missing_values(df)
    
    # Check no missing values in imputed columns
    assert not df_clean['age'].isnull().any()
    assert not df_clean['credit_score'].isnull().any()


def test_handle_outliers_cap(sample_data):
    """Test outlier handling with capping method."""
    df = sample_data.copy()
    
    # Add outlier
    df.loc[0, 'monthly_salary'] = 1000000
    
    # Handle outliers
    df_clean = handle_outliers(df, method='cap')
    
    # Check outlier is capped
    assert df_clean['monthly_salary'].max() < 1000000


def test_handle_outliers_none(sample_data):
    """Test outlier handling with no action."""
    df = sample_data.copy()
    
    df_clean = handle_outliers(df, method='none')
    
    # Should return unchanged DataFrame
    pd.testing.assert_frame_equal(df, df_clean)


def test_encode_categorical_features_label(sample_data):
    """Test label encoding of categorical features."""
    df = sample_data.copy()
    
    df_encoded, encoders = encode_categorical_features(df, encoding_method='label')
    
    # Check categorical columns are now numeric
    assert df_encoded['gender'].dtype in [np.int32, np.int64]
    assert df_encoded['education'].dtype in [np.int32, np.int64]
    
    # Check encoders were created
    assert 'gender' in encoders
    assert 'education' in encoders


def test_encode_categorical_features_onehot(sample_data):
    """Test one-hot encoding of categorical features."""
    df = sample_data.copy()
    
    original_cols = len(df.columns)
    df_encoded, encoders = encode_categorical_features(df, encoding_method='onehot')
    
    # Check new columns were created
    assert len(df_encoded.columns) > original_cols
    
    # Check onehot columns info is stored
    assert 'onehot_columns' in encoders


def test_split_data(sample_data):
    """Test train/val/test splitting."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        sample_data,
        target_column='emi_eligibility',
        test_size=0.4,
        val_size=0.5,
        stratify=False,
        random_state=42
    )
    
    # Check sizes
    total_samples = len(sample_data)
    assert len(X_train) + len(X_val) + len(X_test) == total_samples
    
    # Check target column not in features
    assert 'emi_eligibility' not in X_train.columns
    assert 'emi_eligibility' not in X_val.columns
    assert 'emi_eligibility' not in X_test.columns
    
    # Check y contains target values
    assert all(y_train.isin(['Eligible', 'Not_Eligible']))


def test_preprocess_data(sample_data):
    """Test complete preprocessing pipeline."""
    df = sample_data.copy()
    
    # Add missing value
    df.loc[0, 'age'] = np.nan
    
    df_processed, metadata = preprocess_data(
        df,
        handle_missing=True,
        handle_outliers_method='cap',
        encode_categorical=True
    )
    
    # Check no missing values
    non_target_cols = [col for col in df_processed.columns 
                       if col not in ['emi_eligibility', 'max_monthly_emi']]
    assert not df_processed[non_target_cols].isnull().any().any()
    
    # Check metadata
    assert 'missing_handled' in metadata
    assert 'outliers_handled' in metadata
    assert 'encoders' in metadata


@pytest.mark.parametrize("test_size,val_size", [
    (0.3, 0.5),
    (0.2, 0.4),
    (0.4, 0.6)
])
def test_split_data_sizes(sample_data, test_size, val_size):
    """Test split_data with different size parameters."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        sample_data,
        target_column='emi_eligibility',
        test_size=test_size,
        val_size=val_size,
        stratify=False,
        random_state=42
    )
    
    # Check total samples preserved
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sample_data)

