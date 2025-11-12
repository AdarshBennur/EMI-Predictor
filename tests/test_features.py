"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from emipredict.features.engineering import (
    create_financial_features,
    create_features,
    select_features,
    calculate_correlation_with_target,
    remove_correlated_features
)


@pytest.fixture
def sample_financial_data():
    """Create sample financial dataset."""
    return pd.DataFrame({
        'monthly_salary': [50000, 60000, 70000, 80000, 90000],
        'current_emi_amount': [10000, 12000, 8000, 15000, 20000],
        'monthly_rent': [15000, 0, 10000, 20000, 0],
        'groceries_utilities': [8000, 10000, 12000, 9000, 11000],
        'travel_expenses': [3000, 4000, 5000, 3500, 4500],
        'other_monthly_expenses': [5000, 6000, 7000, 5500, 6500],
        'school_fees': [0, 5000, 0, 10000, 0],
        'college_fees': [0, 0, 15000, 0, 20000],
        'bank_balance': [100000, 150000, 200000, 250000, 300000],
        'emergency_fund': [50000, 75000, 100000, 125000, 150000],
        'credit_score': [650, 700, 750, 800, 850],
        'age': [25, 30, 35, 40, 45],
        'years_of_employment': [2, 5, 8, 12, 15],
        'dependents': [0, 1, 2, 3, 4],
        'family_size': [2, 3, 4, 5, 6],
        'requested_amount': [500000, 600000, 700000, 800000, 900000],
        'requested_tenure': [60, 72, 84, 96, 108]
    })


def test_create_financial_features(sample_financial_data):
    """Test creation of derived financial features."""
    df = sample_financial_data.copy()
    
    df_engineered = create_financial_features(df)
    
    # Check new features were created
    assert 'total_monthly_expenses' in df_engineered.columns
    assert 'debt_to_income_ratio' in df_engineered.columns
    assert 'expense_ratio' in df_engineered.columns
    assert 'savings_rate' in df_engineered.columns
    assert 'financial_stress_index' in df_engineered.columns
    
    # Check values are reasonable
    assert all(df_engineered['debt_to_income_ratio'] >= 0)
    assert all(df_engineered['debt_to_income_ratio'] <= 1)
    assert all(df_engineered['expense_ratio'] >= 0)


def test_create_financial_features_no_missing(sample_financial_data):
    """Test that feature engineering doesn't introduce missing values."""
    df = sample_financial_data.copy()
    
    df_engineered = create_financial_features(df)
    
    # Check for NaN values in derived features
    derived_features = [
        'total_monthly_expenses',
        'debt_to_income_ratio',
        'expense_ratio',
        'savings_rate'
    ]
    
    for feat in derived_features:
        if feat in df_engineered.columns:
            assert not df_engineered[feat].isnull().any(), f"{feat} has NaN values"


def test_create_features_complete(sample_financial_data):
    """Test complete feature engineering pipeline."""
    df = sample_financial_data.copy()
    
    original_features = len(df.columns)
    df_engineered = create_features(df, include_financial=True)
    
    # Check new features were added
    assert len(df_engineered.columns) > original_features


def test_select_features_classification(sample_financial_data):
    """Test feature selection for classification."""
    df = sample_financial_data.copy()
    
    # Add target
    y = pd.Series([0, 1, 0, 1, 0], name='target')
    
    # Select top 5 features
    X_selected, selected_features, scores = select_features(
        df, y, task='classification', k=5, method='f_test'
    )
    
    # Check correct number of features selected
    assert X_selected.shape[1] == 5
    assert len(selected_features) == 5
    assert len(scores) == len(df.columns)


def test_select_features_regression(sample_financial_data):
    """Test feature selection for regression."""
    df = sample_financial_data.copy()
    
    # Add target
    y = pd.Series([10000, 15000, 20000, 25000, 30000], name='target')
    
    # Select top 5 features
    X_selected, selected_features, scores = select_features(
        df, y, task='regression', k=5, method='f_test'
    )
    
    # Check correct number of features selected
    assert X_selected.shape[1] == 5
    assert len(selected_features) == 5


def test_calculate_correlation_with_target(sample_financial_data):
    """Test correlation calculation with target."""
    df = sample_financial_data.copy()
    
    # Add target
    y = pd.Series([10000, 15000, 20000, 25000, 30000], name='target')
    
    corr_df = calculate_correlation_with_target(df, y, top_n=5)
    
    # Check result structure
    assert 'feature' in corr_df.columns
    assert 'correlation' in corr_df.columns
    assert 'abs_correlation' in corr_df.columns
    assert len(corr_df) == 5


def test_remove_correlated_features():
    """Test removal of highly correlated features."""
    # Create DataFrame with correlated features
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Highly correlated with feature1
        'feature3': [10, 20, 30, 40, 50],  # Not correlated
    })
    
    df_reduced, removed = remove_correlated_features(df, threshold=0.95)
    
    # Check one feature was removed
    assert len(df_reduced.columns) == 2
    assert len(removed) == 1


def test_debt_to_income_ratio_calculation(sample_financial_data):
    """Test DTI ratio calculation."""
    df = sample_financial_data.copy()
    
    df_engineered = create_financial_features(df)
    
    # Manually calculate DTI for first row
    expected_dti = df.loc[0, 'current_emi_amount'] / df.loc[0, 'monthly_salary']
    actual_dti = df_engineered.loc[0, 'debt_to_income_ratio']
    
    # Check calculation is correct
    assert np.isclose(expected_dti, actual_dti, atol=0.01)


def test_savings_rate_calculation(sample_financial_data):
    """Test savings rate calculation."""
    df = sample_financial_data.copy()
    
    df_engineered = create_financial_features(df)
    
    # Check savings rate is between -1 and 1
    assert all(df_engineered['savings_rate'] >= -1)
    assert all(df_engineered['savings_rate'] <= 1)


def test_boolean_indicators(sample_financial_data):
    """Test boolean indicator features."""
    df = sample_financial_data.copy()
    
    df_engineered = create_financial_features(df)
    
    # Check boolean features exist and are 0/1
    if 'has_rent_payment' in df_engineered.columns:
        assert set(df_engineered['has_rent_payment'].unique()).issubset({0, 1})
    
    if 'has_education_expenses' in df_engineered.columns:
        assert set(df_engineered['has_education_expenses'].unique()).issubset({0, 1})

