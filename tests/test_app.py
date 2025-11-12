"""
Unit tests for Streamlit application components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


def test_imports():
    """Test that all app modules can be imported."""
    # Main app
    from emipredict.app import main
    
    # Pages should be importable as modules
    # (In actual Streamlit, they're run as scripts, but we can test imports)
    assert main is not None


def test_config_initialization():
    """Test configuration initialization."""
    from emipredict.config.settings import Config
    
    # Check basic config attributes exist
    assert hasattr(Config, 'PROJECT_ROOT')
    assert hasattr(Config, 'DATA_PATH')
    assert hasattr(Config, 'MODELS_DIR')
    assert hasattr(Config, 'RANDOM_STATE')


def test_config_ensure_directories():
    """Test directory creation."""
    from emipredict.config.settings import Config
    
    # Should not raise any errors
    Config.ensure_directories()
    
    # Check directories exist
    assert Config.MODELS_DIR.exists()
    assert Config.LOGS_DIR.exists()


def test_load_model_helper():
    """Test model loading helper function."""
    from emipredict.utils.helpers import save_model, load_model
    from sklearn.linear_model import LogisticRegression
    from pathlib import Path
    import tempfile
    
    # Create temp model
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        save_model(model, tmp_path, metadata={'test': 'value'})
        
        # Load model
        loaded_model, metadata = load_model(tmp_path)
        
        # Check model works
        assert loaded_model is not None
        predictions = loaded_model.predict(X)
        assert len(predictions) == len(X)
        
        # Check metadata
        assert metadata['test'] == 'value'
        
    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()


def test_save_plot_helper():
    """Test plot saving helper function."""
    from emipredict.utils.helpers import save_plot
    import matplotlib.pyplot as plt
    import tempfile
    from pathlib import Path
    
    # Create simple plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        saved_path = save_plot(fig, 'test_plot.png', directory=tmp_path)
        
        # Check file was created
        assert saved_path.exists()
        assert saved_path.name == 'test_plot.png'
    
    plt.close(fig)


def test_helpers_validation():
    """Test input validation helpers."""
    from emipredict.utils.helpers import validate_input_data
    
    # Valid DataFrame
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })
    
    # Should not raise
    validate_input_data(df, expected_columns=['age', 'salary'])
    
    # Missing column should raise
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input_data(df, expected_columns=['age', 'salary', 'missing_col'])
    
    # Too few rows should raise
    with pytest.raises(ValueError, match="Minimum required"):
        validate_input_data(df, min_rows=10)


def test_currency_formatting():
    """Test currency formatting helper."""
    from emipredict.utils.helpers import format_currency
    
    # Test INR formatting
    assert format_currency(50000) == "₹50,000"
    assert format_currency(1234567) == "₹1,234,567"
    
    # Test USD formatting
    assert format_currency(50000, currency='USD') == "$50,000.00"


def test_financial_calculations():
    """Test financial calculation helpers."""
    from emipredict.utils.helpers import (
        calculate_debt_to_income_ratio,
        calculate_expense_ratio,
        calculate_savings_rate
    )
    
    # DTI calculation
    dti = calculate_debt_to_income_ratio(10000, 50000)
    assert dti == 0.2  # 20%
    
    # Expense ratio
    expense_ratio = calculate_expense_ratio(20000, 50000)
    assert expense_ratio == 0.4  # 40%
    
    # Savings rate
    savings_rate = calculate_savings_rate(50000, 20000, 10000)
    assert savings_rate == 0.4  # 40%
    
    # Zero income edge case
    dti_zero = calculate_debt_to_income_ratio(10000, 0)
    assert dti_zero == 0.0


def test_metrics_summary():
    """Test metrics summary helpers."""
    from emipredict.utils.helpers import (
        get_classification_metrics_summary,
        get_regression_metrics_summary
    )
    
    # Classification metrics
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    metrics = get_classification_metrics_summary(y_true, y_pred)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    
    # Regression metrics
    y_true_reg = np.array([10000, 15000, 20000, 25000, 30000])
    y_pred_reg = np.array([11000, 14000, 21000, 24000, 31000])
    
    metrics_reg = get_regression_metrics_summary(y_true_reg, y_pred_reg)
    
    assert 'rmse' in metrics_reg
    assert 'mae' in metrics_reg
    assert 'r2_score' in metrics_reg
    assert metrics_reg['rmse'] > 0


@pytest.mark.parametrize("salary,emi,expected", [
    (50000, 10000, 0.2),
    (100000, 30000, 0.3),
    (75000, 0, 0.0),
])
def test_dti_various_inputs(salary, emi, expected):
    """Test DTI calculation with various inputs."""
    from emipredict.utils.helpers import calculate_debt_to_income_ratio
    
    dti = calculate_debt_to_income_ratio(emi, salary)
    assert np.isclose(dti, expected, atol=0.01)

