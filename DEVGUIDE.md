# EMI-Predict AI - Development Guidelines

This document outlines coding standards, best practices, and guidelines for contributing to the EMI-Predict AI project.

## Table of Contents

1. [Code Style](#code-style)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Testing Guidelines](#testing-guidelines)
5. [Documentation Standards](#documentation-standards)
6. [Git Workflow](#git-workflow)
7. [MLflow Best Practices](#mlflow-best-practices)
8. [Performance Guidelines](#performance-guidelines)
9. [Security Guidelines](#security-guidelines)

## Code Style

### Python Style Guide

Follow **PEP 8** with the following specifics:

#### Line Length
- Maximum line length: **88 characters** (Black formatter default)
- Docstrings and comments: **72 characters**

#### Naming Conventions

```python
# Modules and packages: lowercase with underscores
emipredict/data/loader.py

# Classes: PascalCase
class EMIPredictor:
    pass

# Functions and variables: lowercase with underscores
def calculate_debt_to_income_ratio():
    monthly_salary = 50000
    return ratio

# Constants: UPPERCASE with underscores
MAX_LOAN_AMOUNT = 5000000
DEFAULT_RANDOM_STATE = 42

# Private methods/variables: leading underscore
def _internal_calculation():
    _temp_value = 10
```

#### Imports

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local application imports
from emipredict.config.settings import Config
from emipredict.utils.helpers import validate_data
```

### Type Hints

**Always use type hints** for function parameters and return values:

```python
from typing import Tuple, Dict, List, Optional, Union
import pandas as pd
import numpy as np

def load_data(
    file_path: str,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        nrows: Number of rows to load (optional)
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(file_path, nrows=nrows)

def preprocess_features(
    X: pd.DataFrame,
    feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess features for model training.
    
    Args:
        X: Input DataFrame
        feature_names: List of feature names to use
        
    Returns:
        Tuple of (processed array, selected features)
    """
    # Processing logic
    return X_processed, selected_features
```

### Docstrings

Use **Google-style docstrings**:

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    hyperparameters: Optional[Dict] = None
) -> Tuple[object, Dict[str, float]]:
    """
    Train a machine learning model with given data.
    
    This function trains a model using the specified algorithm and 
    hyperparameters, then evaluates it on validation data.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        model_type: Type of model ('xgboost', 'random_forest', 'logistic')
        hyperparameters: Dictionary of hyperparameters (optional)
        
    Returns:
        Tuple containing:
            - Trained model object
            - Dictionary of evaluation metrics
            
    Raises:
        ValueError: If model_type is not supported
        TypeError: If X_train is not a DataFrame
        
    Example:
        >>> X_train, y_train = load_training_data()
        >>> model, metrics = train_model(X_train, y_train, 'xgboost')
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
        Accuracy: 0.92
        
    Note:
        This function logs all experiments to MLflow for tracking.
    """
    if model_type not in ['xgboost', 'random_forest', 'logistic']:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Training logic here
    return trained_model, metrics
```

### Code Formatting

Use **Black** for automatic code formatting:

```bash
# Install Black
pip install black

# Format entire project
black .

# Format specific file
black emipredict/data/loader.py

# Check without making changes
black --check .
```

### Linting

Use **Flake8** for linting:

```bash
# Install Flake8
pip install flake8

# Run linter
flake8 emipredict/

# Configuration in .flake8 or setup.cfg
# [flake8]
# max-line-length = 88
# extend-ignore = E203, W503
# exclude = .git, __pycache__, venv
```

## Project Structure

### Module Organization

```python
# Each module should have clear responsibilities

# emipredict/data/loader.py
# - Data loading
# - Data validation
# - Basic preprocessing

# emipredict/features/engineering.py
# - Feature creation
# - Feature selection
# - Feature transformations

# emipredict/models/classification.py
# - Classification model training
# - Hyperparameter tuning
# - Model evaluation

# emipredict/utils/helpers.py
# - Shared utility functions
# - Logging helpers
# - Validation functions
```

### Configuration Management

Store all configuration in `config/settings.py`:

```python
# emipredict/config/settings.py
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration management."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_PATH = PROJECT_ROOT / os.getenv('DATA_PATH', 'data/emi_prediction_dataset.csv')
    MODELS_DIR = PROJECT_ROOT / 'models'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # Data processing
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    TEST_SIZE = float(os.getenv('TRAIN_TEST_SPLIT_RATIO', 0.3))
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    
    # Model hyperparameters
    XGBOOST_PARAMS: Dict[str, Any] = {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
```

## Development Workflow

### 1. Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/new-model-algorithm

# 2. Develop with incremental commits
git add emipredict/models/new_algorithm.py
git commit -m "Add new algorithm implementation"

# 3. Write tests
git add tests/test_new_algorithm.py
git commit -m "Add tests for new algorithm"

# 4. Update documentation
git add ARCHITECTURE.md
git commit -m "Update docs with new algorithm"

# 5. Run tests and linting
pytest
flake8 emipredict/

# 6. Push and create pull request
git push origin feature/new-model-algorithm
```

### 2. Code Review Checklist

Before submitting code for review:

- [ ] All tests pass (`pytest`)
- [ ] Code coverage ≥85% for new code
- [ ] No linting errors (`flake8`)
- [ ] Type hints present for all functions
- [ ] Docstrings complete with examples
- [ ] No hardcoded values (use config)
- [ ] No sensitive data in code
- [ ] Documentation updated
- [ ] MLflow logging included (for ML code)
- [ ] Error handling implemented

### 3. Testing Your Changes

```python
# Test locally before committing
# 1. Run specific tests
pytest tests/test_data.py -v

# 2. Test with coverage
pytest --cov=emipredict.data

# 3. Integration test
python -m emipredict.models.classification

# 4. Manual testing in Streamlit
streamlit run emipredict/app/main.py
```

## Testing Guidelines

### Test Structure

```python
# tests/test_data.py
import pytest
import pandas as pd
import numpy as np
from emipredict.data.loader import load_data, preprocess_data

class TestDataLoader:
    """Test suite for data loading functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'age': [25, 30, 35],
            'monthly_salary': [50000, 60000, 70000],
            'credit_score': [650, 700, 750]
        })
    
    def test_load_data_success(self, tmp_path):
        """Test successful data loading."""
        # Arrange
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("age,salary\n25,50000\n30,60000")
        
        # Act
        df = load_data(str(csv_file))
        
        # Assert
        assert len(df) == 2
        assert list(df.columns) == ['age', 'salary']
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent.csv")
    
    def test_preprocess_data(self, sample_data):
        """Test data preprocessing."""
        processed = preprocess_data(sample_data)
        
        assert not processed.isnull().any().any()
        assert len(processed) == len(sample_data)
    
    @pytest.mark.parametrize("age,salary,expected", [
        (25, 50000, True),
        (18, 30000, True),
        (17, 50000, False),
        (25, 10000, False)
    ])
    def test_validate_input(self, age, salary, expected):
        """Test input validation with multiple cases."""
        result = validate_input(age, salary)
        assert result == expected
```

### Test Categories

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test module interactions
3. **Fixtures**: Reusable test data
4. **Parametrized Tests**: Multiple test cases
5. **Mock Tests**: Mock external dependencies

### Testing Best Practices

```python
# Use descriptive test names
def test_model_training_with_valid_data_succeeds():
    pass

# Use AAA pattern: Arrange, Act, Assert
def test_feature_engineering():
    # Arrange
    input_data = create_sample_data()
    
    # Act
    features = create_features(input_data)
    
    # Assert
    assert 'debt_to_income_ratio' in features.columns
    assert features['debt_to_income_ratio'].dtype == float

# Test edge cases
def test_handle_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        process_data(empty_df)

# Use fixtures for common setup
@pytest.fixture
def trained_model():
    X, y = load_sample_data()
    model = train_model(X, y)
    return model

def test_model_prediction(trained_model):
    prediction = trained_model.predict([[25, 50000, 700]])
    assert prediction in [0, 1]
```

## Documentation Standards

### Code Comments

```python
# Good: Explain WHY, not WHAT
# Calculate DTI because it's the primary indicator for loan approval
debt_to_income = total_debt / monthly_income

# Bad: States the obvious
# Calculate debt to income
debt_to_income = total_debt / monthly_income

# Good: Explain complex logic
# Use log transformation to handle right-skewed distribution
# This improves model performance by normalizing the feature
log_salary = np.log1p(salary)

# Good: Document assumptions
# Assuming monthly salary is gross (before tax)
# Assuming existing EMI includes all loans
total_obligations = existing_emi + new_emi
```

### README Updates

Update README.md when:
- Adding new features
- Changing installation steps
- Updating dependencies
- Modifying project structure

### Architecture Documentation

Update ARCHITECTURE.md when:
- Adding new models
- Changing data flow
- Adding new components
- Modifying system design

## Git Workflow

### Commit Messages

Follow **Conventional Commits**:

```bash
# Format: <type>(<scope>): <subject>

# Types:
# feat: New feature
# fix: Bug fix
# docs: Documentation
# style: Formatting
# refactor: Code restructuring
# test: Adding tests
# chore: Maintenance

# Examples:
git commit -m "feat(models): add XGBoost regression model"
git commit -m "fix(data): handle missing values in credit_score"
git commit -m "docs(architecture): update model training flow diagram"
git commit -m "test(features): add tests for feature engineering"
git commit -m "refactor(utils): simplify logging configuration"
```

### Branch Naming

```bash
# Feature branches
feature/add-gradient-boosting
feature/improve-data-validation

# Bug fix branches
fix/memory-leak-in-training
fix/streamlit-caching-issue

# Documentation branches
docs/update-setup-guide
docs/add-api-examples

# Refactor branches
refactor/simplify-preprocessing
refactor/optimize-feature-engineering
```

## MLflow Best Practices

### Experiment Tracking

```python
import mlflow
from emipredict.config.settings import Config

def train_with_mlflow(X_train, y_train, X_val, y_val):
    """Train model with comprehensive MLflow tracking."""
    
    # Set experiment
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME_CLASSIFICATION)
    
    with mlflow.start_run(run_name="xgboost_v1"):
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        
        # Log data info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("features", X_train.shape[1])
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log artifacts
        import matplotlib.pyplot as plt
        plt.figure()
        plot_confusion_matrix(y_val, y_pred)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Log tags
        mlflow.set_tag("stage", "development")
        mlflow.set_tag("developer", "team")
        
    return model
```

## Performance Guidelines

### Data Processing

```python
# Use vectorized operations
# Good
df['debt_ratio'] = df['debt'] / df['income']

# Bad
df['debt_ratio'] = df.apply(lambda x: x['debt'] / x['income'], axis=1)

# Use appropriate data types
# Good
df['age'] = df['age'].astype('uint8')  # 0-255 range
df['category'] = df['category'].astype('category')

# Use chunking for large files
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

### Model Training

```python
# Use early stopping
from xgboost import XGBClassifier

model = XGBClassifier(early_stopping_rounds=10)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Parallel processing
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_jobs=-1)  # Use all cores
```

## Security Guidelines

### Environment Variables

```python
# Never commit secrets
# Bad
API_KEY = "sk-1234567890abcdef"

# Good
import os
API_KEY = os.getenv('API_KEY')

# Validate environment variables
required_vars = ['MLFLOW_TRACKING_URI', 'DATA_PATH']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise ValueError(f"Missing environment variables: {missing}")
```

### Input Validation

```python
def predict_emi(age: int, salary: float, credit_score: int) -> float:
    """Predict EMI with input validation."""
    
    # Validate ranges
    if not (18 <= age <= 100):
        raise ValueError(f"Age must be between 18 and 100, got {age}")
    
    if salary <= 0:
        raise ValueError(f"Salary must be positive, got {salary}")
    
    if not (300 <= credit_score <= 850):
        raise ValueError(f"Credit score must be 300-850, got {credit_score}")
    
    # Sanitize inputs
    age = int(age)
    salary = float(salary)
    credit_score = int(credit_score)
    
    return model.predict([[age, salary, credit_score]])[0]
```

### Data Privacy

```python
# Don't log sensitive data
# Bad
logger.info(f"Processing user: {name}, SSN: {ssn}")

# Good
logger.info(f"Processing user ID: {hash(user_id)}")

# Anonymize data in logs
def anonymize_user_id(user_id: str) -> str:
    """Hash user ID for logging."""
    import hashlib
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]
```

## Code Review Guidelines

### As a Reviewer

- Be constructive and respectful
- Focus on code quality, not personal preferences
- Suggest improvements with examples
- Approve only if all checks pass

### As an Author

- Respond to all comments
- Don't take feedback personally
- Ask for clarification if needed
- Make requested changes or explain why not

## Continuous Improvement

- Refactor regularly
- Keep dependencies updated
- Monitor performance metrics
- Review and update documentation
- Share knowledge with team
- Learn from mistakes

---

**Remember**: Good code is code that others can understand, maintain, and extend easily.

