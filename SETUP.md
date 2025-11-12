# EMI-Predict AI - Setup Guide

This guide provides detailed instructions for setting up the EMI-Predict AI project in development and production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Training Models](#training-models)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB (16GB recommended for model training)
- **Disk Space**: Minimum 5GB free space
- **Internet**: Required for package installation and deployment

### Required Software

1. **Python 3.9+**
   ```bash
   python --version  # Should show 3.9 or higher
   ```

2. **pip** (Python package manager)
   ```bash
   pip --version
   ```

3. **Git** (for version control)
   ```bash
   git --version
   ```

4. **Virtual Environment Tool** (venv or conda)

## Development Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone <repository-url>
cd "3. EMI-Predict-AI"
```

### 2. Create Virtual Environment

**Option A: Using venv (recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n emipredict python=3.9

# Activate environment
conda activate emipredict
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check if key packages are installed
python -c "import pandas, sklearn, xgboost, mlflow, streamlit; print('All packages installed successfully!')"
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME_CLASSIFICATION=emi_eligibility_classification
MLFLOW_EXPERIMENT_NAME_REGRESSION=emi_amount_regression

# Data Configuration
DATA_PATH=data/emi_prediction_dataset.csv
TRAIN_TEST_SPLIT_RATIO=0.70
VAL_TEST_SPLIT_RATIO=0.50
RANDOM_STATE=42

# Model Configuration
MODELS_DIR=models
CLASSIFICATION_MODEL_PATH=models/classification_model.pkl
REGRESSION_MODEL_PATH=models/regression_model.pkl

# Feature Engineering
CREATE_DERIVED_FEATURES=True
FEATURE_SELECTION_THRESHOLD=0.01

# Training Configuration
CLASSIFICATION_TARGET=emi_eligibility
REGRESSION_TARGET=max_monthly_emi
N_HYPERPARAMETER_TRIALS=50

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_THEME=light

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/emipredict.log
```

### 2. Directory Structure Setup

The application will automatically create necessary directories on first run:

```bash
# Manually create directories (optional)
mkdir -p models
mkdir -p mlruns
mkdir -p logs
mkdir -p plots
```

### 3. Verify Data

Ensure the dataset is present:

```bash
# Check if dataset exists
ls -lh data/emi_prediction_dataset.csv

# View first few lines
head -n 5 data/emi_prediction_dataset.csv
```

## Running the Application

### 1. Launch Streamlit Web App

```bash
# Start the Streamlit application
streamlit run emipredict/app/main.py
```

The app will be available at: `http://localhost:8501`

**Alternative: Specify custom port**
```bash
streamlit run emipredict/app/main.py --server.port 8080
```

### 2. Access MLflow UI

In a separate terminal:

```bash
# Activate virtual environment
source venv/bin/activate  # or: conda activate emipredict

# Launch MLflow UI
mlflow ui

# Alternative: Specify custom port
mlflow ui --port 5001
```

Access MLflow dashboard at: `http://localhost:5000`

## Training Models

### Quick Start Training

Train all models with default settings:

```bash
# Train classification models
python -m emipredict.models.classification

# Train regression models
python -m emipredict.models.regression
```

### Training Individual Models

```python
# Create a training script (train.py)
from emipredict.data.loader import load_and_preprocess_data
from emipredict.features.engineering import create_features
from emipredict.models.classification import train_classification_models

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

# Create features
X_train_eng = create_features(X_train)
X_val_eng = create_features(X_val)

# Train models
results = train_classification_models(X_train_eng, y_train, X_val_eng, y_val)
```

### Monitor Training Progress

```bash
# View MLflow experiments
mlflow ui

# Check logs
tail -f logs/emipredict.log
```

## Testing

### Run All Tests

```bash
# Run complete test suite
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=emipredict --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
# or
start htmlcov/index.html  # Windows
```

### Run Specific Tests

```bash
# Test data module only
pytest tests/test_data.py

# Test specific function
pytest tests/test_data.py::test_load_data

# Test with markers
pytest -m "not slow"
```

### Test Coverage Goals

- Overall coverage: **85%+**
- Critical modules (data, models): **90%+**
- Utility functions: **80%+**

## Deployment

### Option 1: Streamlit Cloud (Recommended)

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free at streamlit.io)

#### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `emipredict/app/main.py`
   - Click "Deploy"

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to App Settings → Secrets
   - Add environment variables from `.env` file:
   ```toml
   MLFLOW_TRACKING_URI = "file:./mlruns"
   DATA_PATH = "data/emi_prediction_dataset.csv"
   # Add other secrets as needed
   ```

4. **Access Your App**
   - Your app will be live at: `https://<your-app-name>.streamlit.app`

### Option 2: Local Server Deployment

```bash
# Install production dependencies
pip install gunicorn

# Note: Streamlit uses its own server, so just run:
streamlit run emipredict/app/main.py --server.port 80 --server.address 0.0.0.0
```

### Option 3: Docker Deployment (Future Enhancement)

Currently not implemented, but planned for future releases.

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'emipredict'`

**Solution**:
```bash
# Ensure you're in the project root directory
pwd  # Should show: /path/to/3. EMI-Predict-AI

# Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Data Loading Issues

**Problem**: `FileNotFoundError: data/emi_prediction_dataset.csv not found`

**Solution**:
```bash
# Verify file exists
ls -la data/

# Check absolute path
readlink -f data/emi_prediction_dataset.csv

# Update DATA_PATH in .env if necessary
```

#### 3. MLflow Tracking Issues

**Problem**: `MLflow tracking URI not found`

**Solution**:
```bash
# Create mlruns directory
mkdir -p mlruns

# Reset MLflow tracking
rm -rf mlruns/*

# Restart MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

#### 4. Memory Issues During Training

**Problem**: `MemoryError` or system slowdown

**Solution**:
```python
# In your training script, reduce data size temporarily
from emipredict.data.loader import load_and_preprocess_data

# Load subset of data
data = load_and_preprocess_data(nrows=100000)  # Use only 100K rows

# Or increase system swap space
```

#### 5. Streamlit Connection Issues

**Problem**: `StreamlitAPIException: Streamlit cannot connect`

**Solution**:
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Restart Streamlit
streamlit run emipredict/app/main.py
```

#### 6. Package Version Conflicts

**Problem**: `ImportError` or version incompatibility

**Solution**:
```bash
# Create fresh virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Performance Optimization

#### Speed Up Model Training

1. **Reduce hyperparameter search space**
   ```python
   # In training scripts
   param_grid = {
       'n_estimators': [100, 200],  # Instead of [100, 200, 300, 500]
       'max_depth': [5, 10]  # Instead of [5, 10, 15, 20]
   }
   ```

2. **Use fewer cross-validation folds**
   ```python
   from sklearn.model_selection import cross_val_score
   
   scores = cross_val_score(model, X, y, cv=3)  # Instead of cv=5 or cv=10
   ```

3. **Parallel processing**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   
   model = RandomForestClassifier(n_jobs=-1)  # Use all CPU cores
   ```

#### Optimize Streamlit Performance

1. **Use caching**
   ```python
   @st.cache_data
   def load_data():
       return pd.read_csv('data/emi_prediction_dataset.csv')
   ```

2. **Load models once**
   ```python
   @st.cache_resource
   def load_model():
       return joblib.load('models/classification_model.pkl')
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check logs**
   ```bash
   tail -f logs/emipredict.log
   ```

2. **Enable debug mode**
   ```bash
   # In .env
   LOG_LEVEL=DEBUG
   ```

3. **Review MLflow runs**
   - Check for failed experiments in MLflow UI
   - Review parameter and metric logs

4. **Community Support**
   - Create an issue on GitHub
   - Check existing issues for similar problems

## Next Steps

After successful setup:

1. Review [ARCHITECTURE.md](ARCHITECTURE.md) to understand system design
2. Read [DEVGUIDE.md](DEVGUIDE.md) for coding guidelines
3. Explore [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for data understanding
4. Start with data exploration in Streamlit app
5. Train your first models
6. Review results in MLflow UI

## Maintenance

### Regular Tasks

1. **Update dependencies** (monthly)
   ```bash
   pip list --outdated
   pip install --upgrade <package-name>
   ```

2. **Clean up old experiments** (as needed)
   ```bash
   # Archive old MLflow runs
   mlflow gc --backend-store-uri file:./mlruns
   ```

3. **Monitor disk space**
   ```bash
   du -sh mlruns/ models/ logs/
   ```

4. **Backup models and data**
   ```bash
   tar -czf backup_$(date +%Y%m%d).tar.gz models/ mlruns/ data/
   ```

