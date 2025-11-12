# EMI-Predict AI

Intelligent financial risk assessment platform for EMI eligibility and EMI amount prediction using Machine Learning, MLflow, and Streamlit.

## Overview

EMI-Predict AI is a production-ready ML system that helps financial institutions and loan applicants assess:
- **EMI Eligibility**: Binary classification to determine if a borrower qualifies for an EMI
- **EMI Amount**: Regression to predict the maximum monthly EMI a borrower can afford

Built with a modular architecture, comprehensive testing, and MLflow experiment tracking for model lifecycle management.

## Key Features

- 🎯 **90%+ Classification Accuracy** for EMI eligibility prediction
- 💰 **RMSE < 2000 INR** for EMI amount regression
- 📊 **Interactive Web Interface** with Streamlit multi-page app
- 🔬 **MLflow Integration** for experiment tracking and model registry
- 🧪 **85%+ Test Coverage** with comprehensive unit tests
- 📈 **Real-time Monitoring** dashboard for model performance

## Tech Stack

- **ML Framework**: scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Web Framework**: Streamlit
- **Data Processing**: pandas, NumPy
- **Testing**: pytest, pytest-cov
- **Python**: 3.9+

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "3. EMI-Predict-AI"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Usage

#### 1. Train Models

```bash
# Train classification models
python -m emipredict.models.classification

# Train regression models
python -m emipredict.models.regression
```

#### 2. Launch Streamlit App

```bash
streamlit run emipredict/app/main.py
```

The app will be available at `http://localhost:8501`

#### 3. View MLflow Dashboard

```bash
mlflow ui
```

Access the MLflow UI at `http://localhost:5000`

## Project Structure

```
3. EMI-Predict-AI/
├── data/                      # Dataset storage
├── emipredict/                # Main source package
│   ├── data/                  # Data loading & preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # ML model training
│   ├── mlflow_utils/          # MLflow tracking utilities
│   ├── app/                   # Streamlit web application
│   ├── utils/                 # Shared utilities
│   └── config/                # Configuration management
├── tests/                     # Unit & integration tests
├── mlruns/                    # MLflow experiment artifacts (generated)
├── models/                    # Saved model files (generated)
└── docs/                      # Additional documentation
```

## Web Application Pages

1. **📊 Data Explorer**: Visualize dataset statistics and distributions
2. **🎯 Eligibility Prediction**: Predict if a borrower qualifies for EMI
3. **💰 EMI Amount Prediction**: Predict maximum affordable monthly EMI
4. **📈 Admin Monitoring**: View model metrics and MLflow experiments

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=emipredict --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and data flow
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) - Dataset schema and feature descriptions
- [SETUP.md](SETUP.md) - Detailed setup and deployment instructions
- [DEVGUIDE.md](DEVGUIDE.md) - Development guidelines and best practices

## Model Performance

### Classification (EMI Eligibility)
- **Target Metric**: >90% Accuracy
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Regression (EMI Amount)
- **Target Metric**: RMSE < 2000 INR
- **Models**: Linear Regression, Random Forest Regressor, XGBoost Regressor
- **Evaluation**: RMSE, MAE, R² Score

## Contributing

1. Follow the coding guidelines in [DEVGUIDE.md](DEVGUIDE.md)
2. Write tests for all new features (maintain 85%+ coverage)
3. Update documentation with significant changes
4. Use type hints and comprehensive docstrings
5. Track experiments with MLflow

## Deployment

### Streamlit Cloud

1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard
4. Deploy!

See [SETUP.md](SETUP.md) for detailed deployment instructions.

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

Built for financial institutions and loan applicants to make informed EMI decisions.

