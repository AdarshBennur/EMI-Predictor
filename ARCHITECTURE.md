# EMI-Predict AI - System Architecture

## Overview

EMI-Predict AI is a machine learning system designed to assess financial risk for EMI (Equated Monthly Installment) lending decisions. The system performs two primary tasks:

1. **Classification**: Predict EMI eligibility (Eligible/Not_Eligible)
2. **Regression**: Predict maximum monthly EMI amount (in INR)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Raw Dataset (400K+ records, 27 features)            │   │
│  │  - Demographics, Income, Expenses, Credit Score      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Layer                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Data Loader (emipredict.data.loader)               │   │
│  │  - Missing value imputation                          │   │
│  │  - Outlier detection & handling                      │   │
│  │  - Categorical encoding                              │   │
│  │  - Feature scaling                                   │   │
│  │  - Train/Val/Test split (70/15/15)                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Feature Engineering Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Feature Engineer (emipredict.features.engineering)  │   │
│  │  - Debt-to-Income Ratio                             │   │
│  │  - Expense Ratio                                     │   │
│  │  - Savings Rate                                      │   │
│  │  - Financial Stress Index                           │   │
│  │  - Correlation Analysis                             │   │
│  │  - Feature Selection                                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│  ┌──────────────────┐        ┌──────────────────────────┐   │
│  │  Classification  │        │     Regression           │   │
│  │  ───────────────│        │     ──────────           │   │
│  │  • Logistic Reg  │        │  • Linear Regression     │   │
│  │  • Random Forest │        │  • Random Forest Reg     │   │
│  │  • XGBoost       │        │  • XGBoost Reg           │   │
│  │                  │        │                          │   │
│  │  Target:         │        │  Target:                 │   │
│  │  emi_eligibility │        │  max_monthly_emi         │   │
│  └──────────────────┘        └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Experiment Tracking Layer (MLflow)              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MLflow Tracker (emipredict.mlflow_utils.tracker)   │   │
│  │  - Log hyperparameters                               │   │
│  │  - Track metrics (accuracy, RMSE, etc.)            │   │
│  │  - Store model artifacts                            │   │
│  │  - Model registry & versioning                      │   │
│  │  - Experiment comparison                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Application Layer (Streamlit)                 │
│  ┌──────────────┬─────────────┬──────────────┬──────────┐   │
│  │ Data         │ Eligibility │ EMI Amount   │ Admin    │   │
│  │ Explorer     │ Prediction  │ Prediction   │ Monitor  │   │
│  │              │             │              │          │   │
│  │ • EDA        │ • Input Form│ • Input Form │ • Metrics│   │
│  │ • Viz        │ • Predict   │ • Predict    │ • MLflow │   │
│  │ • Stats      │ • Explain   │ • Explain    │ • Logs   │   │
│  └──────────────┴─────────────┴──────────────┴──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Ingestion & Preprocessing

**Input**: `data/emi_prediction_dataset.csv`

**Process**:
- Load 400K+ records with 27 features
- Handle missing values using median/mode imputation
- Detect and treat outliers using IQR method
- Encode categorical variables (gender, education, employment_type, etc.)
- Scale numerical features using StandardScaler
- Split data: 70% training, 15% validation, 15% test

**Output**: Cleaned, processed DataFrames ready for feature engineering

### 2. Feature Engineering

**Derived Features**:

1. **Debt-to-Income Ratio (DTI)**
   ```
   DTI = (current_emi_amount + other_expenses) / monthly_salary
   ```

2. **Total Monthly Expenses**
   ```
   total_expenses = monthly_rent + groceries_utilities + 
                    travel_expenses + other_monthly_expenses +
                    school_fees + college_fees
   ```

3. **Expense Ratio**
   ```
   expense_ratio = total_expenses / monthly_salary
   ```

4. **Savings Rate**
   ```
   savings_rate = (monthly_salary - total_expenses - current_emi) / monthly_salary
   ```

5. **Financial Stress Index**
   ```
   stress_index = (current_emi + total_expenses) / (monthly_salary + bank_balance/12)
   ```

6. **Credit Utilization Score**
   ```
   credit_util = current_emi_amount / (monthly_salary * 0.5)
   ```

**Output**: Enhanced feature set for model training

### 3. Model Training

#### Classification Pipeline (EMI Eligibility)

**Models**:
1. **Logistic Regression** (Baseline)
   - Fast, interpretable
   - Good for linear relationships
   - Hyperparameters: C, penalty, solver

2. **Random Forest Classifier**
   - Handles non-linear patterns
   - Feature importance
   - Hyperparameters: n_estimators, max_depth, min_samples_split

3. **XGBoost Classifier**
   - State-of-the-art performance
   - Gradient boosting
   - Hyperparameters: learning_rate, max_depth, n_estimators, subsample

**Metrics**:
- Accuracy (target: >90%)
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

#### Regression Pipeline (EMI Amount)

**Models**:
1. **Linear Regression** (Baseline)
   - Simple, interpretable
   - Assumptions: linearity, normality

2. **Random Forest Regressor**
   - Non-linear relationships
   - Robust to outliers
   - Hyperparameters: n_estimators, max_depth, min_samples_split

3. **XGBoost Regressor**
   - Best performance expected
   - Gradient boosting
   - Hyperparameters: learning_rate, max_depth, n_estimators, subsample

**Metrics**:
- RMSE (target: <2000 INR)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### 4. MLflow Integration

**Tracking Components**:

1. **Experiments**: Separate experiments for classification and regression
2. **Runs**: Each model training session is a run
3. **Parameters**: All hyperparameters logged
4. **Metrics**: Performance metrics logged per epoch/iteration
5. **Artifacts**: Model files, plots, feature importance
6. **Tags**: Model type, version, dataset version

**Model Registry**:
- Register best-performing models
- Version control
- Stage transitions (Staging → Production)
- Model metadata and lineage

### 5. Application Layer

#### Streamlit Multi-Page App

**Page 1: Data Explorer (📊)**
- Dataset overview and statistics
- Distribution plots for key features
- Correlation heatmaps
- Missing value analysis
- Outlier visualization

**Page 2: Eligibility Prediction (🎯)**
- User input form (22 features)
- Real-time prediction
- Probability scores
- Feature importance for prediction
- Eligibility explanation

**Page 3: EMI Amount Prediction (💰)**
- User input form
- Predicted EMI amount
- Confidence interval
- Affordability analysis
- Recommendation engine

**Page 4: Admin Monitoring (📈)**
- Model performance dashboard
- MLflow experiment comparison
- Data drift detection
- System logs and alerts
- Model retraining triggers

## Component Details

### Configuration Management

**Location**: `emipredict/config/settings.py`

**Configuration Items**:
- Dataset paths
- Model hyperparameters
- MLflow tracking URI
- Feature engineering parameters
- Train/val/test split ratios
- Scaling methods
- Model save paths

### Utilities

**Location**: `emipredict/utils/helpers.py`

**Functions**:
- Data validation
- Custom metrics calculation
- Plot generation
- Error handling
- Logging utilities
- Model loading/saving helpers

## Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Configurability**: Easy to adjust parameters without code changes
3. **Reproducibility**: MLflow ensures experiment reproducibility
4. **Testability**: 85%+ test coverage with unit and integration tests
5. **Maintainability**: Clear documentation, type hints, docstrings
6. **Scalability**: Can handle larger datasets with minor modifications

## Technology Choices

### Why MLflow?
- Industry-standard experiment tracking
- Model registry and versioning
- Easy deployment options
- Open-source and extensible

### Why Streamlit?
- Rapid prototyping
- Python-native (no JS required)
- Great for data science applications
- Easy deployment to Streamlit Cloud

### Why XGBoost?
- State-of-the-art gradient boosting
- High performance on structured data
- Built-in regularization
- Handles missing values

## Security Considerations

1. **Environment Variables**: Sensitive data in `.env` file
2. **Input Validation**: Sanitize user inputs in Streamlit app
3. **Data Privacy**: No PII storage in logs
4. **Model Security**: Protect model files from unauthorized access

## Performance Optimization

1. **Data Loading**: Efficient pandas operations with appropriate dtypes
2. **Feature Engineering**: Vectorized operations
3. **Model Training**: Parallel processing where applicable
4. **Caching**: Streamlit caching for expensive operations
5. **Batch Processing**: For large-scale predictions

## Future Enhancements

1. **API Layer**: REST API for production integration
2. **Real-time Predictions**: Kafka/Redis for streaming
3. **Auto-retraining**: Scheduled model updates
4. **A/B Testing**: Compare model versions in production
5. **Explainability**: SHAP values for model interpretability
6. **Docker**: Containerization for easy deployment
7. **CI/CD**: Automated testing and deployment pipeline

