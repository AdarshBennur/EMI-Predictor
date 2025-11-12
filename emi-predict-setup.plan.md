<!-- acdf9370-41e9-46be-a396-18ab16fdbbc3 8c52c092-fff9-4d45-bfce-792ee91722e0 -->
# EMI-Predict-AI: Production-Ready Setup Plan

## Streamlined Folder Structure

```
3. EMI-Predict-AI/
├── data/                      # ✓ Already exists with dataset
├── emipredict/                # Main source package
│   ├── __init__.py
│   ├── data/                  # Data loading & preprocessing
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/                # ML model training
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   └── regression.py
│   ├── mlflow_utils/          # MLflow tracking
│   │   ├── __init__.py
│   │   └── tracker.py
│   ├── app/                   # Streamlit multi-page app
│   │   ├── main.py
│   │   └── pages/
│   │       ├── 1_📊_Data_Explorer.py
│   │       ├── 2_🎯_Eligibility_Prediction.py
│   │       ├── 3_💰_EMI_Amount_Prediction.py
│   │       └── 4_📈_Admin_Monitoring.py
│   ├── utils/                 # Shared utilities
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── config/                # Configuration management
│       ├── __init__.py
│       └── settings.py
├── tests/                     # Unit & integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_app.py
├── .env.example               # Environment template
├── .cursorrules               # Cursor AI guidelines
├── .gitignore                 # Git ignore patterns
├── requirements.txt           # Python dependencies
├── README.md                  # Quick start guide
├── ARCHITECTURE.md            # System architecture
├── DATABASE_SCHEMA.md         # Data schema documentation
├── SETUP.md                   # Setup instructions
└── DEVGUIDE.md                # Development guidelines
```

**Removed:** `notebooks/` (exploratory work), `API_SPEC.md` (no REST APIs planned yet)

## Implementation Steps

### 1. Core Documentation Files

Create foundational documentation that describes the system architecture, data schema, setup process, and development guidelines.

### 2. Configuration & Environment

Set up `.env.example`, `.cursorrules`, `.gitignore`, and `requirements.txt` with all necessary dependencies (pandas, scikit-learn, xgboost, mlflow, streamlit, pytest, etc.).

### 3. Package Structure

Create the complete `emipredict/` package with all subdirectories and `__init__.py` files for proper Python module organization.

### 4. Data Pipeline Module

Build `emipredict/data/loader.py` to handle dataset loading, missing value imputation, outlier handling, categorical encoding, feature scaling, and train/val/test splitting.

### 5. Feature Engineering Module

Develop `emipredict/features/engineering.py` to create derived financial features (debt-to-income ratio, expense ratios, etc.) and perform correlation analysis.

### 6. Model Development Modules

Implement both classification (`models/classification.py`) and regression (`models/regression.py`) with multiple algorithms, hyperparameter tuning, and model persistence.

### 7. MLflow Integration

Create `mlflow_utils/tracker.py` for experiment tracking, metric logging, model registry, and comparison utilities.

### 8. Streamlit Multi-Page Application

Build the main Streamlit app with four pages: data exploration, eligibility prediction, EMI amount prediction, and admin monitoring dashboard.

### 9. Testing Infrastructure

Write comprehensive unit tests for all modules with pytest, targeting 85-90% code coverage.

### 10. Final Documentation Updates

Update all docs with implementation details, usage examples, and deployment instructions for Streamlit Cloud.

## Key Technical Decisions

- **Data Split:** 70% train, 15% validation, 15% test
- **Classification Models:** Logistic Regression, Random Forest, XGBoost
- **Regression Models:** Linear Regression, Random Forest Regressor, XGBoost Regressor
- **MLflow Tracking:** Local file store for experiments, SQLite backend for metadata
- **UI Framework:** Streamlit with session state for predictions
- **Testing:** pytest with coverage reporting

## Success Criteria

- Modular, well-documented codebase with type hints
- Classification accuracy ≥90%, Regression RMSE <2000 INR
- All major modules have 85%+ test coverage
- Complete documentation ready for team collaboration
- Streamlit app ready for deployment

### To-dos

- [ ] Create all documentation files (ARCHITECTURE.md, DATABASE_SCHEMA.md, SETUP.md, DEVGUIDE.md, README.md)
- [ ] Create .env.example, .cursorrules, .gitignore, and requirements.txt with all dependencies
- [ ] Create complete emipredict/ package structure with all subdirectories and __init__.py files
- [ ] Implement data loading and preprocessing pipeline in emipredict/data/loader.py
- [ ] Implement feature engineering module in emipredict/features/engineering.py
- [ ] Implement classification and regression models with hyperparameter tuning
- [ ] Create MLflow tracking utilities for experiment management and model registry
- [ ] Create multi-page Streamlit application with all four pages (data explorer, predictions, monitoring)
- [ ] Write comprehensive unit tests for all modules targeting 85-90% coverage
- [ ] Update all documentation with implementation details and deployment instructions