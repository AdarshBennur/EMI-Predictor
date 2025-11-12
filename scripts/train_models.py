"""
Main training script for EMI-Predict AI models.
Trains all classification and regression models with configurable data size.

Usage:
    python scripts/train_models.py

This script will:
- Load and preprocess data (50K rows by default for quick validation)
- Auto-select top 500 features using variance threshold and mutual information
- Train 3 classification models (Logistic Regression, Random Forest, XGBoost)
- Train 3 regression models (Linear Regression, Random Forest, XGBoost)
- Save all models to models/ directory
- Create MLflow experiments in mlruns/

For production training on full dataset:
- Change TRAINING_ROWS to None
- Expect 30-60 minutes training time
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from emipredict.config.settings import Config
from emipredict.data.loader import (
    load_data, 
    validate_data, 
    handle_missing_values, 
    encode_categorical_features, 
    encode_target_variable, 
    scale_features
)
from emipredict.features.engineering import create_features
from emipredict.models.classification import train_classification_models
from emipredict.models.regression import train_regression_models
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAINING_ROWS = 50000  # Start with 50K for quick validation (change to None for full dataset)
MAX_FEATURES = 500  # Maximum number of features to keep after selection

# =============================================================================
# FEATURE SELECTION FUNCTIONS
# =============================================================================

def select_best_features(X_train, X_val, X_test, y_train, max_features=500):
    """
    Select the best features using variance threshold and mutual information.
    Returns pandas DataFrames with selected column names.
    
    Args:
        X_train, X_val, X_test: Feature matrices (pandas DataFrames)
        y_train: Target variable for training (classification)
        max_features: Maximum number of features to keep
    
    Returns:
        X_train_selected, X_val_selected, X_test_selected (as DataFrames), feature_selector, selected_columns
    """
    logger.info(f"Original feature count: {X_train.shape[1]}")
    
    # Save original column names
    original_columns = X_train.columns.tolist()
    
    # Step 1: Remove low variance features (variance < 0.01)
    logger.info("Step 1: Removing low-variance features...")
    variance_selector = VarianceThreshold(threshold=0.01)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)
    X_test_var = variance_selector.transform(X_test)
    
    # Get columns that passed variance threshold
    variance_support = variance_selector.get_support()
    columns_after_variance = [col for col, selected in zip(original_columns, variance_support) if selected]
    
    logger.info(f"After variance filtering: {X_train_var.shape[1]} features")
    
    # Step 2: Select top K features using mutual information
    # If we still have more features than max_features, select the best ones
    if X_train_var.shape[1] > max_features:
        logger.info(f"Step 2: Selecting top {max_features} features using mutual information...")
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        X_train_selected = mi_selector.fit_transform(X_train_var, y_train)
        X_val_selected = mi_selector.transform(X_val_var)
        X_test_selected = mi_selector.transform(X_test_var)
        
        # Get columns that passed mutual information selection
        mi_support = mi_selector.get_support()
        selected_columns = [col for col, selected in zip(columns_after_variance, mi_support) if selected]
        
        logger.info(f"After mutual information selection: {X_train_selected.shape[1]} features")
        
        # Create combined selector
        class CombinedSelector:
            def __init__(self, variance_selector, mi_selector):
                self.variance_selector = variance_selector
                self.mi_selector = mi_selector
            
            def transform(self, X):
                X_var = self.variance_selector.transform(X)
                return self.mi_selector.transform(X_var)
        
        feature_selector = CombinedSelector(variance_selector, mi_selector)
    else:
        logger.info(f"Variance filtering sufficient: keeping all {X_train_var.shape[1]} features")
        X_train_selected = X_train_var
        X_val_selected = X_val_var
        X_test_selected = X_test_var
        selected_columns = columns_after_variance
        feature_selector = variance_selector
    
    # Convert numpy arrays back to pandas DataFrames with selected column names
    X_train_df = pd.DataFrame(X_train_selected, columns=selected_columns, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_selected, columns=selected_columns, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_columns, index=X_test.index)
    
    logger.info(f"Final feature count: {len(selected_columns)}")
    logger.info(f"Selected features converted back to DataFrames")
    
    return X_train_df, X_val_df, X_test_df, feature_selector, selected_columns

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training pipeline."""
    
    logger.info("=" * 80)
    logger.info("EMI-PREDICT AI - MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    if TRAINING_ROWS is None:
        logger.info("Training on FULL DATASET (400K+ rows) - This will take 30-60 minutes")
    else:
        logger.info(f"Training on SUBSET ({TRAINING_ROWS:,} rows) - Quick validation mode")
    
    logger.info("=" * 80)
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info("\nSTEP 1: Loading data...")
    df = load_data(nrows=TRAINING_ROWS)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # =========================================================================
    # STEP 2: VALIDATE DATA
    # =========================================================================
    logger.info("\nSTEP 2: Validating data...")
    validate_data(df)
    logger.info("Data validation passed")
    
    # =========================================================================
    # STEP 3: CREATE AND FIT PREPROCESSING PIPELINE
    # =========================================================================
    logger.info("\nSTEP 3: Creating and fitting preprocessing pipeline...")
    
    from emipredict.data.pipeline import EMIPreprocessingPipeline
    
    # Create preprocessing pipeline
    preprocessing_pipeline = EMIPreprocessingPipeline()
    
    # Fit pipeline on full dataset (learns encodings, etc.)
    preprocessing_pipeline.fit(df)
    logger.info("Preprocessing pipeline fitted")
    
    # Apply preprocessing (includes missing value handling, encoding, feature engineering)
    df_preprocessed = preprocessing_pipeline.transform(df.copy())
    
    # Re-add target columns for splitting
    df_preprocessed['emi_eligibility'] = df['emi_eligibility']
    df_preprocessed['max_monthly_emi'] = df['max_monthly_emi']
    
    df = df_preprocessed
    logger.info(f"Preprocessing completed. Shape: {df.shape}")
    
    # =========================================================================
    # SAVE PREPROCESSING PIPELINE (CRITICAL FOR PREDICTION)
    # =========================================================================
    logger.info("\nSaving preprocessing pipeline for prediction...")
    
    pipeline_path = Config.MODELS_DIR / "preprocessing_pipeline.pkl"
    preprocessing_pipeline.save(pipeline_path)
    logger.info(f"✅ Preprocessing pipeline saved to: {pipeline_path}")
    logger.info(f"   Pipeline ensures consistent feature transformation at prediction time")
    
    # =========================================================================
    # STEP 4: SPLIT FEATURES AND TARGETS
    # =========================================================================
    logger.info("\nSTEP 4: Splitting features and targets...")
    
    # Separate features and targets
    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y_clf = df['emi_eligibility']
    y_reg = df['max_monthly_emi']
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Classification target shape: {y_clf.shape}")
    logger.info(f"Regression target shape: {y_reg.shape}")
    
    # Encode classification target
    y_clf_encoded, clf_encoder = encode_target_variable(y_clf, task='classification')
    logger.info("Classification target encoded (3 classes)")
    
    # =========================================================================
    # STEP 5: TRAIN-VAL-TEST SPLIT
    # =========================================================================
    logger.info("\nSTEP 5: Creating train-val-test splits...")
    
    # First split: 85% temp, 15% test
    X_temp, X_test, y_clf_temp, y_clf_test, y_reg_temp, y_reg_test = train_test_split(
        X, y_clf_encoded, y_reg, 
        test_size=0.15, 
        random_state=Config.RANDOM_STATE
    )
    
    # Second split: 70% train, 15% val from temp
    X_train, X_val, y_clf_train, y_clf_val, y_reg_train, y_reg_val = train_test_split(
        X_temp, y_clf_temp, y_reg_temp,
        test_size=0.176,  # 0.176 * 0.85 ≈ 0.15 of total
        random_state=Config.RANDOM_STATE
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Validation set: {len(X_val):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    
    # Note: Feature engineering is already done by the preprocessing pipeline
    logger.info(f"Current features after preprocessing pipeline: {X_train.shape[1]}")
    
    # =========================================================================
    # STEP 6: FEATURE SELECTION
    # =========================================================================
    logger.info("\nSTEP 6: Selecting best features to reduce memory usage...")
    
    X_train, X_val, X_test, feature_selector, selected_columns = select_best_features(
        X_train, X_val, X_test, y_clf_train, max_features=MAX_FEATURES
    )
    
    logger.info(f"Feature selection completed: {X_train.shape[1]} features selected")
    logger.info(f"Selected features are now pandas DataFrames with correct column names")
    
    # =========================================================================
    # SAVE FEATURE SELECTOR (CRITICAL FOR PREDICTION)
    # =========================================================================
    logger.info("\nSaving feature selector for prediction pipeline...")
    
    import joblib
    
    # Save feature selector object
    feature_selector_path = Config.MODELS_DIR / "feature_selector.pkl"
    joblib.dump(feature_selector, feature_selector_path)
    logger.info(f"✅ Feature selector saved to: {feature_selector_path}")
    
    # Save selected column names for reference
    selected_columns_path = Config.MODELS_DIR / "selected_features.pkl"
    joblib.dump(selected_columns, selected_columns_path)
    logger.info(f"✅ Selected feature names saved to: {selected_columns_path}")
    logger.info(f"   Total selected features: {len(selected_columns)}")
    
    # =========================================================================
    # STEP 7: FEATURE SCALING
    # =========================================================================
    logger.info("\nSTEP 7: Scaling features...")
    
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    
    logger.info("Features scaled using StandardScaler")
    
    # =========================================================================
    # STEP 8: TRAIN CLASSIFICATION MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Training Classification Models (3 models)")
    logger.info("=" * 80)
    
    try:
        clf_results = train_classification_models(
            X_train, y_clf_train, 
            X_val, y_clf_val, 
            X_test, y_clf_test,
            hyperparameter_tuning=False  # Set to True for production
        )
        
        logger.info("\n✅ Classification models trained successfully!")
        logger.info(f"Trained {len(clf_results)} classification models")
        
        # Display results
        logger.info("\nClassification Results:")
        for model_name, (model, metrics) in clf_results.items():
            logger.info(f"  {model_name}: Accuracy = {metrics['val_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Classification training failed: {str(e)}")
        raise
    
    # =========================================================================
    # STEP 9: TRAIN REGRESSION MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: Training Regression Models (3 models)")
    logger.info("=" * 80)
    
    try:
        reg_results = train_regression_models(
            X_train, y_reg_train,
            X_val, y_reg_val,
            X_test, y_reg_test,
            hyperparameter_tuning=False  # Set to True for production
        )
        
        logger.info("\n✅ Regression models trained successfully!")
        logger.info(f"Trained {len(reg_results)} regression models")
        
        # Display results
        logger.info("\nRegression Results:")
        for model_name, (model, metrics) in reg_results.items():
            logger.info(f"  {model_name}: RMSE = {metrics['val_rmse']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Regression training failed: {str(e)}")
        raise
    
    # =========================================================================
    # STEP 10: SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\n📊 Data:")
    logger.info(f"  - Training samples: {len(X_train):,}")
    logger.info(f"  - Validation samples: {len(X_val):,}")
    logger.info(f"  - Test samples: {len(X_test):,}")
    logger.info(f"  - Total features: {X_train.shape[1]}")
    
    logger.info(f"\n🤖 Models Trained:")
    logger.info(f"  - Classification: {len(clf_results)} models")
    logger.info(f"  - Regression: {len(reg_results)} models")
    logger.info(f"  - Total: {len(clf_results) + len(reg_results)} models")
    
    logger.info(f"\n💾 Models saved to: {Config.MODELS_DIR}")
    logger.info(f"📊 MLflow experiments in: mlruns/")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    logger.info("\n🚀 Next Steps:")
    logger.info("1. Run the Streamlit app: streamlit run emipredict/app/main.py")
    logger.info("2. View MLflow UI: mlflow ui (then visit http://localhost:5000)")
    logger.info("3. Test predictions in the web application")
    
    if TRAINING_ROWS is not None:
        logger.info("\n⚠️  NOTE: Models trained on 50K subset for quick validation")
        logger.info("   For production, set TRAINING_ROWS = None and retrain on full dataset")
    
    return clf_results, reg_results


if __name__ == "__main__":
    try:
        clf_results, reg_results = main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

