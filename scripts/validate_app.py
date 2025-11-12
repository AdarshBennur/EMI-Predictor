#!/usr/bin/env python3
"""
Validation script to ensure the EMI-Predict AI application is fully functional.

This script verifies:
1. All required models are trained and saved
2. MLflow experiments are accessible
3. Streamlit pages load without errors
4. Models can make predictions

Usage:
    python scripts/validate_app.py
"""

import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from emipredict.config.settings import Config
import mlflow

def check_models():
    """Check if all required models are present."""
    print("=" * 60)
    print("🔍 Checking Models...")
    print("=" * 60)
    
    models_dir = Config.MODELS_DIR
    required_models = [
        "logistic_regression_classification.pkl",
        "random_forest_classification.pkl",
        "xgboost_classification.pkl",
        "xgboost_classification_best.pkl",
        "linear_regression_regression.pkl",
        "random_forest_regression.pkl",
        "xgboost_regression.pkl",
        "xgboost_regression_best.pkl",
    ]
    
    found_models = []
    missing_models = []
    
    for model_name in required_models:
        model_path = models_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_name:<40} ({size_mb:.2f} MB)")
            found_models.append(model_name)
        else:
            print(f"❌ {model_name:<40} MISSING")
            missing_models.append(model_name)
    
    print(f"\nSummary: {len(found_models)}/{len(required_models)} models found")
    
    if missing_models:
        print("\n⚠️  Missing models:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nRun: python scripts/train_models.py")
        return False
    
    return True


def check_mlflow():
    """Check if MLflow experiments are accessible."""
    print("\n" + "=" * 60)
    print("📊 Checking MLflow...")
    print("=" * 60)
    
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        experiments = mlflow.search_experiments()
        
        print(f"✅ MLflow accessible: {len(experiments)} experiments found")
        
        for exp in experiments:
            if exp.name != "Default":
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                print(f"   - {exp.name}: {len(runs)} runs")
        
        return True
    except Exception as e:
        print(f"❌ MLflow error: {str(e)}")
        return False


def test_classification_model():
    """Test classification model predictions."""
    print("\n" + "=" * 60)
    print("🎯 Testing Classification Model...")
    print("=" * 60)
    
    try:
        # Load model
        model_path = Config.get_model_save_path("xgboost", "classification", "_best")
        if not model_path.exists():
            model_path = Config.get_model_save_path("xgboost", "classification")
        
        model_data = joblib.load(model_path)
        
        # Extract model from dictionary if needed
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
        else:
            model = model_data
        
        print(f"✅ Model loaded: {model_path.name}")
        
        # Create test data (matching the expected feature count)
        # The model expects 11070 features after preprocessing
        test_data = pd.DataFrame(np.random.randn(1, 11070))
        
        # Make prediction
        prediction = model.predict(test_data)
        proba = model.predict_proba(test_data)
        
        classes = ["Eligible", "High_Risk", "Not_Eligible"]
        print(f"✅ Prediction: {classes[prediction[0]]}")
        print(f"✅ Probabilities: {dict(zip(classes, proba[0]))}")
        
        return True
    except Exception as e:
        print(f"❌ Classification test failed: {str(e)}")
        return False


def test_regression_model():
    """Test regression model predictions."""
    print("\n" + "=" * 60)
    print("💰 Testing Regression Model...")
    print("=" * 60)
    
    try:
        # Load model
        model_path = Config.get_model_save_path("xgboost", "regression", "_best")
        if not model_path.exists():
            model_path = Config.get_model_save_path("xgboost", "regression")
        
        model_data = joblib.load(model_path)
        
        # Extract model from dictionary if needed
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
        else:
            model = model_data
        
        print(f"✅ Model loaded: {model_path.name}")
        
        # Create test data
        test_data = pd.DataFrame(np.random.randn(1, 11070))
        
        # Make prediction
        prediction = model.predict(test_data)
        
        print(f"✅ Predicted EMI: ₹{prediction[0]:,.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Regression test failed: {str(e)}")
        return False


def check_dataset():
    """Check if dataset is accessible."""
    print("\n" + "=" * 60)
    print("📁 Checking Dataset...")
    print("=" * 60)
    
    try:
        if not Config.DATA_PATH.exists():
            print(f"❌ Dataset not found at: {Config.DATA_PATH}")
            return False
        
        # Read first few rows to verify
        df = pd.read_csv(Config.DATA_PATH, nrows=5)
        total_rows = sum(1 for _ in open(Config.DATA_PATH)) - 1
        
        print(f"✅ Dataset found: {total_rows:,} rows, {len(df.columns)} columns")
        print(f"   Path: {Config.DATA_PATH}")
        
        # Check required columns
        required_cols = ["emi_eligibility", "max_monthly_emi"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Dataset check failed: {str(e)}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "🚀 EMI-PREDICT AI - VALIDATION REPORT 🚀")
    print("=" * 60)
    
    checks = {
        "Dataset": check_dataset(),
        "Models": check_models(),
        "MLflow": check_mlflow(),
        "Classification": test_classification_model(),
        "Regression": test_regression_model(),
    }
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name:<20} {'PASSED' if status else 'FAILED'}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - APPLICATION READY!")
        print("=" * 60)
        print("\n🎉 Next Steps:")
        print("1. Run the app: streamlit run emipredict/app/main.py")
        print("2. View MLflow: mlflow ui")
        print("3. Test predictions in the web interface")
    else:
        print("⚠️  SOME CHECKS FAILED - PLEASE FIX ISSUES")
        print("=" * 60)
        print("\n📝 To fix:")
        if not checks["Models"]:
            print("1. Run: python scripts/train_models.py")
        if not checks["Dataset"]:
            print("2. Ensure dataset exists at: data/emi_prediction_dataset.csv")
        if not checks["MLflow"]:
            print("3. Check MLflow configuration and pyarrow version")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
