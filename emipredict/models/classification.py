"""
Classification models for EMI eligibility prediction.

This module implements multiple classification algorithms for predicting
whether an applicant is eligible for an EMI.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

from emipredict.config.settings import Config
from emipredict.mlflow_utils.tracker import (
    start_experiment, log_params, log_model_metrics,
    log_model_artifacts, register_model, end_run
)
from emipredict.utils.helpers import save_model

# Setup logging
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameter_tuning: bool = False
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train Logistic Regression classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Tuple of (trained model, metrics dict)
        
    Example:
        >>> model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    """
    logger.info("=" * 80)
    logger.info("Training Logistic Regression")
    logger.info("=" * 80)
    
    with start_experiment(
        Config.MLFLOW_EXPERIMENT_CLASSIFICATION,
        run_name="logistic_regression",
        tags={"model_type": "logistic_regression", "n_classes": "3"}
    ):
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1
                ),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            log_params(best_params)
        else:
            # Use default parameters
            model = LogisticRegression(**Config.LOGISTIC_PARAMS)
            model.fit(X_train, y_train)
            log_params(Config.LOGISTIC_PARAMS)
        
        # Log model info
        log_params({
            "model_type": "LogisticRegression",
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train)
        })
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        
        # Calculate and log metrics
        train_metrics = log_model_metrics(
            y_train, y_train_pred, y_train_proba,
            task="classification", prefix="train_"
        )
        
        val_metrics = log_model_metrics(
            y_val, y_val_pred, y_val_proba,
            task="classification", prefix="val_"
        )
        
        # Log artifacts with class names for 3-class classification
        class_names = ['Eligible', 'High_Risk', 'Not_Eligible']
        log_model_artifacts(
            model, y_val, y_val_pred, y_val_proba,
            feature_names=X_train.columns.tolist(),
            task="classification",
            class_names=class_names
        )
        
        # Save model
        model_path = Config.get_model_save_path(
            "logistic_regression", "classification"
        )
        save_model(model, model_path, metadata={
            "model_type": "LogisticRegression",
            "val_accuracy": val_metrics['val_accuracy'],
            "feature_names": X_train.columns.tolist()
        })
        
        # Log model summary
        logger.info(f"Training Accuracy: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"Validation Accuracy: {val_metrics['val_accuracy']:.4f}")
        logger.info(f"Validation Precision: {val_metrics['val_precision']:.4f}")
        logger.info(f"Validation Recall: {val_metrics['val_recall']:.4f}")
        logger.info(f"Validation F1-Score: {val_metrics['val_f1_score']:.4f}")
        
        end_run()
    
    return model, val_metrics


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameter_tuning: bool = False
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    logger.info("=" * 80)
    logger.info("Training Random Forest Classifier")
    logger.info("=" * 80)
    
    with start_experiment(
        Config.MLFLOW_EXPERIMENT_CLASSIFICATION,
        run_name="random_forest",
        tags={"model_type": "random_forest", "n_classes": "3"}
    ):
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1
                ),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            log_params(best_params)
        else:
            # Use default parameters
            model = RandomForestClassifier(**Config.RF_CLF_PARAMS)
            model.fit(X_train, y_train)
            log_params(Config.RF_CLF_PARAMS)
        
        # Log model info
        log_params({
            "model_type": "RandomForestClassifier",
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train)
        })
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        
        # Calculate and log metrics
        train_metrics = log_model_metrics(
            y_train, y_train_pred, y_train_proba,
            task="classification", prefix="train_"
        )
        
        val_metrics = log_model_metrics(
            y_val, y_val_pred, y_val_proba,
            task="classification", prefix="val_"
        )
        
        # Log artifacts with class names for 3-class classification
        class_names = ['Eligible', 'High_Risk', 'Not_Eligible']
        log_model_artifacts(
            model, y_val, y_val_pred, y_val_proba,
            feature_names=X_train.columns.tolist(),
            task="classification",
            class_names=class_names
        )
        
        # Save model
        model_path = Config.get_model_save_path(
            "random_forest", "classification"
        )
        save_model(model, model_path, metadata={
            "model_type": "RandomForestClassifier",
            "val_accuracy": val_metrics['val_accuracy'],
            "feature_names": X_train.columns.tolist()
        })
        
        # Log model summary
        logger.info(f"Training Accuracy: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"Validation Accuracy: {val_metrics['val_accuracy']:.4f}")
        logger.info(f"Validation Precision: {val_metrics['val_precision']:.4f}")
        logger.info(f"Validation Recall: {val_metrics['val_recall']:.4f}")
        logger.info(f"Validation F1-Score: {val_metrics['val_f1_score']:.4f}")
        
        end_run()
    
    return model, val_metrics


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameter_tuning: bool = False
) -> Tuple[XGBClassifier, Dict[str, float]]:
    """
    Train XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    logger.info("=" * 80)
    logger.info("Training XGBoost Classifier")
    logger.info("=" * 80)
    
    with start_experiment(
        Config.MLFLOW_EXPERIMENT_CLASSIFICATION,
        run_name="xgboost",
        tags={"model_type": "xgboost", "n_classes": "3"}
    ):
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            
            param_grid = {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                XGBClassifier(
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1,
                    eval_metric='logloss'
                ),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            log_params(best_params)
        else:
            # Use default parameters with early stopping
            # Configure for 3-class classification
            xgb_params = Config.XGBOOST_CLF_PARAMS.copy()
            xgb_params['objective'] = 'multi:softprob'  # Multi-class with probabilities
            xgb_params['num_class'] = 3  # 3 classes: Eligible, High_Risk, Not_Eligible
            xgb_params['eval_metric'] = 'mlogloss'  # Multi-class log loss
            
            model = XGBClassifier(
                **xgb_params,
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS if Config.ENABLE_EARLY_STOPPING else None
            )
            
            # Fit with eval set for early stopping
            if Config.ENABLE_EARLY_STOPPING:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            log_params(Config.XGBOOST_CLF_PARAMS)
        
        # Log model info
        log_params({
            "model_type": "XGBClassifier",
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train)
        })
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        
        # Calculate and log metrics
        train_metrics = log_model_metrics(
            y_train, y_train_pred, y_train_proba,
            task="classification", prefix="train_"
        )
        
        val_metrics = log_model_metrics(
            y_val, y_val_pred, y_val_proba,
            task="classification", prefix="val_"
        )
        
        # Log artifacts with class names for 3-class classification
        class_names = ['Eligible', 'High_Risk', 'Not_Eligible']
        log_model_artifacts(
            model, y_val, y_val_pred, y_val_proba,
            feature_names=X_train.columns.tolist(),
            task="classification",
            class_names=class_names
        )
        
        # Save model
        model_path = Config.get_model_save_path(
            "xgboost", "classification"
        )
        save_model(model, model_path, metadata={
            "model_type": "XGBClassifier",
            "val_accuracy": val_metrics['val_accuracy'],
            "feature_names": X_train.columns.tolist()
        })
        
        # Log model summary
        logger.info(f"Training Accuracy: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"Validation Accuracy: {val_metrics['val_accuracy']:.4f}")
        logger.info(f"Validation Precision: {val_metrics['val_precision']:.4f}")
        logger.info(f"Validation Recall: {val_metrics['val_recall']:.4f}")
        logger.info(f"Validation F1-Score: {val_metrics['val_f1_score']:.4f}")
        
        # Check if target accuracy met
        if val_metrics['val_accuracy'] >= Config.MIN_CLASSIFICATION_ACCURACY:
            logger.info(
                f"✓ Target accuracy achieved: {val_metrics['val_accuracy']:.4f} "
                f">= {Config.MIN_CLASSIFICATION_ACCURACY}"
            )
        else:
            logger.warning(
                f"✗ Target accuracy not met: {val_metrics['val_accuracy']:.4f} "
                f"< {Config.MIN_CLASSIFICATION_ACCURACY}"
            )
        
        end_run()
    
    return model, val_metrics


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    hyperparameter_tuning: bool = False
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all classification models and return results.
    
    This is the main function to use for training classification models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features (optional)
        y_test: Test target (optional)
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary with model names as keys and (model, metrics) tuples as values
        
    Example:
        >>> results = train_classification_models(
        ...     X_train, y_train, X_val, y_val, X_test, y_test
        ... )
        >>> best_model = results['xgboost'][0]
    """
    logger.info("=" * 80)
    logger.info("Training All Classification Models")
    logger.info("=" * 80)
    
    results = {}
    
    # Train Logistic Regression
    logger.info("\n1/3: Training Logistic Regression...")
    lr_model, lr_metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val, hyperparameter_tuning
    )
    results['logistic_regression'] = (lr_model, lr_metrics)
    
    # Train Random Forest
    logger.info("\n2/3: Training Random Forest...")
    rf_model, rf_metrics = train_random_forest_classifier(
        X_train, y_train, X_val, y_val, hyperparameter_tuning
    )
    results['random_forest'] = (rf_model, rf_metrics)
    
    # Train XGBoost
    logger.info("\n3/3: Training XGBoost...")
    xgb_model, xgb_metrics = train_xgboost_classifier(
        X_train, y_train, X_val, y_val, hyperparameter_tuning
    )
    results['xgboost'] = (xgb_model, xgb_metrics)
    
    # Compare models
    logger.info("\n" + "=" * 80)
    logger.info("Model Comparison Summary")
    logger.info("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [m['val_accuracy'] for _, m in results.values()],
        'Precision': [m['val_precision'] for _, m in results.values()],
        'Recall': [m['val_recall'] for _, m in results.values()],
        'F1-Score': [m['val_f1_score'] for _, m in results.values()],
    })
    
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Accuracy']
    
    logger.info(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # Evaluate best model on test set if provided
    if X_test is not None and y_test is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Best Model - Test Set Evaluation")
        logger.info("=" * 80)
        
        best_model = results[best_model_name][0]
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1-Score: {test_f1:.4f}")
        
        # Save best model
        best_model_path = Config.get_model_save_path(
            best_model_name, "classification", "_best"
        )
        save_model(best_model, best_model_path, metadata={
            "model_name": best_model_name,
            "val_accuracy": best_accuracy,
            "test_accuracy": test_accuracy,
            "feature_names": X_train.columns.tolist()
        })
        
        logger.info(f"\nBest model saved to: {best_model_path}")
    
    logger.info("=" * 80)
    logger.info("Classification Model Training Complete")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    """
    Main execution for training classification models.
    """
    from emipredict.data.loader import load_and_preprocess_data
    from emipredict.features.engineering import create_features
    
    logger.info("Starting Classification Model Training Pipeline")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata = \
        load_and_preprocess_data(task="classification")
    
    # Feature engineering
    X_train = create_features(X_train)
    X_val = create_features(X_val)
    X_test = create_features(X_test)
    
    # Train models
    results = train_classification_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        hyperparameter_tuning=False
    )
    
    logger.info("Classification pipeline completed successfully!")

