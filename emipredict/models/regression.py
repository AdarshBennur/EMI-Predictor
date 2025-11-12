"""
Regression models for EMI amount prediction.

This module implements multiple regression algorithms for predicting
the maximum monthly EMI amount an applicant can afford.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

from emipredict.config.settings import Config
from emipredict.mlflow_utils.tracker import (
    start_experiment, log_params, log_model_metrics,
    log_model_artifacts, register_model, end_run
)
from emipredict.utils.helpers import save_model

# Setup logging
logger = logging.getLogger(__name__)


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Tuple of (trained model, metrics dict)
        
    Example:
        >>> model, metrics = train_linear_regression(X_train, y_train, X_val, y_val)
    """
    logger.info("=" * 80)
    logger.info("Training Linear Regression")
    logger.info("=" * 80)
    
    with start_experiment(
        Config.MLFLOW_EXPERIMENT_REGRESSION,
        run_name="linear_regression",
        tags={"model_type": "linear_regression"}
    ):
        # Use default parameters
        model = LinearRegression(**Config.LINEAR_REG_PARAMS)
        model.fit(X_train, y_train)
        
        log_params(Config.LINEAR_REG_PARAMS)
        
        # Log model info
        log_params({
            "model_type": "LinearRegression",
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train)
        })
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        
        # Calculate and log metrics
        train_metrics = log_model_metrics(
            y_train, y_train_pred,
            task="regression", prefix="train_"
        )
        
        val_metrics = log_model_metrics(
            y_val, y_val_pred,
            task="regression", prefix="val_"
        )
        
        # Log artifacts
        log_model_artifacts(
            model, y_val, y_val_pred,
            feature_names=X_train.columns.tolist(),
            task="regression"
        )
        
        # Save model
        model_path = Config.get_model_save_path(
            "linear_regression", "regression"
        )
        save_model(model, model_path, metadata={
            "model_type": "LinearRegression",
            "val_rmse": val_metrics['val_rmse'],
            "feature_names": X_train.columns.tolist()
        })
        
        # Log model summary
        logger.info(f"Training RMSE: {train_metrics['train_rmse']:.2f}")
        logger.info(f"Training MAE: {train_metrics['train_mae']:.2f}")
        logger.info(f"Training R²: {train_metrics['train_r2_score']:.4f}")
        logger.info(f"Validation RMSE: {val_metrics['val_rmse']:.2f}")
        logger.info(f"Validation MAE: {val_metrics['val_mae']:.2f}")
        logger.info(f"Validation R²: {val_metrics['val_r2_score']:.4f}")
        
        end_run()
    
    return model, val_metrics


def train_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameter_tuning: bool = False
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train Random Forest regressor.
    
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
    logger.info("Training Random Forest Regressor")
    logger.info("=" * 80)
    
    with start_experiment(
        Config.MLFLOW_EXPERIMENT_REGRESSION,
        run_name="random_forest",
        tags={"model_type": "random_forest"}
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
                RandomForestRegressor(
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1
                ),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
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
            model = RandomForestRegressor(**Config.RF_REG_PARAMS)
            model.fit(X_train, y_train)
            log_params(Config.RF_REG_PARAMS)
        
        # Log model info
        log_params({
            "model_type": "RandomForestRegressor",
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train)
        })
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        
        # Calculate and log metrics
        train_metrics = log_model_metrics(
            y_train, y_train_pred,
            task="regression", prefix="train_"
        )
        
        val_metrics = log_model_metrics(
            y_val, y_val_pred,
            task="regression", prefix="val_"
        )
        
        # Log artifacts
        log_model_artifacts(
            model, y_val, y_val_pred,
            feature_names=X_train.columns.tolist(),
            task="regression"
        )
        
        # Save model
        model_path = Config.get_model_save_path(
            "random_forest", "regression"
        )
        save_model(model, model_path, metadata={
            "model_type": "RandomForestRegressor",
            "val_rmse": val_metrics['val_rmse'],
            "feature_names": X_train.columns.tolist()
        })
        
        # Log model summary
        logger.info(f"Training RMSE: {train_metrics['train_rmse']:.2f}")
        logger.info(f"Training MAE: {train_metrics['train_mae']:.2f}")
        logger.info(f"Training R²: {train_metrics['train_r2_score']:.4f}")
        logger.info(f"Validation RMSE: {val_metrics['val_rmse']:.2f}")
        logger.info(f"Validation MAE: {val_metrics['val_mae']:.2f}")
        logger.info(f"Validation R²: {val_metrics['val_r2_score']:.4f}")
        
        end_run()
    
    return model, val_metrics


def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameter_tuning: bool = False
) -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    Train XGBoost regressor.
    
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
    logger.info("Training XGBoost Regressor")
    logger.info("=" * 80)
    
    with start_experiment(
        Config.MLFLOW_EXPERIMENT_REGRESSION,
        run_name="xgboost",
        tags={"model_type": "xgboost"}
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
                XGBRegressor(
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1
                ),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
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
            model = XGBRegressor(
                **Config.XGBOOST_REG_PARAMS,
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
            
            log_params(Config.XGBOOST_REG_PARAMS)
        
        # Log model info
        log_params({
            "model_type": "XGBRegressor",
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train)
        })
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        
        # Calculate and log metrics
        train_metrics = log_model_metrics(
            y_train, y_train_pred,
            task="regression", prefix="train_"
        )
        
        val_metrics = log_model_metrics(
            y_val, y_val_pred,
            task="regression", prefix="val_"
        )
        
        # Log artifacts
        log_model_artifacts(
            model, y_val, y_val_pred,
            feature_names=X_train.columns.tolist(),
            task="regression"
        )
        
        # Save model
        model_path = Config.get_model_save_path(
            "xgboost", "regression"
        )
        save_model(model, model_path, metadata={
            "model_type": "XGBRegressor",
            "val_rmse": val_metrics['val_rmse'],
            "feature_names": X_train.columns.tolist()
        })
        
        # Log model summary
        logger.info(f"Training RMSE: {train_metrics['train_rmse']:.2f}")
        logger.info(f"Training MAE: {train_metrics['train_mae']:.2f}")
        logger.info(f"Training R²: {train_metrics['train_r2_score']:.4f}")
        logger.info(f"Validation RMSE: {val_metrics['val_rmse']:.2f}")
        logger.info(f"Validation MAE: {val_metrics['val_mae']:.2f}")
        logger.info(f"Validation R²: {val_metrics['val_r2_score']:.4f}")
        
        # Check if target RMSE met
        if val_metrics['val_rmse'] <= Config.MAX_REGRESSION_RMSE:
            logger.info(
                f"✓ Target RMSE achieved: {val_metrics['val_rmse']:.2f} "
                f"<= {Config.MAX_REGRESSION_RMSE}"
            )
        else:
            logger.warning(
                f"✗ Target RMSE not met: {val_metrics['val_rmse']:.2f} "
                f"> {Config.MAX_REGRESSION_RMSE}"
            )
        
        end_run()
    
    return model, val_metrics


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    hyperparameter_tuning: bool = False
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all regression models and return results.
    
    This is the main function to use for training regression models.
    
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
        >>> results = train_regression_models(
        ...     X_train, y_train, X_val, y_val, X_test, y_test
        ... )
        >>> best_model = results['xgboost'][0]
    """
    logger.info("=" * 80)
    logger.info("Training All Regression Models")
    logger.info("=" * 80)
    
    results = {}
    
    # Train Linear Regression
    logger.info("\n1/3: Training Linear Regression...")
    lr_model, lr_metrics = train_linear_regression(
        X_train, y_train, X_val, y_val
    )
    results['linear_regression'] = (lr_model, lr_metrics)
    
    # Train Random Forest
    logger.info("\n2/3: Training Random Forest...")
    rf_model, rf_metrics = train_random_forest_regressor(
        X_train, y_train, X_val, y_val, hyperparameter_tuning
    )
    results['random_forest'] = (rf_model, rf_metrics)
    
    # Train XGBoost
    logger.info("\n3/3: Training XGBoost...")
    xgb_model, xgb_metrics = train_xgboost_regressor(
        X_train, y_train, X_val, y_val, hyperparameter_tuning
    )
    results['xgboost'] = (xgb_model, xgb_metrics)
    
    # Compare models
    logger.info("\n" + "=" * 80)
    logger.info("Model Comparison Summary")
    logger.info("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [m['val_rmse'] for _, m in results.values()],
        'MAE': [m['val_mae'] for _, m in results.values()],
        'R²': [m['val_r2_score'] for _, m in results.values()],
    })
    
    comparison_df = comparison_df.sort_values('RMSE', ascending=True)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    # Identify best model (lowest RMSE)
    best_model_name = comparison_df.iloc[0]['Model']
    best_rmse = comparison_df.iloc[0]['RMSE']
    
    logger.info(f"\nBest Model: {best_model_name} (RMSE: {best_rmse:.2f})")
    
    # Evaluate best model on test set if provided
    if X_test is not None and y_test is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Best Model - Test Set Evaluation")
        logger.info("=" * 80)
        
        best_model = results[best_model_name][0]
        y_test_pred = best_model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        logger.info(f"Test RMSE: {test_rmse:.2f}")
        logger.info(f"Test MAE: {test_mae:.2f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        
        # Save best model
        best_model_path = Config.get_model_save_path(
            best_model_name, "regression", "_best"
        )
        save_model(best_model, best_model_path, metadata={
            "model_name": best_model_name,
            "val_rmse": best_rmse,
            "test_rmse": test_rmse,
            "feature_names": X_train.columns.tolist()
        })
        
        logger.info(f"\nBest model saved to: {best_model_path}")
    
    logger.info("=" * 80)
    logger.info("Regression Model Training Complete")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    """
    Main execution for training regression models.
    """
    from emipredict.data.loader import load_and_preprocess_data
    from emipredict.features.engineering import create_features
    
    logger.info("Starting Regression Model Training Pipeline")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata = \
        load_and_preprocess_data(task="regression")
    
    # Feature engineering
    X_train = create_features(X_train)
    X_val = create_features(X_val)
    X_test = create_features(X_test)
    
    # Train models
    results = train_regression_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        hyperparameter_tuning=False
    )
    
    logger.info("Regression pipeline completed successfully!")

