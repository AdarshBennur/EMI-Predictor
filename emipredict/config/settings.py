"""
Central configuration management for EMI-Predict AI.

This module manages all configuration settings including paths,
hyperparameters, and environment-specific variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for EMI-Predict AI."""
    
    # ========================================================================
    # Project Paths
    # ========================================================================
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DATA_PATH: Path = PROJECT_ROOT / os.getenv(
        "DATA_PATH", "data/emi_prediction_dataset.csv"
    )
    MODELS_DIR: Path = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    PLOTS_DIR: Path = PROJECT_ROOT / "plots"
    
    # ========================================================================
    # Data Configuration
    # ========================================================================
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TRAIN_SPLIT: float = float(os.getenv("TRAIN_TEST_SPLIT_RATIO", "0.70"))
    VAL_TEST_SPLIT: float = float(os.getenv("VAL_TEST_SPLIT_RATIO", "0.50"))
    
    # ========================================================================
    # Target Variables
    # ========================================================================
    CLASSIFICATION_TARGET: str = os.getenv(
        "CLASSIFICATION_TARGET", "emi_eligibility"
    )
    REGRESSION_TARGET: str = os.getenv(
        "REGRESSION_TARGET", "max_monthly_emi"
    )
    
    # ========================================================================
    # Feature Engineering
    # ========================================================================
    CREATE_DERIVED_FEATURES: bool = os.getenv(
        "CREATE_DERIVED_FEATURES", "True"
    ).lower() == "true"
    FEATURE_SELECTION_THRESHOLD: float = float(
        os.getenv("FEATURE_SELECTION_THRESHOLD", "0.01")
    )
    ENABLE_FEATURE_SCALING: bool = os.getenv(
        "ENABLE_FEATURE_SCALING", "True"
    ).lower() == "true"
    SCALING_METHOD: str = os.getenv("SCALING_METHOD", "standard")
    
    # ========================================================================
    # MLflow Configuration
    # ========================================================================
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "file:./mlruns"
    )
    MLFLOW_EXPERIMENT_CLASSIFICATION: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME_CLASSIFICATION",
        "emi_eligibility_classification"
    )
    MLFLOW_EXPERIMENT_REGRESSION: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME_REGRESSION",
        "emi_amount_regression"
    )
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    N_HYPERPARAMETER_TRIALS: int = int(
        os.getenv("N_HYPERPARAMETER_TRIALS", "50")
    )
    CROSS_VALIDATION_FOLDS: int = int(
        os.getenv("CROSS_VALIDATION_FOLDS", "5")
    )
    ENABLE_EARLY_STOPPING: bool = os.getenv(
        "ENABLE_EARLY_STOPPING", "True"
    ).lower() == "true"
    EARLY_STOPPING_ROUNDS: int = int(
        os.getenv("EARLY_STOPPING_ROUNDS", "10")
    )
    
    # ========================================================================
    # Model Hyperparameters - XGBoost Classification
    # ========================================================================
    XGBOOST_CLF_PARAMS: Dict[str, Any] = {
        "learning_rate": float(os.getenv("XGBOOST_CLF_LEARNING_RATE", "0.1")),
        "max_depth": int(os.getenv("XGBOOST_CLF_MAX_DEPTH", "6")),
        "n_estimators": int(os.getenv("XGBOOST_CLF_N_ESTIMATORS", "100")),
        "subsample": float(os.getenv("XGBOOST_CLF_SUBSAMPLE", "0.8")),
        "colsample_bytree": float(
            os.getenv("XGBOOST_CLF_COLSAMPLE_BYTREE", "0.8")
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    
    # ========================================================================
    # Model Hyperparameters - XGBoost Regression
    # ========================================================================
    XGBOOST_REG_PARAMS: Dict[str, Any] = {
        "learning_rate": float(os.getenv("XGBOOST_REG_LEARNING_RATE", "0.1")),
        "max_depth": int(os.getenv("XGBOOST_REG_MAX_DEPTH", "6")),
        "n_estimators": int(os.getenv("XGBOOST_REG_N_ESTIMATORS", "100")),
        "subsample": float(os.getenv("XGBOOST_REG_SUBSAMPLE", "0.8")),
        "colsample_bytree": float(
            os.getenv("XGBOOST_REG_COLSAMPLE_BYTREE", "0.8")
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    
    # ========================================================================
    # Model Hyperparameters - Random Forest Classification
    # ========================================================================
    RF_CLF_PARAMS: Dict[str, Any] = {
        "n_estimators": int(os.getenv("RF_CLF_N_ESTIMATORS", "100")),
        "max_depth": int(os.getenv("RF_CLF_MAX_DEPTH", "15")),
        "min_samples_split": int(os.getenv("RF_CLF_MIN_SAMPLES_SPLIT", "2")),
        "min_samples_leaf": int(os.getenv("RF_CLF_MIN_SAMPLES_LEAF", "1")),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    
    # ========================================================================
    # Model Hyperparameters - Random Forest Regression
    # ========================================================================
    RF_REG_PARAMS: Dict[str, Any] = {
        "n_estimators": int(os.getenv("RF_REG_N_ESTIMATORS", "100")),
        "max_depth": int(os.getenv("RF_REG_MAX_DEPTH", "15")),
        "min_samples_split": int(os.getenv("RF_REG_MIN_SAMPLES_SPLIT", "2")),
        "min_samples_leaf": int(os.getenv("RF_REG_MIN_SAMPLES_LEAF", "1")),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    
    # ========================================================================
    # Logistic Regression Parameters
    # ========================================================================
    LOGISTIC_PARAMS: Dict[str, Any] = {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    
    # ========================================================================
    # Linear Regression Parameters
    # ========================================================================
    LINEAR_REG_PARAMS: Dict[str, Any] = {
        "n_jobs": -1,
    }
    
    # ========================================================================
    # Performance Configuration
    # ========================================================================
    N_JOBS: int = int(os.getenv("N_JOBS", "-1"))
    CACHE_SIZE_MB: int = int(os.getenv("CACHE_SIZE_MB", "500"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1000"))
    
    # ========================================================================
    # Logging Configuration
    # ========================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = PROJECT_ROOT / os.getenv(
        "LOG_FILE", "logs/emipredict.log"
    )
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ENABLE_CONSOLE_LOGGING: bool = os.getenv(
        "ENABLE_CONSOLE_LOGGING", "True"
    ).lower() == "true"
    ENABLE_FILE_LOGGING: bool = os.getenv(
        "ENABLE_FILE_LOGGING", "True"
    ).lower() == "true"
    
    # ========================================================================
    # Data Quality Configuration
    # ========================================================================
    OUTLIER_DETECTION_METHOD: str = os.getenv(
        "OUTLIER_DETECTION_METHOD", "iqr"
    )
    OUTLIER_THRESHOLD: float = float(
        os.getenv("OUTLIER_THRESHOLD", "1.5")
    )
    MISSING_VALUE_THRESHOLD: float = float(
        os.getenv("MISSING_VALUE_THRESHOLD", "0.3")
    )
    MIN_SAMPLES_FOR_TRAINING: int = int(
        os.getenv("MIN_SAMPLES_FOR_TRAINING", "1000")
    )
    
    # ========================================================================
    # Model Performance Thresholds
    # ========================================================================
    MIN_CLASSIFICATION_ACCURACY: float = float(
        os.getenv("MIN_CLASSIFICATION_ACCURACY", "0.90")
    )
    MAX_REGRESSION_RMSE: float = float(
        os.getenv("MAX_REGRESSION_RMSE", "2000")
    )
    MIN_CLASSIFICATION_PRECISION: float = float(
        os.getenv("MIN_CLASSIFICATION_PRECISION", "0.85")
    )
    MIN_CLASSIFICATION_RECALL: float = float(
        os.getenv("MIN_CLASSIFICATION_RECALL", "0.85")
    )
    
    # ========================================================================
    # Streamlit Configuration
    # ========================================================================
    STREAMLIT_SERVER_PORT: int = int(
        os.getenv("STREAMLIT_SERVER_PORT", "8501")
    )
    STREAMLIT_SERVER_ADDRESS: str = os.getenv(
        "STREAMLIT_SERVER_ADDRESS", "localhost"
    )
    STREAMLIT_THEME: str = os.getenv("STREAMLIT_THEME", "light")
    
    # ========================================================================
    # Environment Configuration
    # ========================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create necessary directories if they don't exist.
        
        This method should be called during application initialization
        to ensure all required directories are present.
        """
        directories = [
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.PLOTS_DIR,
            cls.DATA_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create .gitkeep files to preserve empty directories in git
        for directory in [cls.MODELS_DIR, cls.LOGS_DIR, cls.PLOTS_DIR]:
            gitkeep = directory / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
    
    @classmethod
    def validate_config(cls) -> None:
        """
        Validate configuration settings.
        
        Raises:
            ValueError: If required files or settings are invalid
            FileNotFoundError: If required files don't exist
        """
        # Validate data file exists
        if not cls.DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {cls.DATA_PATH}. "
                f"Please ensure the data file is present."
            )
        
        # Validate split ratios
        if not (0 < cls.TRAIN_SPLIT < 1):
            raise ValueError(
                f"TRAIN_SPLIT must be between 0 and 1, got {cls.TRAIN_SPLIT}"
            )
        
        if not (0 < cls.VAL_TEST_SPLIT < 1):
            raise ValueError(
                f"VAL_TEST_SPLIT must be between 0 and 1, "
                f"got {cls.VAL_TEST_SPLIT}"
            )
        
        # Validate threshold values
        if cls.MIN_CLASSIFICATION_ACCURACY < 0 or cls.MIN_CLASSIFICATION_ACCURACY > 1:
            raise ValueError(
                f"MIN_CLASSIFICATION_ACCURACY must be between 0 and 1"
            )
    
    @classmethod
    def get_model_save_path(
        cls,
        model_type: str,
        task: str,
        suffix: str = ""
    ) -> Path:
        """
        Get the save path for a model file.
        
        Args:
            model_type: Type of model (e.g., 'xgboost', 'random_forest')
            task: Task type ('classification' or 'regression')
            suffix: Optional suffix for the filename
            
        Returns:
            Path object for the model file
        """
        filename = f"{model_type}_{task}{suffix}.pkl"
        return cls.MODELS_DIR / filename
    
    @classmethod
    def initialize(cls) -> None:
        """
        Initialize configuration by creating directories and validating settings.
        
        This method should be called at application startup.
        """
        cls.ensure_directories()
        cls.validate_config()


# Initialize on import
try:
    Config.initialize()
except FileNotFoundError as e:
    # Data file might not exist in certain contexts (e.g., initial setup)
    # Just ensure directories exist
    Config.ensure_directories()

