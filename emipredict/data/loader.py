"""
Data loading and preprocessing module for EMI-Predict AI.

This module provides functions for:
- Loading dataset from CSV
- Handling missing values
- Detecting and treating outliers
- Encoding categorical variables
- Scaling numerical features
- Splitting data into train/validation/test sets
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from emipredict.config.settings import Config

# Setup logging
logger = logging.getLogger(__name__)


def load_data(
    file_path: Optional[Path] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file. If None, uses Config.DATA_PATH
        nrows: Number of rows to load. If None, loads all rows
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        
    Example:
        >>> df = load_data()
        >>> print(f"Loaded {len(df)} records with {len(df.columns)} features")
    """
    if file_path is None:
        file_path = Config.DATA_PATH
    
    logger.info(f"Loading data from {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        logger.info(
            f"Successfully loaded {len(df)} records with "
            f"{len(df.columns)} columns"
        )
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Data file is empty: {file_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate dataset structure and content.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating dataset")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check minimum samples
    if len(df) < Config.MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Dataset has only {len(df)} records. "
            f"Minimum required: {Config.MIN_SAMPLES_FOR_TRAINING}"
        )
    
    # Check for target columns
    required_columns = [
        Config.CLASSIFICATION_TARGET,
        Config.REGRESSION_TARGET
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("Dataset validation passed")


def handle_missing_values(
    df: pd.DataFrame,
    strategy_numeric: str = "median",
    strategy_categorical: str = "most_frequent"
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy_numeric: Strategy for numeric columns ('mean', 'median', 'constant')
        strategy_categorical: Strategy for categorical columns ('most_frequent', 'constant')
        
    Returns:
        DataFrame with imputed missing values
        
    Example:
        >>> df_clean = handle_missing_values(df)
        >>> assert not df_clean.isnull().any().any()
    """
    logger.info("Handling missing values")
    df = df.copy()
    
    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target columns from imputation
    target_cols = [Config.CLASSIFICATION_TARGET, Config.REGRESSION_TARGET]
    numeric_columns = [col for col in numeric_columns if col not in target_cols]
    categorical_columns = [col for col in categorical_columns if col not in target_cols]
    
    # Log missing values before imputation
    missing_before = df[numeric_columns + categorical_columns].isnull().sum()
    cols_with_missing = missing_before[missing_before > 0]
    if not cols_with_missing.empty:
        logger.info(f"Columns with missing values:\n{cols_with_missing}")
    
    # Impute numeric columns
    if numeric_columns:
        imputer_numeric = SimpleImputer(strategy=strategy_numeric)
        df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])
        logger.info(
            f"Imputed {len(numeric_columns)} numeric columns "
            f"using {strategy_numeric} strategy"
        )
    
    # Impute categorical columns
    if categorical_columns:
        imputer_categorical = SimpleImputer(strategy=strategy_categorical)
        df[categorical_columns] = imputer_categorical.fit_transform(
            df[categorical_columns]
        )
        logger.info(
            f"Imputed {len(categorical_columns)} categorical columns "
            f"using {strategy_categorical} strategy"
        )
    
    return df


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers in numerical columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check. If None, checks all numeric columns
        method: Method for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with boolean mask indicating outliers
    """
    logger.info(f"Detecting outliers using {method} method")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target columns
        columns = [
            col for col in columns 
            if col not in [Config.CLASSIFICATION_TARGET, Config.REGRESSION_TARGET]
        ]
    
    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
    
    if method == "iqr":
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    elif method == "zscore":
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask[col] = z_scores > threshold
    
    # Log outlier counts
    outlier_counts = outlier_mask.sum()
    cols_with_outliers = outlier_counts[outlier_counts > 0]
    if not cols_with_outliers.empty:
        logger.info(f"Outliers detected:\n{cols_with_outliers}")
    
    return outlier_mask


def handle_outliers(
    df: pd.DataFrame,
    method: str = "cap",
    detection_method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Handle outliers in numerical columns.
    
    Args:
        df: Input DataFrame
        method: How to handle outliers ('cap', 'remove', or 'none')
        detection_method: Method for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
        
    Example:
        >>> df_clean = handle_outliers(df, method='cap')
    """
    logger.info(f"Handling outliers using '{method}' method")
    df = df.copy()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target columns
    numeric_columns = [
        col for col in numeric_columns 
        if col not in [Config.CLASSIFICATION_TARGET, Config.REGRESSION_TARGET]
    ]
    
    if method == "none":
        logger.info("Outlier handling skipped")
        return df
    
    if method == "cap":
        # Cap outliers at percentiles
        for col in numeric_columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)
        logger.info(f"Capped outliers in {len(numeric_columns)} columns")
    
    elif method == "remove":
        outlier_mask = detect_outliers(
            df,
            columns=numeric_columns,
            method=detection_method,
            threshold=threshold
        )
        # Remove rows with outliers in any column
        rows_to_remove = outlier_mask.any(axis=1)
        original_len = len(df)
        df = df[~rows_to_remove]
        logger.info(
            f"Removed {original_len - len(df)} rows "
            f"({(original_len - len(df)) / original_len * 100:.2f}%)"
        )
    
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    encoding_method: str = "onehot"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        encoding_method: Encoding method ('onehot' or 'label')
        
    Returns:
        Tuple of (encoded DataFrame, encoding metadata)
        
    Example:
        >>> df_encoded, encoders = encode_categorical_features(df)
    """
    logger.info(f"Encoding categorical features using {encoding_method}")
    df = df.copy()
    encoders = {}
    
    # Identify categorical columns (excluding target if it's categorical)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if Config.CLASSIFICATION_TARGET in categorical_columns:
        categorical_columns.remove(Config.CLASSIFICATION_TARGET)
    
    if not categorical_columns:
        logger.info("No categorical columns to encode")
        return df, encoders
    
    if encoding_method == "label":
        # Label encoding for all categorical columns
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        logger.info(f"Label encoded {len(categorical_columns)} columns")
    
    elif encoding_method == "onehot":
        # One-hot encoding
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            prefix=categorical_columns,
            drop_first=True
        )
        encoders['onehot_columns'] = categorical_columns
        logger.info(
            f"One-hot encoded {len(categorical_columns)} columns, "
            f"resulting in {len(df.columns)} total columns"
        )
    
    return df, encoders


def encode_target_variable(
    y: pd.Series,
    task: str = "classification"
) -> Tuple[pd.Series, Optional[LabelEncoder]]:
    """
    Encode target variable.
    
    Args:
        y: Target Series
        task: Task type ('classification' or 'regression')
        
    Returns:
        Tuple of (encoded target, encoder or None)
        
    Example:
        >>> y_encoded, encoder = encode_target_variable(y, task='classification')
    
    Note:
        For classification, encodes to:
        - 0: Eligible (low risk)
        - 1: High_Risk (marginal case)
        - 2: Not_Eligible (high risk)
    """
    if task == "regression":
        logger.info("Regression target - no encoding needed")
        return y, None
    
    # Classification - encode if categorical
    if y.dtype == 'object':
        logger.info("Encoding 3-class classification target")
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        
        # Log class mapping
        class_mapping = {cls: le.transform([cls])[0] for cls in le.classes_}
        logger.info(f"Target classes (3-class classification):")
        logger.info(f"  Class mapping: {class_mapping}")
        logger.info(f"  Number of classes: {len(le.classes_)}")
        
        # Verify we have 3 classes
        if len(le.classes_) != 3:
            logger.warning(
                f"Expected 3 classes (Eligible, High_Risk, Not_Eligible), "
                f"but found {len(le.classes_)}: {list(le.classes_)}"
            )
        
        return y_encoded, le
    
    return y, None


def scale_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    method: str = "standard"
) -> Tuple[pd.DataFrame, ...]:
    """
    Scale numerical features using training data statistics.
    
    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        method: Scaling method ('standard', 'minmax', or 'none')
        
    Returns:
        Tuple of scaled DataFrames (train, val, test) with scaler
        
    Example:
        >>> X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        ...     X_train, X_val, X_test
        ... )
    """
    if method == "none" or not Config.ENABLE_FEATURE_SCALING:
        logger.info("Feature scaling disabled")
        return (X_train, X_val, X_test, None)
    
    logger.info(f"Scaling features using {method} method")
    
    if method == "standard":
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    # Fit only on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    result = [X_train_scaled]
    
    # Transform validation data if provided
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        result.append(X_val_scaled)
    else:
        result.append(None)
    
    # Transform test data if provided
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        result.append(X_test_scaled)
    else:
        result.append(None)
    
    result.append(scaler)
    
    logger.info("Feature scaling completed")
    return tuple(result)


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.3,
    val_size: float = 0.5,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion for test+val split (default 0.3 = 30%)
        val_size: Proportion of test set to use for validation (default 0.5)
        stratify: Whether to stratify split (for classification)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        
    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        ...     df, 'emi_eligibility'
        ... )
    """
    logger.info(
        f"Splitting data: train={1-test_size:.0%}, "
        f"val={test_size*val_size:.0%}, test={test_size*(1-val_size):.0%}"
    )
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split: train vs (val + test)
    stratify_y = y if stratify and y.dtype == 'object' else None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )
    
    # Second split: val vs test
    stratify_temp = y_temp if stratify and y.dtype == 'object' else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    logger.info(
        f"Split sizes - Train: {len(X_train)}, "
        f"Val: {len(X_val)}, Test: {len(X_test)}"
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(
    df: pd.DataFrame,
    handle_missing: bool = True,
    handle_outliers_method: str = "cap",
    encode_categorical: bool = True,
    encoding_method: str = "label"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        handle_missing: Whether to handle missing values
        handle_outliers_method: Method to handle outliers
        encode_categorical: Whether to encode categorical features
        encoding_method: Method for encoding ('label' or 'onehot')
        
    Returns:
        Tuple of (preprocessed DataFrame, metadata dict)
        
    Example:
        >>> df_processed, metadata = preprocess_data(df)
    """
    logger.info("Starting preprocessing pipeline")
    df = df.copy()
    metadata = {}
    
    # Validate data
    validate_data(df)
    
    # Handle missing values
    if handle_missing:
        df = handle_missing_values(df)
        metadata['missing_handled'] = True
    
    # Handle outliers
    if handle_outliers_method != "none":
        df = handle_outliers(
            df,
            method=handle_outliers_method,
            detection_method=Config.OUTLIER_DETECTION_METHOD,
            threshold=Config.OUTLIER_THRESHOLD
        )
        metadata['outliers_handled'] = True
    
    # Encode categorical features
    if encode_categorical:
        df, encoders = encode_categorical_features(df, encoding_method)
        metadata['encoders'] = encoders
        metadata['encoding_method'] = encoding_method
    
    logger.info("Preprocessing pipeline completed")
    return df, metadata


def load_and_preprocess_data(
    file_path: Optional[Path] = None,
    target_column: str = None,
    task: str = "classification",
    nrows: Optional[int] = None,
    scale_features_flag: bool = True
) -> Tuple:
    """
    Complete data loading and preprocessing pipeline with train/val/test split.
    
    This is the main function to use for loading and preparing data for modeling.
    
    Args:
        file_path: Path to data file
        target_column: Target column name
        task: Task type ('classification' or 'regression')
        nrows: Number of rows to load
        scale_features_flag: Whether to scale features
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata)
        
    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test, scaler, meta = \\
        ...     load_and_preprocess_data(task='classification')
    """
    logger.info("=" * 80)
    logger.info("Starting complete data loading and preprocessing pipeline")
    logger.info("=" * 80)
    
    # Set target column based on task
    if target_column is None:
        target_column = (
            Config.CLASSIFICATION_TARGET if task == "classification"
            else Config.REGRESSION_TARGET
        )
    
    # Load data
    df = load_data(file_path, nrows)
    
    # Preprocess data (except target encoding)
    df_processed, metadata = preprocess_data(
        df,
        handle_missing=True,
        handle_outliers_method="cap",
        encode_categorical=True,
        encoding_method="label"  # Label encoding works better with tree-based models
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_processed,
        target_column=target_column,
        test_size=1 - Config.TRAIN_SPLIT,
        val_size=Config.VAL_TEST_SPLIT,
        stratify=(task == "classification"),
        random_state=Config.RANDOM_STATE
    )
    
    # Encode target variable
    y_train, target_encoder = encode_target_variable(y_train, task)
    if target_encoder is not None:
        y_val = pd.Series(
            target_encoder.transform(y_val),
            index=y_val.index,
            name=y_val.name
        )
        y_test = pd.Series(
            target_encoder.transform(y_test),
            index=y_test.index,
            name=y_test.name
        )
    metadata['target_encoder'] = target_encoder
    
    # Scale features
    if scale_features_flag:
        X_train, X_val, X_test, scaler = scale_features(
            X_train, X_val, X_test, method=Config.SCALING_METHOD
        )
        metadata['scaler'] = scaler
    else:
        scaler = None
    
    # Store metadata
    metadata['task'] = task
    metadata['target_column'] = target_column
    metadata['n_features'] = X_train.shape[1]
    metadata['feature_names'] = X_train.columns.tolist()
    
    logger.info("=" * 80)
    logger.info("Data loading and preprocessing completed successfully")
    logger.info(f"Task: {task}")
    logger.info(f"Target: {target_column}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info("=" * 80)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata

