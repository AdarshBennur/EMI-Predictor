"""
Preprocessing Pipeline for EMI-Predict AI.

This module provides a unified preprocessing pipeline that ensures
consistency between training and prediction by saving all transformation
steps and applying them in the exact same order.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class EMIPreprocessingPipeline:
    """
    Complete preprocessing pipeline for EMI prediction.
    
    This pipeline handles all data transformations consistently:
    1. Missing value imputation
    2. Categorical encoding
    3. Feature engineering
    4. Feature selection
    
    The pipeline can be saved after training and loaded for prediction
    to ensure exact feature name and order consistency.
    """
    
    def __init__(self):
        """Initialize the preprocessing pipeline."""
        self.is_fitted = False
        self.categorical_encoders = {}
        self.feature_columns = None  # Column names after preprocessing
        self.feature_selector = None
        self.selected_columns = None
        
    def fit(self, df: pd.DataFrame) -> 'EMIPreprocessingPipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            df: Raw input DataFrame with all columns including targets
            
        Returns:
            self (fitted pipeline)
        """
        logger.info("Fitting preprocessing pipeline...")
        
        # Store categorical columns for encoding
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target columns from categorical list
        targets = ['emi_eligibility', 'max_monthly_emi']
        self.categorical_columns = [col for col in self.categorical_columns if col not in targets]
        
        # Fit label encoders for each categorical column
        for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.categorical_encoders[col] = le
            logger.info(f"Fitted encoder for {col}: {len(le.classes_)} classes")
        
        # Apply transformations to get feature column names
        df_transformed = self._transform_data(df.copy())
        
        # Store feature columns (without targets)
        targets_present = [t for t in targets if t in df_transformed.columns]
        self.feature_columns = [col for col in df_transformed.columns if col not in targets_present]
        
        self.is_fitted = True
        logger.info(f"Pipeline fitted. Feature columns: {len(self.feature_columns)}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Transformed DataFrame with engineered features
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform. Call fit() first.")
        
        logger.info("Transforming data with fitted pipeline...")
        
        # Apply transformations
        df_transformed = self._transform_data(df.copy())
        
        # Ensure we have all required columns
        for col in self.feature_columns:
            if col not in df_transformed.columns:
                df_transformed[col] = 0
                logger.warning(f"Missing feature {col}, filled with 0")
        
        # Select only the features that were present during training
        df_final = df_transformed[self.feature_columns]
        
        logger.info(f"Transform completed. Output shape: {df_final.shape}")
        
        return df_final
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to apply all transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        # Step 1: Handle missing values
        df = self._handle_missing(df)
        
        # Step 2: Encode categorical features
        df = self._encode_categorical(df)
        
        # Step 3: Feature engineering
        df = self._engineer_features(df)
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        from sklearn.impute import SimpleImputer
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove targets from imputation
        targets = ['emi_eligibility', 'max_monthly_emi']
        numeric_cols = [col for col in numeric_cols if col not in targets]
        categorical_cols = [col for col in categorical_cols if col not in targets]
        
        # Impute numeric
        if numeric_cols:
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Impute categorical
        if categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using stored encoders."""
        for col in self.categorical_columns:
            if col in df.columns:
                # Use fitted encoder
                df[col] = self.categorical_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        from emipredict.features.engineering import create_features
        return create_features(df)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline and transform data in one step.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: Path) -> None:
        """
        Save the fitted pipeline to disk.
        
        Args:
            path: Path to save the pipeline
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline. Call fit() first.")
        
        joblib.dump(self, path)
        logger.info(f"Pipeline saved to {path}")
    
    @staticmethod
    def load(path: Path) -> 'EMIPreprocessingPipeline':
        """
        Load a fitted pipeline from disk.
        
        Args:
            path: Path to the saved pipeline
            
        Returns:
            Loaded pipeline
        """
        pipeline = joblib.load(path)
        logger.info(f"Pipeline loaded from {path}")
        return pipeline
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names after transformation.
        
        Returns:
            List of feature column names
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first.")
        
        return self.feature_columns.copy()

