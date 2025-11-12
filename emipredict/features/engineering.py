"""
Feature engineering module for EMI-Predict AI.

This module provides functions for creating derived features that
improve model performance for EMI eligibility and amount prediction.
"""

import logging
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif

from emipredict.config.settings import Config

# Setup logging
logger = logging.getLogger(__name__)


def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived financial risk features.
    
    This function creates features that capture financial health and
    risk indicators based on income, expenses, debt, and savings.
    
    Args:
        df: Input DataFrame with raw features
        
    Returns:
        DataFrame with added derived features
        
    Example:
        >>> df_engineered = create_financial_features(df)
        >>> print(df_engineered.columns)
    """
    logger.info("Creating derived financial features")
    df = df.copy()
    
    # ========================================================================
    # Total Monthly Expenses
    # ========================================================================
    expense_columns = [
        'monthly_rent', 'groceries_utilities', 'travel_expenses',
        'other_monthly_expenses', 'school_fees', 'college_fees'
    ]
    
    # Only include columns that exist in the DataFrame
    existing_expense_cols = [col for col in expense_columns if col in df.columns]
    
    if existing_expense_cols:
        df['total_monthly_expenses'] = df[existing_expense_cols].sum(axis=1)
        logger.info("Created: total_monthly_expenses")
    
    # ========================================================================
    # Debt-to-Income Ratio (DTI)
    # ========================================================================
    if 'current_emi_amount' in df.columns and 'monthly_salary' in df.columns:
        df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary'].replace(0, np.nan)
        df['debt_to_income_ratio'] = df['debt_to_income_ratio'].fillna(0).clip(0, 1)
        logger.info("Created: debt_to_income_ratio")
    
    # ========================================================================
    # Expense Ratio
    # ========================================================================
    if 'total_monthly_expenses' in df.columns and 'monthly_salary' in df.columns:
        df['expense_ratio'] = df['total_monthly_expenses'] / df['monthly_salary'].replace(0, np.nan)
        df['expense_ratio'] = df['expense_ratio'].fillna(0).clip(0, 2)
        logger.info("Created: expense_ratio")
    
    # ========================================================================
    # Savings Rate
    # ========================================================================
    if all(col in df.columns for col in ['monthly_salary', 'total_monthly_expenses', 'current_emi_amount']):
        monthly_savings = (
            df['monthly_salary'] 
            - df['total_monthly_expenses'] 
            - df['current_emi_amount']
        )
        df['savings_rate'] = monthly_savings / df['monthly_salary'].replace(0, np.nan)
        df['savings_rate'] = df['savings_rate'].fillna(0).clip(-1, 1)
        logger.info("Created: savings_rate")
        
        # Monthly disposable income
        df['monthly_disposable_income'] = monthly_savings.clip(lower=0)
        logger.info("Created: monthly_disposable_income")
    
    # ========================================================================
    # Financial Stress Index
    # ========================================================================
    if all(col in df.columns for col in ['monthly_salary', 'bank_balance', 'total_monthly_expenses', 'current_emi_amount']):
        monthly_liquidity = df['monthly_salary'] + (df['bank_balance'] / 12)
        total_obligations = df['total_monthly_expenses'] + df['current_emi_amount']
        df['financial_stress_index'] = total_obligations / monthly_liquidity.replace(0, np.nan)
        df['financial_stress_index'] = df['financial_stress_index'].fillna(1).clip(0, 2)
        logger.info("Created: financial_stress_index")
    
    # ========================================================================
    # Credit Utilization Score
    # ========================================================================
    if 'current_emi_amount' in df.columns and 'monthly_salary' in df.columns:
        # Assuming max healthy EMI is 50% of salary
        expected_max_emi = df['monthly_salary'] * 0.5
        df['credit_utilization'] = df['current_emi_amount'] / expected_max_emi.replace(0, np.nan)
        df['credit_utilization'] = df['credit_utilization'].fillna(0).clip(0, 2)
        logger.info("Created: credit_utilization")
    
    # ========================================================================
    # Emergency Fund Ratio
    # ========================================================================
    if 'emergency_fund' in df.columns and 'monthly_salary' in df.columns:
        # Ideal emergency fund is 6 months of salary
        ideal_emergency_fund = df['monthly_salary'] * 6
        df['emergency_fund_ratio'] = df['emergency_fund'] / ideal_emergency_fund.replace(0, np.nan)
        df['emergency_fund_ratio'] = df['emergency_fund_ratio'].fillna(0).clip(0, 2)
        logger.info("Created: emergency_fund_ratio")
    
    # ========================================================================
    # Bank Balance to Salary Ratio
    # ========================================================================
    if 'bank_balance' in df.columns and 'monthly_salary' in df.columns:
        df['bank_balance_to_salary'] = df['bank_balance'] / df['monthly_salary'].replace(0, np.nan)
        df['bank_balance_to_salary'] = df['bank_balance_to_salary'].fillna(0).clip(0, 50)
        logger.info("Created: bank_balance_to_salary")
    
    # ========================================================================
    # Dependents per Income (normalized)
    # ========================================================================
    if 'dependents' in df.columns and 'monthly_salary' in df.columns:
        # Normalize by salary in units of 10,000
        df['dependents_per_10k_income'] = df['dependents'] / (df['monthly_salary'] / 10000).replace(0, np.nan)
        df['dependents_per_10k_income'] = df['dependents_per_10k_income'].fillna(0).clip(0, 10)
        logger.info("Created: dependents_per_10k_income")
    
    # ========================================================================
    # Family Size to Income Ratio
    # ========================================================================
    if 'family_size' in df.columns and 'monthly_salary' in df.columns:
        df['family_size_to_income'] = df['family_size'] / (df['monthly_salary'] / 10000).replace(0, np.nan)
        df['family_size_to_income'] = df['family_size_to_income'].fillna(0).clip(0, 10)
        logger.info("Created: family_size_to_income")
    
    # ========================================================================
    # Age and Employment Interaction
    # ========================================================================
    if 'age' in df.columns and 'years_of_employment' in df.columns:
        df['age_employment_stability'] = df['age'] * df['years_of_employment']
        logger.info("Created: age_employment_stability")
    
    # ========================================================================
    # Salary and Credit Score Interaction
    # ========================================================================
    if 'monthly_salary' in df.columns and 'credit_score' in df.columns:
        # Normalize salary and credit score before multiplying
        df['salary_credit_interaction'] = (
            (df['monthly_salary'] / 100000) * (df['credit_score'] / 100)
        )
        logger.info("Created: salary_credit_interaction")
    
    # ========================================================================
    # Boolean Indicators
    # ========================================================================
    
    # Has rent payment
    if 'monthly_rent' in df.columns:
        df['has_rent_payment'] = (df['monthly_rent'] > 0).astype(int)
        logger.info("Created: has_rent_payment")
    
    # Has education expenses
    if 'school_fees' in df.columns and 'college_fees' in df.columns:
        df['has_education_expenses'] = (
            (df['school_fees'] + df['college_fees']) > 0
        ).astype(int)
        logger.info("Created: has_education_expenses")
    
    # High credit score (700+)
    if 'credit_score' in df.columns:
        df['high_credit_score'] = (df['credit_score'] >= 700).astype(int)
        logger.info("Created: high_credit_score")
    
    # Has existing loans
    if 'existing_loans' in df.columns:
        # If it's already encoded as 0/1, keep it; if it's Yes/No, encode it
        if df['existing_loans'].dtype == 'object':
            df['has_existing_loans'] = (df['existing_loans'] == 'Yes').astype(int)
            logger.info("Created: has_existing_loans")
    
    # Healthy DTI (< 40%)
    if 'debt_to_income_ratio' in df.columns:
        df['healthy_dti'] = (df['debt_to_income_ratio'] < 0.4).astype(int)
        logger.info("Created: healthy_dti")
    
    # Positive savings
    if 'savings_rate' in df.columns:
        df['has_positive_savings'] = (df['savings_rate'] > 0).astype(int)
        logger.info("Created: has_positive_savings")
    
    # ========================================================================
    # Loan Request Features (if present)
    # ========================================================================
    
    # Monthly EMI for requested loan (simple calculation)
    if 'requested_amount' in df.columns and 'requested_tenure' in df.columns:
        # Simple EMI calculation (without interest for feature engineering)
        df['requested_monthly_emi'] = df['requested_amount'] / df['requested_tenure'].replace(0, np.nan)
        df['requested_monthly_emi'] = df['requested_monthly_emi'].fillna(0)
        logger.info("Created: requested_monthly_emi")
        
        # Requested EMI to salary ratio
        if 'monthly_salary' in df.columns:
            df['requested_emi_to_salary'] = (
                df['requested_monthly_emi'] / df['monthly_salary'].replace(0, np.nan)
            )
            df['requested_emi_to_salary'] = df['requested_emi_to_salary'].fillna(0).clip(0, 2)
            logger.info("Created: requested_emi_to_salary")
    
    # Requested amount to annual salary ratio
    if 'requested_amount' in df.columns and 'monthly_salary' in df.columns:
        annual_salary = df['monthly_salary'] * 12
        df['loan_to_annual_salary'] = df['requested_amount'] / annual_salary.replace(0, np.nan)
        df['loan_to_annual_salary'] = df['loan_to_annual_salary'].fillna(0).clip(0, 10)
        logger.info("Created: loan_to_annual_salary")
    
    # Requested amount to bank balance ratio
    if 'requested_amount' in df.columns and 'bank_balance' in df.columns:
        df['loan_to_bank_balance'] = df['requested_amount'] / df['bank_balance'].replace(0, np.nan)
        df['loan_to_bank_balance'] = df['loan_to_bank_balance'].fillna(10).clip(0, 100)
        logger.info("Created: loan_to_bank_balance")
    
    logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
    
    return df


def create_polynomial_features(
    df: pd.DataFrame,
    features: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Args:
        df: Input DataFrame
        features: List of feature names to create polynomials for
        degree: Polynomial degree (default 2 for squared terms)
        
    Returns:
        DataFrame with polynomial features added
        
    Example:
        >>> df = create_polynomial_features(df, ['age', 'salary'], degree=2)
    """
    logger.info(f"Creating polynomial features (degree={degree})")
    df = df.copy()
    
    for feature in features:
        if feature in df.columns:
            for d in range(2, degree + 1):
                new_feature_name = f"{feature}_power_{d}"
                df[new_feature_name] = df[feature] ** d
                logger.info(f"Created: {new_feature_name}")
    
    return df


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Create interaction features from feature pairs.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples containing feature name pairs
        
    Returns:
        DataFrame with interaction features added
        
    Example:
        >>> pairs = [('age', 'salary'), ('credit_score', 'employment_years')]
        >>> df = create_interaction_features(df, pairs)
    """
    logger.info("Creating interaction features")
    df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
            logger.info(f"Created: {interaction_name}")
    
    return df


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    k: int = 50,
    method: str = "mutual_info"
) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Select top k features based on statistical tests.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        task: Task type ('classification' or 'regression')
        k: Number of features to select
        method: Selection method ('f_test', 'mutual_info')
        
    Returns:
        Tuple of (selected features DataFrame, feature names, scores)
        
    Example:
        >>> X_selected, selected_features, scores = select_features(
        ...     X, y, task='classification', k=30
        ... )
    """
    logger.info(f"Selecting top {k} features using {method} method")
    
    # Ensure k doesn't exceed number of features
    k = min(k, X.shape[1])
    
    # Choose scoring function
    if task == "classification":
        if method == "mutual_info":
            score_func = mutual_info_classif
        else:
            score_func = f_classif
    else:  # regression
        score_func = f_regression
    
    # Select features
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Get feature scores
    scores = selector.scores_
    
    # Create DataFrame with selected features
    X_selected_df = pd.DataFrame(
        X_selected,
        columns=selected_features,
        index=X.index
    )
    
    logger.info(f"Selected {len(selected_features)} features")
    
    # Log top 10 features by score
    top_10_indices = np.argsort(scores)[-10:][::-1]
    top_10_features = [(X.columns[i], scores[i]) for i in top_10_indices]
    logger.info("Top 10 features by score:")
    for feat, score in top_10_features:
        logger.info(f"  {feat}: {score:.4f}")
    
    return X_selected_df, selected_features, scores


def get_feature_importance(
    model: any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importances sorted by importance
        
    Example:
        >>> importance_df = get_feature_importance(model, X_train.columns)
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Top {top_n} important features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df.head(top_n)


def calculate_correlation_with_target(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Calculate correlation of features with target variable.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        top_n: Number of top correlated features to return
        
    Returns:
        DataFrame with correlations sorted by absolute correlation
        
    Example:
        >>> corr_df = calculate_correlation_with_target(X_train, y_train)
    """
    logger.info("Calculating feature correlations with target")
    
    # Combine features and target
    df_combined = X.copy()
    df_combined['target'] = y
    
    # Calculate correlations
    correlations = df_combined.corr()['target'].drop('target')
    
    # Sort by absolute correlation
    corr_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values,
        'abs_correlation': np.abs(correlations.values)
    }).sort_values('abs_correlation', ascending=False)
    
    logger.info(f"Top {top_n} correlated features:")
    for idx, row in corr_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['correlation']:.4f}")
    
    return corr_df.head(top_n)


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features.
    
    Args:
        df: Input DataFrame
        threshold: Correlation threshold (default 0.95)
        
    Returns:
        Tuple of (DataFrame with removed features, list of removed features)
        
    Example:
        >>> df_reduced, removed = remove_correlated_features(df, threshold=0.95)
    """
    logger.info(f"Removing features with correlation > {threshold}")
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than threshold
    to_drop = [
        column for column in upper.columns 
        if any(upper[column] > threshold)
    ]
    
    # Drop features
    df_reduced = df.drop(columns=to_drop)
    
    logger.info(f"Removed {len(to_drop)} highly correlated features")
    if to_drop:
        logger.info(f"Removed features: {to_drop}")
    
    return df_reduced, to_drop


def create_features(
    df: pd.DataFrame,
    include_financial: bool = True,
    include_interactions: bool = False,
    remove_high_correlation: bool = False,
    correlation_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    This is the main function to use for feature engineering.
    
    Args:
        df: Input DataFrame
        include_financial: Whether to create financial features
        include_interactions: Whether to create interaction features
        remove_high_correlation: Whether to remove highly correlated features
        correlation_threshold: Correlation threshold for removal
        
    Returns:
        DataFrame with engineered features
        
    Example:
        >>> df_engineered = create_features(df)
    """
    logger.info("=" * 80)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 80)
    
    df = df.copy()
    original_features = len(df.columns)
    
    # Create financial features
    if include_financial and Config.CREATE_DERIVED_FEATURES:
        df = create_financial_features(df)
    
    # Create interaction features (optional)
    if include_interactions:
        # Define important interactions
        interactions = [
            ('age', 'years_of_employment'),
            ('monthly_salary', 'credit_score'),
            ('debt_to_income_ratio', 'credit_score'),
        ]
        df = create_interaction_features(df, interactions)
    
    # Remove highly correlated features (optional)
    if remove_high_correlation:
        # Separate target columns if present
        target_cols = []
        for col in [Config.CLASSIFICATION_TARGET, Config.REGRESSION_TARGET]:
            if col in df.columns:
                target_cols.append(col)
        
        if target_cols:
            targets = df[target_cols]
            features = df.drop(columns=target_cols)
            features, removed = remove_correlated_features(
                features, correlation_threshold
            )
            df = pd.concat([features, targets], axis=1)
        else:
            df, removed = remove_correlated_features(df, correlation_threshold)
    
    new_features = len(df.columns)
    logger.info("=" * 80)
    logger.info(f"Feature engineering completed")
    logger.info(f"Original features: {original_features}")
    logger.info(f"New features: {new_features}")
    logger.info(f"Added: {new_features - original_features} features")
    logger.info("=" * 80)
    
    return df

