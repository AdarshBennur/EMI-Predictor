"""
EMI Eligibility Prediction Page - EMI-Predict AI

Predict whether an applicant is eligible for an EMI.
BULLETPROOF VERSION: Comprehensive error handling and debug logging
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add project root to Python path to ensure emipredict module is importable
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="EMI Eligibility - EMI-Predict AI",
    page_icon="🎯",
    layout="wide"
)

# Debug mode disabled for production
DEBUG_MODE = False


def log_debug(message: str, data=None):
    """
    Log debug information (disabled in production).
    
    Args:
        message: Debug message to display
        data: Optional data to display (dict, shape, type, etc.)
    """
    # Debug logging disabled - no sidebar clutter for users
    pass


@st.cache_resource
def load_preprocessing_pipeline():
    """
    Load the saved preprocessing pipeline used during training.
    
    Returns:
        Tuple of (preprocessing_pipeline, feature_selector, selected_columns, error_message)
    """
    import joblib
    from emipredict.config.settings import Config
    from emipredict.data.pipeline import EMIPreprocessingPipeline
    
    try:
        log_debug("Loading preprocessing pipeline...")
        
        pipeline_path = Config.MODELS_DIR / "preprocessing_pipeline.pkl"
        if not pipeline_path.exists():
            error_msg = (
                "Preprocessing pipeline not found. This is needed to match training features.\n\n"
                "**To fix:**\n"
                "1. Run: `python scripts/train_models.py`\n"
                "2. This will create preprocessing_pipeline.pkl in models/ directory\n"
                "3. Reload this page"
            )
            log_debug(f"❌ Pipeline not found at: {pipeline_path}")
            return None, None, None, error_msg
        
        preprocessing_pipeline = EMIPreprocessingPipeline.load(pipeline_path)
        log_debug(f"✅ Pipeline loaded from: {pipeline_path}")
        
        # Also load feature selector
        feature_selector_path = Config.MODELS_DIR / "feature_selector.pkl"
        selected_columns_path = Config.MODELS_DIR / "selected_features.pkl"
        
        feature_selector = None
        selected_columns = None
        
        if feature_selector_path.exists():
            feature_selector = joblib.load(feature_selector_path)
            log_debug(f"✅ Feature selector loaded: {type(feature_selector).__name__}")
            
        if selected_columns_path.exists():
            selected_columns = joblib.load(selected_columns_path)
            log_debug(f"✅ Selected columns loaded: {len(selected_columns) if selected_columns else 0} features")
        
        return preprocessing_pipeline, feature_selector, selected_columns, None
        
    except Exception as e:
        error_msg = f"Error loading preprocessing pipeline: {str(e)}"
        log_debug(f"❌ Exception: {error_msg}")
        import traceback
        log_debug("Traceback:", traceback.format_exc())
        return None, None, None, error_msg


@st.cache_resource
def load_classification_model():
    """Load the trained classification model with comprehensive error handling."""
    import joblib
    from emipredict.config.settings import Config
    
    try:
        log_debug("Loading classification model...")
        
        # Try loading best model first
        model_path = Config.get_model_save_path("xgboost", "classification", "_best")
        if not model_path.exists():
            model_path = Config.get_model_save_path("xgboost", "classification")
        
        if not model_path.exists():
            # Check if any classification models exist
            models_dir = Config.MODELS_DIR
            available_models = list(models_dir.glob("*_classification*.pkl"))
            
            if available_models:
                # Use the first available classification model
                model_path = available_models[0]
                log_debug(f"Using available model: {model_path.name}")
                st.info(f"Using available model: {model_path.name}")
            else:
                error_msg = (
                    f"Model not found at {model_path}\n\n"
                    "**To fix this:**\n"
                    "1. Open a terminal\n"
                    "2. Run: `python scripts/train_models.py`\n"
                    "3. Wait for training to complete (~10 minutes)\n"
                    "4. Reload this page"
                )
                log_debug(f"❌ No classification models found in {models_dir}")
                return None, None, error_msg
        
        loaded_object = joblib.load(model_path)
        log_debug(f"✅ Model file loaded: {model_path.name}")
        log_debug(f"Loaded object type: {type(loaded_object)}")
        
        # Check if loaded object is a dict (some training scripts save model in a dict)
        if isinstance(loaded_object, dict):
            log_debug(f"Model is dict with keys: {list(loaded_object.keys())}")
            # Try to extract the model from common dict keys
            if 'model' in loaded_object:
                model = loaded_object['model']
            elif 'classifier' in loaded_object:
                model = loaded_object['classifier']
            else:
                # If dict doesn't have expected keys, try to find first sklearn/xgboost model
                for key, value in loaded_object.items():
                    if hasattr(value, 'predict') and hasattr(value, 'predict_proba'):
                        model = value
                        log_debug(f"Found model in dict key: {key}")
                        break
                else:
                    error_msg = f"Loaded dict from {model_path.name} but couldn't find model object inside it. Keys: {list(loaded_object.keys())}"
                    log_debug(f"❌ {error_msg}")
                    return None, None, error_msg
        else:
            model = loaded_object
        
        log_debug(f"Final model type: {type(model).__name__}")
        
        # Verify model has required methods
        if not hasattr(model, 'predict'):
            error_msg = f"Model missing 'predict' method. Type: {type(model)}"
            log_debug(f"❌ {error_msg}")
            return None, None, error_msg
            
        if not hasattr(model, 'predict_proba'):
            error_msg = f"Model missing 'predict_proba' method. Type: {type(model)}"
            log_debug(f"❌ {error_msg}")
            return None, None, error_msg
        
        # Check if model has n_features_in_ attribute
        if hasattr(model, 'n_features_in_'):
            log_debug(f"Model expects {model.n_features_in_} features")
        
        metadata = {
            "model_type": "XGBoost Classifier (3-class)",
            "path": str(model_path.name),
            "n_features": getattr(model, 'n_features_in_', 'Unknown')
        }
        log_debug(f"✅ Model loaded successfully: {metadata}")
        return model, metadata, None
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        log_debug(f"❌ Exception: {error_msg}")
        import traceback
        log_debug("Traceback:", traceback.format_exc())
        return None, None, error_msg


def create_input_form():
    """Create input form for user data with validation."""
    st.subheader("📝 Enter Applicant Details")
    
    with st.form("eligibility_form"):
        # Demographics
        st.markdown("### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced", "Widowed"]
            )
            education = st.selectbox(
                "Education Level",
                ["High School", "Graduate", "Professional"]
            )
        
        with col3:
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3, step=1)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
        
        # Employment
        st.markdown("### 💼 Employment Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monthly_salary = st.number_input(
                "Monthly Salary (INR)",
                min_value=10000,
                max_value=500000,
                value=50000,
                step=1000
            )
            employment_type = st.selectbox(
                "Employment Type",
                ["Private", "Government", "Self-employed"]
            )
        
        with col2:
            years_of_employment = st.number_input(
                "Years of Employment",
                min_value=0.0,
                max_value=40.0,
                value=5.0,
                step=0.5
            )
            company_type = st.selectbox(
                "Company Type",
                ["Startup", "Mid-size", "MNC", "PSU"]
            )
        
        with col3:
            pass
        
        # Housing
        st.markdown("### 🏠 Housing Details")
        col1, col2 = st.columns(2)
        
        with col1:
            house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])
            monthly_rent = st.number_input(
                "Monthly Rent (INR)",
                min_value=0,
                max_value=100000,
                value=0 if house_type != "Rented" else 15000,
                step=1000,
                disabled=(house_type != "Rented")
            )
        
        with col2:
            pass
        
        # Expenses
        st.markdown("### 💸 Monthly Expenses")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            groceries_utilities = st.number_input(
                "Groceries & Utilities (INR)",
                min_value=0,
                max_value=50000,
                value=10000,
                step=500
            )
            travel_expenses = st.number_input(
                "Travel Expenses (INR)",
                min_value=0,
                max_value=30000,
                value=5000,
                step=500
            )
        
        with col2:
            school_fees = st.number_input(
                "School Fees (INR)",
                min_value=0,
                max_value=50000,
                value=0,
                step=500
            )
            college_fees = st.number_input(
                "College Fees (INR)",
                min_value=0,
                max_value=100000,
                value=0,
                step=1000
            )
        
        with col3:
            other_monthly_expenses = st.number_input(
                "Other Monthly Expenses (INR)",
                min_value=0,
                max_value=50000,
                value=5000,
                step=500
            )
        
        # Financial Status
        st.markdown("### 💰 Financial Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
            current_emi_amount = st.number_input(
                "Current EMI Amount (INR)",
                min_value=0,
                max_value=100000,
                value=0 if existing_loans == "No" else 10000,
                step=1000,
                disabled=(existing_loans == "No")
            )
        
        with col2:
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=700,
                step=10
            )
            bank_balance = st.number_input(
                "Bank Balance (INR)",
                min_value=0,
                max_value=10000000,
                value=100000,
                step=10000
            )
        
        with col3:
            emergency_fund = st.number_input(
                "Emergency Fund (INR)",
                min_value=0,
                max_value=5000000,
                value=50000,
                step=5000
            )
        
        # Loan Request
        st.markdown("### 📄 Loan Request Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            emi_scenario = st.selectbox(
                "EMI Scenario",
                [
                    "Personal Loan EMI",
                    "Vehicle EMI",
                    "Education EMI",
                    "Home Loan EMI",
                    "E-commerce Shopping EMI"
                ]
            )
        
        with col2:
            requested_amount = st.number_input(
                "Requested Loan Amount (INR)",
                min_value=10000,
                max_value=10000000,
                value=500000,
                step=10000
            )
        
        with col3:
            requested_tenure = st.number_input(
                "Requested Tenure (months)",
                min_value=6,
                max_value=360,
                value=60,
                step=6
            )
        
        # Submit button
        submitted = st.form_submit_button("🎯 Predict Eligibility", use_container_width=True)
    
    if submitted:
        log_debug("Form submitted, creating input DataFrame...")
        
        # Create input dataframe
        input_data = {
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education': education,
            'monthly_salary': monthly_salary,
            'employment_type': employment_type,
            'years_of_employment': years_of_employment,
            'company_type': company_type,
            'house_type': house_type,
            'monthly_rent': monthly_rent,
            'family_size': family_size,
            'dependents': dependents,
            'school_fees': school_fees,
            'college_fees': college_fees,
            'travel_expenses': travel_expenses,
            'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses,
            'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount,
            'credit_score': credit_score,
            'bank_balance': bank_balance,
            'emergency_fund': emergency_fund,
            'emi_scenario': emi_scenario,
            'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }
        
        input_df = pd.DataFrame([input_data])
        log_debug(f"Input DataFrame created: shape={input_df.shape}")
        log_debug(f"Input columns: {list(input_df.columns)}")
        return input_df
    
    return None


def preprocess_input_with_pipeline(input_df, preprocessing_pipeline):
    """
    Preprocess input data using the saved preprocessing pipeline.
    
    Args:
        input_df: Raw input DataFrame
        preprocessing_pipeline: Fitted preprocessing pipeline
        
    Returns:
        Preprocessed DataFrame with engineered features
        
    Raises:
        Exception: If preprocessing fails
    """
    try:
        log_debug("Starting preprocessing...")
        log_debug(f"Input shape: {input_df.shape}")
        log_debug(f"Input dtypes: {dict(input_df.dtypes)}")
        
        # Add dummy target columns (required by pipeline but will be ignored)
        input_df['emi_eligibility'] = 'Eligible'
        input_df['max_monthly_emi'] = 10000
        
        log_debug("Added dummy target columns")
        
        # Apply preprocessing pipeline (handles encoding, feature engineering)
        start_time = time.time()
        processed_df = preprocessing_pipeline.transform(input_df)
        elapsed = time.time() - start_time
        
        log_debug(f"Preprocessing completed in {elapsed:.3f}s")
        log_debug(f"Output shape: {processed_df.shape}")
        log_debug(f"Output columns (first 10): {list(processed_df.columns[:10])}")
        log_debug(f"Output dtypes sample: {dict(list(processed_df.dtypes.items())[:5])}")
        
        # Check for NaN or infinite values
        if processed_df.isnull().any().any():
            nan_cols = processed_df.columns[processed_df.isnull().any()].tolist()
            log_debug(f"⚠️ WARNING: NaN values found in columns: {nan_cols}")
        
        if np.isinf(processed_df.select_dtypes(include=[np.number]).values).any():
            log_debug("⚠️ WARNING: Infinite values found in processed data")
        
        return processed_df
        
    except Exception as e:
        log_debug(f"❌ Preprocessing failed: {str(e)}")
        import traceback
        log_debug("Traceback:", traceback.format_exc())
        raise


def apply_feature_selection(processed_df, feature_selector, selected_columns):
    """
    Apply feature selection to preprocessed data.
    
    Args:
        processed_df: Preprocessed DataFrame
        feature_selector: Fitted feature selector
        selected_columns: List of selected column names
        
    Returns:
        DataFrame with selected features only
        
    Raises:
        Exception: If feature selection fails
    """
    try:
        log_debug("Starting feature selection...")
        log_debug(f"Input shape: {processed_df.shape}")
        log_debug(f"Feature selector type: {type(feature_selector).__name__}")
        
        if hasattr(feature_selector, 'n_features_in_'):
            log_debug(f"Selector expects {feature_selector.n_features_in_} features")
        
        start_time = time.time()
        processed_features = feature_selector.transform(processed_df)
        elapsed = time.time() - start_time
        
        log_debug(f"Feature selection completed in {elapsed:.3f}s")
        log_debug(f"Output shape: {processed_features.shape}")
        
        # Convert back to DataFrame with selected column names
        if selected_columns is not None:
            log_debug(f"Using {len(selected_columns)} selected column names")
            processed_df = pd.DataFrame(
                processed_features,
                columns=selected_columns,
                index=processed_df.index
            )
        else:
            log_debug("No column names available, using numeric indices")
            processed_df = pd.DataFrame(processed_features, index=processed_df.index)
        
        log_debug(f"Final DataFrame shape: {processed_df.shape}")
        log_debug(f"Final columns (first 10): {list(processed_df.columns[:10])}")
        
        return processed_df
        
    except Exception as e:
        log_debug(f"❌ Feature selection failed: {str(e)}")
        import traceback
        log_debug("Traceback:", traceback.format_exc())
        raise


def make_prediction(model, processed_df):
    """
    Make prediction using the trained model.
    
    Args:
        model: Trained classification model
        processed_df: Preprocessed and feature-selected DataFrame
        
    Returns:
        Tuple of (prediction, probability)
        
    Raises:
        Exception: If prediction fails
    """
    try:
        log_debug("Starting prediction...")
        log_debug(f"Input shape: {processed_df.shape}")
        log_debug(f"Model type: {type(model).__name__}")
        
        # Verify model has required methods
        if not hasattr(model, 'predict'):
            raise ValueError(f"Model {type(model)} does not have 'predict' method")
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {type(model)} does not have 'predict_proba' method")
        
        # Check feature count match
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            actual_features = processed_df.shape[1]
            if expected_features != actual_features:
                log_debug(f"⚠️ WARNING: Feature count mismatch!")
                log_debug(f"Expected: {expected_features}, Got: {actual_features}")
        
        # Make prediction
        start_time = time.time()
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
        elapsed = time.time() - start_time
        
        log_debug(f"Prediction completed in {elapsed:.3f}s")
        log_debug(f"Prediction type: {type(prediction)}, dtype: {prediction.dtype if hasattr(prediction, 'dtype') else 'N/A'}")
        log_debug(f"Prediction value: {prediction}")
        log_debug(f"Probability type: {type(probability)}, dtype: {probability.dtype if hasattr(probability, 'dtype') else 'N/A'}")
        log_debug(f"Probability shape: {probability.shape}")
        log_debug(f"Probability values: {probability}")
        
        return prediction, probability
        
    except Exception as e:
        log_debug(f"❌ Prediction failed: {str(e)}")
        import traceback
        log_debug("Traceback:", traceback.format_exc())
        raise


def display_prediction_result(prediction, probability, input_df):
    """
    Display 3-class prediction result with comprehensive error handling.
    
    Classes:
    - 0: Eligible (Low risk)
    - 1: High_Risk (Marginal case)
    - 2: Not_Eligible (High risk)
    """
    try:
        log_debug("Displaying prediction result...")
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        # Get prediction class - convert to native Python int
        pred_class = int(prediction[0])
        log_debug(f"Prediction class: {pred_class} (type: {type(pred_class)})")
        
        # Class names
        class_names = ['Eligible', 'High_Risk', 'Not_Eligible']
        predicted_label = class_names[pred_class]
        log_debug(f"Predicted label: {predicted_label}")
        
        # Get probabilities for all 3 classes - convert to native Python float
        # Streamlit requires Python float, not numpy.float32
        prob_eligible = float(probability[0][0])
        prob_high_risk = float(probability[0][1])
        prob_not_eligible = float(probability[0][2])
        
        log_debug(f"Probabilities: Eligible={prob_eligible:.4f}, High_Risk={prob_high_risk:.4f}, Not_Eligible={prob_not_eligible:.4f}")
        
        # Verify probabilities sum to ~1.0
        prob_sum = prob_eligible + prob_high_risk + prob_not_eligible
        if abs(prob_sum - 1.0) > 0.01:
            log_debug(f"⚠️ WARNING: Probabilities sum to {prob_sum:.4f}, expected ~1.0")
        
        # Display main result with color coding
        st.markdown("### 📊 Classification Result")
        
        if pred_class == 0:  # Eligible
            st.success(f"✅ **ELIGIBLE** for EMI")
            st.markdown("**Status:** Low risk, comfortable EMI affordability")
            main_confidence = prob_eligible
        elif pred_class == 1:  # High_Risk
            st.warning(f"⚠️ **HIGH RISK** - Marginal Case")
            st.markdown("**Status:** Marginal case, may require higher interest rates or additional documentation")
            main_confidence = prob_high_risk
        else:  # Not_Eligible (class 2)
            st.error(f"❌ **NOT ELIGIBLE** for EMI")
            st.markdown("**Status:** High risk, loan not recommended")
            main_confidence = prob_not_eligible
        
        st.metric("Primary Confidence Score", f"{main_confidence*100:.1f}%")
        
        # Display probability distribution for all classes
        st.markdown("---")
        st.subheader("📈 Probability Distribution (All Classes)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "✅ Eligible",
                f"{prob_eligible*100:.1f}%",
                help="Low risk, comfortable affordability"
            )
            # Streamlit st.progress() requires native Python float (0.0-1.0), not numpy.float32
            st.progress(prob_eligible)
        
        with col2:
            st.metric(
                "⚠️ High Risk",
                f"{prob_high_risk*100:.1f}%",
                help="Marginal case, higher interest may apply"
            )
            # Streamlit st.progress() requires native Python float (0.0-1.0), not numpy.float32
            st.progress(prob_high_risk)
        
        with col3:
            st.metric(
                "❌ Not Eligible",
                f"{prob_not_eligible*100:.1f}%",
                help="High risk, loan not recommended"
            )
            # Streamlit st.progress() requires native Python float (0.0-1.0), not numpy.float32
            st.progress(prob_not_eligible)
        
        # Financial summary
        st.markdown("---")
        st.subheader("📊 Financial Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Ensure values are native Python types for display
        monthly_salary = float(input_df.iloc[0].get('monthly_salary', 0))
        current_emi = float(input_df.iloc[0].get('current_emi_amount', 0))
        
        # Calculate total expenses
        expense_cols = [
            'monthly_rent', 'groceries_utilities', 'travel_expenses',
            'school_fees', 'college_fees', 'other_monthly_expenses'
        ]
        total_expenses = sum(float(input_df.iloc[0].get(col, 0)) for col in expense_cols)
        
        # DTI ratio
        dti_ratio = (current_emi / monthly_salary * 100) if monthly_salary > 0 else 0
        
        with col1:
            st.metric("Monthly Salary", f"₹{monthly_salary:,.0f}")
        
        with col2:
            st.metric("Total Expenses", f"₹{total_expenses:,.0f}")
        
        with col3:
            st.metric("Current EMI", f"₹{current_emi:,.0f}")
        
        with col4:
            st.metric("DTI Ratio", f"{dti_ratio:.1f}%")
        
        # Recommendations based on predicted class
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if pred_class == 0:  # Eligible
            st.success("""
            **Congratulations! You are ELIGIBLE for an EMI. Here are some tips:**
            
            ✅ **Your Application:**
            - You have a strong financial profile
            - Low risk assessment means competitive interest rates
            - Your application is likely to be approved quickly
            
            📝 **Best Practices:**
            - Maintain a healthy debt-to-income ratio (below 40%)
            - Keep your credit score above 700
            - Ensure timely payment of all EMIs
            - Maintain an emergency fund of 6 months' salary
            - Consider prepayment options to reduce total interest
            """)
        
        elif pred_class == 1:  # High_Risk
            st.warning("""
            **HIGH RISK - Marginal Case. Your application requires careful review:**
            
            ⚠️ **Current Status:**
            - You are on the borderline of eligibility
            - May qualify with higher interest rates
            - Additional documentation may be required
            - Consider co-applicant or guarantor
            
            💪 **Immediate Actions to Improve:**
            - Reduce existing debt obligations
            - Increase your credit score (aim for 700+)
            - Build a larger emergency fund
            - Consider reducing the loan amount or extending tenure
            - Demonstrate stable income for 6+ months
            
            📊 **Alternative Options:**
            - Apply for a smaller loan amount
            - Extend the repayment tenure
            - Add a co-applicant with good credit
            - Provide additional collateral
            """)
        
        else:  # Not_Eligible (class 2)
            st.error("""
            **NOT ELIGIBLE - High Risk. Loan not recommended at this time.**
            
            ❌ **Current Assessment:**
            - Financial risk indicators are too high
            - Debt-to-income ratio may be unfavorable
            - Current obligations may be excessive
            - Insufficient financial capacity for additional EMI
            
            🎯 **Path to Eligibility (6-12 months plan):**
            
            **Priority Actions:**
            1. **Reduce Existing Debt:**
               - Pay off high-interest loans first
               - Clear credit card balances
               - Consolidate multiple small loans
            
            2. **Improve Credit Score:**
               - Check credit report for errors
               - Pay all bills on time
               - Keep credit utilization below 30%
               - Avoid new credit inquiries
            
            3. **Increase Income:**
               - Seek salary increment or promotion
               - Add secondary income sources
               - Include spouse's income (joint application)
            
            4. **Build Emergency Fund:**
               - Save at least 6 months of expenses
               - Set aside 10-15% of monthly income
            
            5. **Reduce Monthly Obligations:**
               - Cut discretionary expenses
               - Negotiate lower rent or consider relocation
               - Optimize insurance and subscription costs
            
            📞 **Next Steps:**
            - Revisit your application after 6-12 months
            - Consider financial counseling
            - Explore alternative financing options
            - Build a detailed improvement plan
            """)
        
        log_debug("✅ Display completed successfully")
        
    except Exception as e:
        log_debug(f"❌ Display failed: {str(e)}")
        import traceback
        log_debug("Traceback:", traceback.format_exc())
        st.error(f"Error displaying results: {str(e)}")
        st.exception(e)
        raise


def main():
    """Main function for Eligibility Prediction page with bulletproof error handling."""
    
    st.title("🎯 EMI Eligibility Prediction")
    st.markdown("Predict whether you are eligible for an EMI based on your financial profile.")
    st.markdown("---")
    
    # STEP 1: Load preprocessing pipeline
    log_debug("=== STEP 1: Loading preprocessing pipeline ===")
    try:
        preprocessing_pipeline, feature_selector, selected_columns, pipeline_error = load_preprocessing_pipeline()
        
        if pipeline_error:
            st.error("⚠️ Preprocessing Pipeline Error")
            st.error(pipeline_error)
            st.warning("Preprocessing pipeline is required to transform input features to match training.")
            log_debug("❌ Pipeline loading failed - stopping execution")
            return
        
        if preprocessing_pipeline is None:
            st.error("Preprocessing pipeline not available. Please train models first.")
            log_debug("❌ Pipeline is None - stopping execution")
            return
        
        st.sidebar.success("✅ Preprocessing pipeline loaded")
        log_debug("✅ Step 1 completed successfully")
        
    except Exception as e:
        st.error(f"Fatal error loading preprocessing pipeline: {str(e)}")
        st.exception(e)
        log_debug(f"❌ Fatal error in Step 1: {str(e)}")
        return
    
    # STEP 2: Load model
    log_debug("=== STEP 2: Loading classification model ===")
    try:
        model, metadata, model_error = load_classification_model()
        
        if model_error:
            st.error("⚠️ Model Loading Error")
            st.error(model_error)
            log_debug("❌ Model loading failed - stopping execution")
            return
        
        if model is None:
            st.error("Model not available. Please train the model first.")
            log_debug("❌ Model is None - stopping execution")
            return
        
        # Display model information
        if metadata:
            st.sidebar.markdown("### 🤖 Model Information")
            st.sidebar.success("✅ Model Ready")
            st.sidebar.info(f"**Type:** {metadata.get('model_type', 'Unknown')}")
            st.sidebar.info(f"**File:** {metadata.get('path', 'Unknown')}")
            if 'n_features' in metadata:
                st.sidebar.info(f"**Features:** {metadata.get('n_features')}")
        
        log_debug("✅ Step 2 completed successfully")
        
    except Exception as e:
        st.error(f"Fatal error loading model: {str(e)}")
        st.exception(e)
        log_debug(f"❌ Fatal error in Step 2: {str(e)}")
        return
    
    # STEP 3: Create input form
    log_debug("=== STEP 3: Creating input form ===")
    try:
        input_df = create_input_form()
        log_debug("✅ Step 3 completed")
    except Exception as e:
        st.error(f"Error creating input form: {str(e)}")
        st.exception(e)
        log_debug(f"❌ Fatal error in Step 3: {str(e)}")
        return
    
    # STEP 4: Make prediction (only if form was submitted)
    if input_df is not None:
        log_debug("=== STEP 4: Making prediction ===")
        log_debug(f"Input DataFrame shape: {input_df.shape}")
        
        # Create a progress container
        progress_container = st.empty()
        status_container = st.empty()
        
        try:
            # STEP 4.1: Preprocessing
            with st.spinner("🔄 Step 1/3: Applying preprocessing pipeline..."):
                log_debug("=== STEP 4.1: Preprocessing ===")
                status_container.info("🔄 Applying preprocessing pipeline...")
                
                try:
                    start_time = time.time()
                    processed_df = preprocess_input_with_pipeline(input_df.copy(), preprocessing_pipeline)
                    elapsed = time.time() - start_time
                    
                    status_container.success(f"✅ Preprocessing complete: {processed_df.shape[1]} features engineered (took {elapsed:.2f}s)")
                    log_debug(f"✅ Step 4.1 completed in {elapsed:.2f}s")
                    
                except Exception as prep_error:
                    status_container.error(f"❌ Preprocessing failed: {str(prep_error)}")
                    st.error("**Preprocessing Error Details:**")
                    st.error(str(prep_error))
                    st.exception(prep_error)
                    log_debug(f"❌ Preprocessing exception: {str(prep_error)}")
                    return
            
            # STEP 4.2: Feature Selection
            with st.spinner("🔄 Step 2/3: Applying feature selection..."):
                log_debug("=== STEP 4.2: Feature Selection ===")
                
                if feature_selector is not None:
                    status_container.info("🔄 Applying feature selection...")
                    
                    try:
                        start_time = time.time()
                        processed_df = apply_feature_selection(processed_df, feature_selector, selected_columns)
                        elapsed = time.time() - start_time
                        
                        status_container.success(f"✅ Feature selection applied: {processed_df.shape[1]} features selected (took {elapsed:.2f}s)")
                        log_debug(f"✅ Step 4.2 completed in {elapsed:.2f}s")
                        
                    except Exception as selector_error:
                        status_container.error(f"❌ Feature selection failed: {str(selector_error)}")
                        st.error("**Feature Selection Error Details:**")
                        st.error(str(selector_error))
                        st.error(f"Current features: {processed_df.shape[1]}")
                        
                        if hasattr(feature_selector, 'feature_names_in_'):
                            st.error(f"Expected features: {len(feature_selector.feature_names_in_)}")
                            with st.expander("Show feature comparison"):
                                st.write({
                                    'Current features (first 10)': processed_df.columns[:10].tolist(),
                                    'Expected features (first 10)': feature_selector.feature_names_in_[:10].tolist()
                                })
                        
                        st.exception(selector_error)
                        log_debug(f"❌ Feature selection exception: {str(selector_error)}")
                        return
                else:
                    status_container.warning("⚠️ Feature selector not available, using all features")
                    log_debug("⚠️ No feature selector - using all features")
            
            # STEP 4.3: Verify feature count
            log_debug("=== STEP 4.3: Verifying feature count ===")
            expected_n_features = len(selected_columns) if selected_columns else (model.n_features_in_ if hasattr(model, 'n_features_in_') else None)
            
            if expected_n_features and processed_df.shape[1] != expected_n_features:
                warning_msg = f"⚠️ Feature count mismatch! Expected: {expected_n_features}, Got: {processed_df.shape[1]}"
                st.warning(warning_msg)
                log_debug(warning_msg)
                st.info("Proceeding with prediction, but results may be unreliable.")
            else:
                log_debug(f"✅ Feature count matches: {processed_df.shape[1]}")
            
            # STEP 4.4: Make Prediction
            with st.spinner("🔄 Step 3/3: Making prediction..."):
                log_debug("=== STEP 4.4: Making Prediction ===")
                status_container.info("🔄 Making prediction with model...")
                
                try:
                    start_time = time.time()
                    prediction, probability = make_prediction(model, processed_df)
                    elapsed = time.time() - start_time
                    
                    status_container.success(f"✅ Prediction complete (took {elapsed:.2f}s)")
                    log_debug(f"✅ Step 4.4 completed in {elapsed:.2f}s")
                    
                except Exception as pred_error:
                    status_container.error(f"❌ Prediction failed: {str(pred_error)}")
                    st.error("**Prediction Error Details:**")
                    st.error(f"Model type: {type(model)}")
                    st.error(f"Has predict: {hasattr(model, 'predict')}")
                    st.error(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
                    st.error(f"Input shape: {processed_df.shape}")
                    if hasattr(model, 'n_features_in_'):
                        st.error(f"Expected features: {model.n_features_in_}")
                    st.error(str(pred_error))
                    st.exception(pred_error)
                    log_debug(f"❌ Prediction exception: {str(pred_error)}")
                    return
            
            # Clear status messages
            status_container.empty()
            
            # STEP 4.5: Display Results
            log_debug("=== STEP 4.5: Displaying Results ===")
            try:
                display_prediction_result(prediction, probability, input_df)
                log_debug("✅ Step 4.5 completed successfully")
                log_debug("=== ALL STEPS COMPLETED SUCCESSFULLY ===")
                
            except Exception as display_error:
                st.error(f"❌ Error displaying results: {str(display_error)}")
                st.exception(display_error)
                log_debug(f"❌ Display exception: {str(display_error)}")
                return
            
        except Exception as e:
            # Catch-all for any unexpected errors
            st.error("❌ **Unexpected Error During Prediction**")
            st.error(f"Error: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            st.exception(e)
            log_debug(f"❌ Unexpected exception in Step 4: {str(e)}")
            import traceback
            log_debug("Full traceback:", traceback.format_exc())
            return


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:
        st.error("❌ **FATAL ERROR**")
        st.error(f"The application encountered a fatal error: {str(fatal_error)}")
        st.exception(fatal_error)
        log_debug(f"❌ FATAL ERROR: {str(fatal_error)}")
        import traceback
        log_debug("Fatal traceback:", traceback.format_exc())
