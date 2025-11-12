"""
Data Explorer Page - EMI-Predict AI

Explore dataset statistics, visualizations, and feature distributions.
"""

import sys
from pathlib import Path

# Add project root to Python path to ensure emipredict module is importable
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Data Explorer - EMI-Predict AI",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(nrows=None):
    """Load and cache the dataset with configurable row limit."""
    from emipredict.config.settings import Config
    
    try:
        df = pd.read_csv(Config.DATA_PATH, nrows=nrows)
        
        # Get total row count efficiently
        with open(Config.DATA_PATH, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # -1 for header
        
        return df, total_rows, None
    except Exception as e:
        return None, 0, str(e)

def show_dataset_overview(df):
    """Display dataset overview statistics."""
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Total Features", len(df.columns))
    
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")
    
    # Data types
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types Distribution**")
        dtype_counts = df.dtypes.value_counts()
        st.bar_chart(dtype_counts)
    
    with col2:
        st.write("**Sample Data (First 10 Rows)**")
        st.dataframe(df.head(10), use_container_width=True)


def show_numerical_features(df):
    """Display numerical feature statistics and distributions."""
    st.subheader("📈 Numerical Features Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Feature selector
    selected_feature = st.selectbox(
        "Select a numerical feature to explore",
        numerical_cols,
        index=0
    )
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Statistics for {selected_feature}**")
            stats = df[selected_feature].describe()
            st.dataframe(stats.to_frame(), use_container_width=True)
        
        with col2:
            st.write(f"**Distribution of {selected_feature}**")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df[selected_feature].dropna(), bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {selected_feature}')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Box plot
        st.write(f"**Box Plot for {selected_feature}**")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(df[selected_feature].dropna(), vert=False)
        ax.set_xlabel(selected_feature)
        ax.set_title(f'Box Plot of {selected_feature}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()


def show_categorical_features(df):
    """Display categorical feature statistics."""
    st.subheader("📊 Categorical Features Analysis")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        st.info("No categorical features found in the dataset.")
        return
    
    selected_feature = st.selectbox(
        "Select a categorical feature to explore",
        categorical_cols,
        index=0
    )
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Value Counts for {selected_feature}**")
            value_counts = df[selected_feature].value_counts()
            st.dataframe(value_counts.to_frame(), use_container_width=True)
        
        with col2:
            st.write(f"**Distribution of {selected_feature}**")
            fig, ax = plt.subplots(figsize=(8, 5))
            value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {selected_feature}')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close()


def show_target_analysis(df):
    """Analyze target variables."""
    st.subheader("🎯 Target Variables Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**EMI Eligibility Distribution**")
        if 'emi_eligibility' in df.columns:
            eligibility_counts = df['emi_eligibility'].value_counts()
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                eligibility_counts.values,
                labels=eligibility_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=['#ff9999', '#66b3ff']
            )
            ax.set_title('EMI Eligibility Distribution')
            st.pyplot(fig)
            plt.close()
            
            st.dataframe(eligibility_counts.to_frame(), use_container_width=True)
        else:
            st.info("emi_eligibility column not found")
    
    with col2:
        st.write("**Max Monthly EMI Distribution**")
        if 'max_monthly_emi' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['max_monthly_emi'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
            ax.set_xlabel('Max Monthly EMI (INR)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Max Monthly EMI')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            stats = df['max_monthly_emi'].describe()
            st.dataframe(stats.to_frame(), use_container_width=True)
        else:
            st.info("max_monthly_emi column not found")


def show_correlation_analysis(df):
    """Display correlation heatmap."""
    st.subheader("🔥 Correlation Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        # Select features for correlation
        n_features = st.slider(
            "Number of features to include in correlation",
            min_value=5,
            max_value=min(20, len(numerical_cols)),
            value=min(10, len(numerical_cols))
        )
        
        selected_cols = numerical_cols[:n_features]
        
        # Calculate correlation
        corr_matrix = df[selected_cols].corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Not enough numerical features for correlation analysis")


def show_missing_values_analysis(df):
    """Analyze missing values in the dataset."""
    st.subheader("🔍 Missing Values Analysis")
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Feature': missing.index,
        'Missing Count': missing.values,
        'Missing Percentage': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', ascending=False
    )
    
    if len(missing_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values Summary**")
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            st.write("**Missing Values Visualization**")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(missing_df['Feature'], missing_df['Missing Percentage'], color='coral')
            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title('Missing Values by Feature')
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            plt.close()
    else:
        st.success("✅ No missing values found in the dataset!")


def main():
    """Main function for Data Explorer page."""
    
    st.title("📊 Data Explorer")
    st.markdown("Explore the EMI prediction dataset with interactive visualizations and statistics.")
    st.markdown("---")
    
    # Row count selector in sidebar
    st.sidebar.markdown("### 📊 Data Loading Options")
    row_options = {
        '10,000 rows (Fast)': 10000,
        '50,000 rows (Medium)': 50000,
        '100,000 rows (Slow)': 100000,
        'All rows (Very Slow)': None
    }
    
    selected_option = st.sidebar.selectbox(
        "Select number of rows to display:",
        options=list(row_options.keys()),
        index=0,
        help="Choose how many rows to load. More rows = slower performance but more complete view."
    )
    
    nrows = row_options[selected_option]
    
    # Load data
    with st.spinner("Loading dataset..."):
        df, total_rows, error = load_data(nrows=nrows)
    
    if error:
        st.error(f"Error loading dataset: {error}")
        st.info("Please ensure the dataset is available at the configured path.")
        return
    
    if df is None:
        st.error("Failed to load dataset.")
        return
    
    # Display dataset info banner
    loaded_rows = len(df)
    st.info(f"📊 Displaying **{loaded_rows:,}** of **{total_rows:,}** total records in dataset")
    
    if loaded_rows < total_rows:
        st.caption(f"💡 Tip: Use the sidebar to load more rows for complete analysis")
    
    st.markdown("---")
    
    # Sidebar options
    st.sidebar.markdown("---")
    st.sidebar.header("Explorer Options")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Dataset Overview",
            "Numerical Features",
            "Categorical Features",
            "Target Variables",
            "Correlation Analysis",
            "Missing Values"
        ]
    )
    
    # Display selected analysis
    if analysis_type == "Dataset Overview":
        show_dataset_overview(df)
    
    elif analysis_type == "Numerical Features":
        show_numerical_features(df)
    
    elif analysis_type == "Categorical Features":
        show_categorical_features(df)
    
    elif analysis_type == "Target Variables":
        show_target_analysis(df)
    
    elif analysis_type == "Correlation Analysis":
        show_correlation_analysis(df)
    
    elif analysis_type == "Missing Values":
        show_missing_values_analysis(df)
    
    # Download option
    st.markdown("---")
    st.subheader("📥 Download Data")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample Data as CSV",
        data=csv,
        file_name="emi_prediction_sample.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()

