"""
Admin Monitoring Page - EMI-Predict AI

Monitor model performance, view MLflow experiments, and track system metrics.
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
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Admin Monitoring - EMI-Predict AI",
    page_icon="📈",
    layout="wide"
)

@st.cache_data(ttl=60)
def get_model_files():
    """Get list of model files."""
    from emipredict.config.settings import Config
    
    models_dir = Config.MODELS_DIR
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.pkl"))
    return [(f.name, f.stat().st_size, f.stat().st_mtime) for f in model_files]


@st.cache_data(ttl=60)
def get_mlflow_experiments():
    """Get MLflow experiments data."""
    try:
        import mlflow
        from emipredict.config.settings import Config
        
        # Set tracking URI
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        
        # Search experiments (this will create tracking dir if not exists)
        experiments = mlflow.search_experiments()
        
        if not experiments or len(experiments) == 0:
            return None  # No experiments yet
        
        return experiments
    except Exception as e:
        st.error(f"MLflow error: {str(e)}")
        return None


@st.cache_data(ttl=60)
def get_mlflow_runs(experiment_name):
    """Get MLflow runs for an experiment."""
    try:
        import mlflow
        from emipredict.config.settings import Config
        
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=20,
            order_by=["start_time DESC"]
        )
        
        return runs
    except Exception as e:
        return None


def show_model_overview():
    """Show overview of trained models."""
    st.subheader("🤖 Trained Models")
    
    model_files = get_model_files()
    
    if not model_files:
        st.warning("No trained models found. Please train models first.")
        return
    
    # Convert to DataFrame
    model_df = pd.DataFrame(
        model_files,
        columns=['Model File', 'Size (bytes)', 'Last Modified']
    )
    
    # Convert timestamp to datetime
    model_df['Last Modified'] = pd.to_datetime(
        model_df['Last Modified'],
        unit='s'
    )
    
    # Convert size to MB
    model_df['Size (MB)'] = model_df['Size (bytes)'] / (1024 * 1024)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", len(model_df))
    
    with col2:
        total_size_mb = model_df['Size (MB)'].sum()
        st.metric("Total Size", f"{total_size_mb:.2f} MB")
    
    with col3:
        if len(model_df) > 0:
            latest_model = model_df.sort_values('Last Modified', ascending=False).iloc[0]
            st.metric("Latest Model", latest_model['Model File'])
    
    # Display table
    st.dataframe(
        model_df[['Model File', 'Size (MB)', 'Last Modified']],
        use_container_width=True
    )


def show_mlflow_experiments():
    """Show MLflow experiments."""
    st.subheader("🔬 MLflow Experiments")
    
    experiments = get_mlflow_experiments()
    
    if experiments is None:
        st.warning("Unable to load MLflow experiments. Ensure MLflow is configured properly.")
        return
    
    if len(experiments) == 0:
        st.info("No experiments found. Train models to create experiments.")
        return
    
    # Display experiments
    exp_data = []
    for exp in experiments:
        exp_data.append({
            'Name': exp.name,
            'Experiment ID': exp.experiment_id,
            'Artifact Location': exp.artifact_location,
            'Lifecycle Stage': exp.lifecycle_stage
        })
    
    exp_df = pd.DataFrame(exp_data)
    st.dataframe(exp_df, use_container_width=True)
    
    # Select experiment to view runs
    st.markdown("---")
    experiment_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("Select Experiment to View Runs", experiment_names)
    
    if selected_exp:
        show_experiment_runs(selected_exp)


def show_experiment_runs(experiment_name):
    """Show runs for a specific experiment."""
    st.subheader(f"📊 Runs for {experiment_name}")
    
    runs = get_mlflow_runs(experiment_name)
    
    if runs is None:
        st.warning("Unable to load runs for this experiment.")
        return
    
    if len(runs) == 0:
        st.info("No runs found for this experiment.")
        return
    
    # Display run statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Runs", len(runs))
    
    with col2:
        finished_runs = len(runs[runs['status'] == 'FINISHED'])
        st.metric("Finished Runs", finished_runs)
    
    with col3:
        if 'start_time' in runs.columns:
            latest_run_time = runs['start_time'].max()
            st.metric("Latest Run", latest_run_time.strftime("%Y-%m-%d %H:%M"))
    
    # Display runs table
    st.markdown("#### Recent Runs")
    
    # Select relevant columns
    display_cols = ['run_id', 'start_time', 'status']
    
    # Add metric columns if they exist
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    display_cols.extend(metric_cols[:5])  # Show first 5 metrics
    
    # Filter existing columns
    display_cols = [col for col in display_cols if col in runs.columns]
    
    st.dataframe(
        runs[display_cols].head(10),
        use_container_width=True
    )
    
    # Visualize metrics
    if metric_cols:
        st.markdown("#### Metrics Visualization")
        
        metric_to_plot = st.selectbox(
            "Select metric to visualize",
            metric_cols
        )
        
        if metric_to_plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Sort by start time
            plot_data = runs.sort_values('start_time')
            
            ax.plot(
                range(len(plot_data)),
                plot_data[metric_to_plot],
                marker='o',
                linewidth=2,
                markersize=8
            )
            
            ax.set_xlabel('Run Number')
            ax.set_ylabel(metric_to_plot.replace('metrics.', ''))
            ax.set_title(f'{metric_to_plot.replace("metrics.", "")} Over Runs')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()


def show_system_metrics():
    """Show system-level metrics."""
    st.subheader("💻 System Metrics")
    
    from emipredict.config.settings import Config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Configuration")
        config_info = {
            'MLflow Tracking URI': Config.MLFLOW_TRACKING_URI,
            'Random State': Config.RANDOM_STATE,
            'Train Split': f"{Config.TRAIN_SPLIT:.0%}",
            'Data Path': str(Config.DATA_PATH),
            'Models Directory': str(Config.MODELS_DIR)
        }
        
        for key, value in config_info.items():
            st.text(f"{key}: {value}")
    
    with col2:
        st.markdown("#### Model Thresholds")
        thresholds = {
            'Min Classification Accuracy': f"{Config.MIN_CLASSIFICATION_ACCURACY:.0%}",
            'Max Regression RMSE': f"₹{Config.MAX_REGRESSION_RMSE:,.0f}",
            'Min Precision': f"{Config.MIN_CLASSIFICATION_PRECISION:.0%}",
            'Min Recall': f"{Config.MIN_CLASSIFICATION_RECALL:.0%}"
        }
        
        for key, value in thresholds.items():
            st.text(f"{key}: {value}")


def show_logs():
    """Show recent application logs."""
    st.subheader("📋 Application Logs")
    
    from emipredict.config.settings import Config
    
    log_file = Config.LOG_FILE
    
    if not log_file.exists():
        st.info("No log file found.")
        return
    
    # Read last N lines
    n_lines = st.slider("Number of lines to display", 10, 100, 50)
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-n_lines:]
        
        # Display logs in a code block
        log_text = ''.join(recent_lines)
        st.code(log_text, language='log')
        
        # Download button
        st.download_button(
            label="Download Full Log",
            data=log_text,
            file_name="emipredict.log",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"Error reading log file: {str(e)}")


def show_data_statistics():
    """Show dataset statistics."""
    st.subheader("📊 Dataset Statistics")
    
    from emipredict.config.settings import Config
    
    if not Config.DATA_PATH.exists():
        st.warning("Dataset not found.")
        return
    
    try:
        # Load subset for stats
        df = pd.read_csv(Config.DATA_PATH, nrows=1000)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sample Size", f"{len(df):,}")
        
        with col2:
            st.metric("Features", len(df.columns))
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory (Sample)", f"{memory_mb:.2f} MB")
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing %", f"{missing_pct:.2f}%")
        
        # Target distribution
        st.markdown("#### Target Variable Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'emi_eligibility' in df.columns:
                st.write("**EMI Eligibility**")
                eligibility_counts = df['emi_eligibility'].value_counts()
                st.bar_chart(eligibility_counts)
        
        with col2:
            if 'max_monthly_emi' in df.columns:
                st.write("**Max Monthly EMI**")
                st.write(df['max_monthly_emi'].describe())
        
    except Exception as e:
        st.error(f"Error loading dataset statistics: {str(e)}")


def main():
    """Main function for Admin Monitoring page."""
    
    st.title("📈 Admin Monitoring Dashboard")
    st.markdown("Monitor model performance, experiments, and system metrics.")
    st.markdown("---")
    
    # Sidebar options
    st.sidebar.header("Monitoring Options")
    
    view_option = st.sidebar.selectbox(
        "Select View",
        [
            "Model Overview",
            "MLflow Experiments",
            "System Metrics",
            "Application Logs",
            "Dataset Statistics"
        ]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Display selected view
    if view_option == "Model Overview":
        show_model_overview()
    
    elif view_option == "MLflow Experiments":
        show_mlflow_experiments()
    
    elif view_option == "System Metrics":
        show_system_metrics()
    
    elif view_option == "Application Logs":
        show_logs()
    
    elif view_option == "Dataset Statistics":
        show_data_statistics()
    
    # System status
    st.markdown("---")
    st.subheader("🟢 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Application Running")
    
    with col2:
        # Check if models exist
        models = get_model_files()
        if models:
            st.success(f"✅ {len(models)} Models Found")
        else:
            st.warning("⚠️ No Models Found")
            st.info("Run: `python scripts/train_models.py`")
    
    with col3:
        # Check if MLflow is accessible
        experiments = get_mlflow_experiments()
        if experiments is not None and len(experiments) > 0:
            st.success(f"✅ MLflow Connected ({len(experiments)} experiments)")
        else:
            st.warning("⚠️ No MLflow Experiments")
            st.info("Models will create experiments when trained")


if __name__ == "__main__":
    main()

