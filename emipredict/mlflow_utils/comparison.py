"""
MLflow Model Comparison Utilities

This module provides utilities for comparing multiple MLflow experiments
and models side-by-side for classification and regression tasks.

Author: EMI-Predict AI Team
Date: 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare multiple MLflow models and experiments.
    
    Provides comprehensive comparison of model performance metrics,
    hyperparameters, and artifacts across different runs.
    
    Example:
        >>> comparator = ModelComparator(experiment_name="EMI_Classification")
        >>> comparison_df = comparator.compare_models()
        >>> comparator.plot_metric_comparison('accuracy')
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize ModelComparator.
        
        Args:
            experiment_name: Name of MLflow experiment to analyze
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                logger.warning(f"Experiment '{experiment_name}' not found")
                self.experiment_id = None
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Loaded experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Error loading experiment: {str(e)}")
            self.experiment_id = None
    
    def get_all_runs(self) -> pd.DataFrame:
        """
        Get all runs from the experiment.
        
        Returns:
            DataFrame with run information
        """
        if self.experiment_id is None:
            logger.warning("No experiment loaded")
            return pd.DataFrame()
        
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time DESC"]
            )
            
            logger.info(f"Found {len(runs)} runs in experiment")
            return runs
        except Exception as e:
            logger.error(f"Error retrieving runs: {str(e)}")
            return pd.DataFrame()
    
    def compare_models(
        self,
        metric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare all models in the experiment.
        
        Args:
            metric_columns: Specific metric columns to include
            
        Returns:
            DataFrame with model comparison
        """
        runs = self.get_all_runs()
        
        if runs.empty:
            return pd.DataFrame()
        
        # Select relevant columns
        comparison_cols = ['run_id', 'start_time', 'tags.model_type', 'tags.n_classes']
        
        # Add metric columns
        if metric_columns is None:
            metric_columns = [col for col in runs.columns if col.startswith('metrics.')]
        else:
            metric_columns = [f'metrics.{m}' if not m.startswith('metrics.') else m 
                            for m in metric_columns]
        
        comparison_cols.extend(metric_columns)
        
        # Filter available columns
        available_cols = [col for col in comparison_cols if col in runs.columns]
        
        comparison_df = runs[available_cols].copy()
        
        # Clean column names
        comparison_df.columns = [
            col.replace('metrics.', '').replace('tags.', '') 
            for col in comparison_df.columns
        ]
        
        # Sort by best metric (first metric column)
        if len(metric_columns) > 0:
            first_metric = metric_columns[0].replace('metrics.', '')
            if first_metric in comparison_df.columns:
                comparison_df = comparison_df.sort_values(
                    by=first_metric, 
                    ascending=False
                )
        
        return comparison_df
    
    def get_best_model(
        self,
        metric: str = 'val_accuracy',
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to optimize
            maximize: If True, higher is better; if False, lower is better
            
        Returns:
            Dictionary with best model information
        """
        runs = self.get_all_runs()
        
        if runs.empty:
            return None
        
        metric_col = f'metrics.{metric}' if not metric.startswith('metrics.') else metric
        
        if metric_col not in runs.columns:
            logger.warning(f"Metric '{metric}' not found in runs")
            return None
        
        # Find best run
        if maximize:
            best_idx = runs[metric_col].idxmax()
        else:
            best_idx = runs[metric_col].idxmin()
        
        best_run = runs.loc[best_idx]
        
        return {
            'run_id': best_run['run_id'],
            'model_type': best_run.get('tags.model_type', 'Unknown'),
            'metric_value': best_run[metric_col],
            'metric_name': metric,
            'start_time': best_run['start_time']
        }
    
    def plot_metric_comparison(
        self,
        metric: str,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of a specific metric across all models.
        
        Args:
            metric: Metric to compare
            figsize: Figure size
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        runs = self.get_all_runs()
        
        if runs.empty:
            logger.warning("No runs to plot")
            return None
        
        metric_col = f'metrics.{metric}' if not metric.startswith('metrics.') else metric
        
        if metric_col not in runs.columns:
            logger.warning(f"Metric '{metric}' not found")
            return None
        
        # Prepare data
        plot_data = runs[['tags.model_type', metric_col]].copy()
        plot_data.columns = ['model_type', 'metric_value']
        plot_data = plot_data.dropna()
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot
        sns.barplot(
            data=plot_data,
            x='model_type',
            y='metric_value',
            ax=ax,
            palette='viridis'
        )
        
        ax.set_title(
            f'{metric.replace("_", " ").title()} Comparison Across Models',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Model Type', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_all_metrics_heatmap(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of all metrics across models.
        
        Args:
            metrics: List of metrics to include (None for all)
            figsize: Figure size
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        comparison_df = self.compare_models(metric_columns=metrics)
        
        if comparison_df.empty:
            logger.warning("No data to plot")
            return None
        
        # Select only numeric columns (metrics)
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric metrics found")
            return None
        
        # Prepare data for heatmap
        heatmap_data = comparison_df[['model_type'] + list(numeric_cols)].copy()
        heatmap_data = heatmap_data.groupby('model_type').mean()
        
        # Normalize each metric to 0-1 for better visualization
        heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            heatmap_normalized.T,
            annot=heatmap_data.T,
            fmt='.4f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Score'},
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title(
            'Model Performance Heatmap (All Metrics)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Model Type', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def generate_comparison_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        comparison_df = self.compare_models()
        
        if comparison_df.empty:
            return "No runs found for comparison."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"MODEL COMPARISON REPORT: {self.experiment_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Runs: {len(comparison_df)}")
        
        if 'model_type' in comparison_df.columns:
            model_counts = comparison_df['model_type'].value_counts()
            report_lines.append("\nRuns by Model Type:")
            for model, count in model_counts.items():
                report_lines.append(f"  - {model}: {count}")
        
        report_lines.append("")
        
        # Metric comparisons
        report_lines.append("METRIC COMPARISONS")
        report_lines.append("-" * 80)
        
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        
        for metric in numeric_cols:
            report_lines.append(f"\n{metric.upper()}:")
            
            # Overall statistics
            report_lines.append(f"  Mean: {comparison_df[metric].mean():.4f}")
            report_lines.append(f"  Std:  {comparison_df[metric].std():.4f}")
            report_lines.append(f"  Min:  {comparison_df[metric].min():.4f}")
            report_lines.append(f"  Max:  {comparison_df[metric].max():.4f}")
            
            # Best model for this metric
            if 'model_type' in comparison_df.columns:
                best_idx = comparison_df[metric].idxmax()
                best_model = comparison_df.loc[best_idx, 'model_type']
                best_value = comparison_df.loc[best_idx, metric]
                report_lines.append(f"  Best: {best_model} ({best_value:.4f})")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def compare_classification_regression(
    classification_exp: str = "EMI_Classification",
    regression_exp: str = "EMI_Regression"
) -> Dict[str, pd.DataFrame]:
    """
    Compare both classification and regression experiments.
    
    Args:
        classification_exp: Name of classification experiment
        regression_exp: Name of regression experiment
        
    Returns:
        Dictionary with comparison DataFrames
    """
    results = {}
    
    # Classification comparison
    clf_comparator = ModelComparator(classification_exp)
    results['classification'] = clf_comparator.compare_models()
    results['classification_best'] = clf_comparator.get_best_model('val_accuracy')
    
    # Regression comparison
    reg_comparator = ModelComparator(regression_exp)
    results['regression'] = reg_comparator.compare_models()
    results['regression_best'] = reg_comparator.get_best_model('val_rmse', maximize=False)
    
    logger.info("Comparison complete for both tasks")
    
    return results


def generate_model_selection_report(
    classification_exp: str = "EMI_Classification",
    regression_exp: str = "EMI_Regression",
    output_dir: str = "reports"
) -> None:
    """
    Generate comprehensive model selection report with visualizations.
    
    Args:
        classification_exp: Classification experiment name
        regression_exp: Regression experiment name
        output_dir: Directory to save reports
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info("Generating model selection report...")
    
    # Classification
    clf_comparator = ModelComparator(classification_exp)
    clf_report = clf_comparator.generate_comparison_report(
        output_path=str(output_path / "classification_comparison.txt")
    )
    
    # Plots for classification
    clf_comparator.plot_metric_comparison(
        'val_accuracy',
        save_path=str(output_path / "classification_accuracy_comparison.png")
    )
    
    clf_comparator.plot_all_metrics_heatmap(
        save_path=str(output_path / "classification_metrics_heatmap.png")
    )
    
    # Regression
    reg_comparator = ModelComparator(regression_exp)
    reg_report = reg_comparator.generate_comparison_report(
        output_path=str(output_path / "regression_comparison.txt")
    )
    
    # Plots for regression
    reg_comparator.plot_metric_comparison(
        'val_rmse',
        save_path=str(output_path / "regression_rmse_comparison.png")
    )
    
    reg_comparator.plot_all_metrics_heatmap(
        save_path=str(output_path / "regression_metrics_heatmap.png")
    )
    
    logger.info(f"Reports generated in {output_dir}/")
    print(f"\n✅ Model selection reports saved to: {output_dir}/")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate comprehensive reports
    generate_model_selection_report()
    
    # Individual comparisons
    clf_comparator = ModelComparator("EMI_Classification")
    clf_comparison = clf_comparator.compare_models()
    print("\nClassification Model Comparison:")
    print(clf_comparison)
    
    best_clf = clf_comparator.get_best_model('val_accuracy')
    print(f"\nBest Classification Model: {best_clf}")

