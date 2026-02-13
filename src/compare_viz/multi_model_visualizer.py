#!/usr/bin/env python3
"""
Multi-Model Evaluation Visualizer

A generalized visualization module that can load and compare any number of
evaluation result JSON files from different models and methods.

Features:
- Load multiple JSON result files via CLI or programmatically
- Auto-detect model/method from result files
- Generate comparison charts for N models
- Support for custom labels and grouping
- Export comprehensive HTML reports and charts

Usage:
    # Compare multiple result files
    python multi_model_visualizer.py results/*.json --output comparison_output
    
    # Compare specific files with custom labels
    python multi_model_visualizer.py file1.json file2.json file3.json --labels "GPT-4" "Claude" "Gemini"
    
    # Load from a directory
    python multi_model_visualizer.py --dir results/ --pattern "evaluation_*.json"
"""

import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available - interactive charts will be disabled")

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class ModelResult:
    """Container for a single model's evaluation results."""
    file_path: str
    label: str
    provider: str
    model: str
    method: str  # 'RAG', 'Direct', etc.
    timestamp: str
    metrics: Dict[str, float]
    error_analysis: Dict[str, int]
    detailed_results: List[Dict] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)
    
    @property 
    def display_name(self) -> str:
        """Generate a display name for this result."""
        if self.label:
            return self.label
        return f"{self.method}:{self.provider}/{self.model}"
    
    @property
    def short_name(self) -> str:
        """Short name for chart labels."""
        if self.label:
            return self.label
        return f"{self.model}"


class MultiModelVisualizer:
    """
    Visualizer for comparing multiple model evaluation results.
    
    This class provides methods to:
    - Load results from multiple JSON files
    - Compare metrics across all loaded models
    - Generate various comparison visualizations
    - Export reports in multiple formats
    """
    
    # Standard metrics to compare
    STANDARD_METRICS = ['recall', 'precision', 'f1', 'exact_match_rate', 'mAP']
    
    # Color palettes for different numbers of models
    COLOR_PALETTES = {
        'default': px.colors.qualitative.Set2 if PLOTLY_AVAILABLE else None,
        'many': px.colors.qualitative.Alphabet if PLOTLY_AVAILABLE else None,
    }
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the multi-model visualizer.
        
        Args:
            output_dir: Directory to save outputs. Defaults to 'results/comparisons'
        """
        self.results: List[ModelResult] = []
        
        # Set output directory
        if output_dir is None:
            workspace_root = Path(__file__).parent.parent.parent
            output_dir = workspace_root / 'results' / 'comparisons'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_results(self, file_paths: List[str], labels: List[str] = None) -> 'MultiModelVisualizer':
        """
        Load evaluation results from multiple JSON files.
        
        Args:
            file_paths: List of paths to JSON result files
            labels: Optional custom labels for each file (must match length of file_paths)
            
        Returns:
            self for method chaining
        """
        if labels and len(labels) != len(file_paths):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of files ({len(file_paths)})")
        
        for i, file_path in enumerate(file_paths):
            label = labels[i] if labels else None
            self._load_single_result(file_path, label)
        
        print(f"✓ Loaded {len(self.results)} evaluation results")
        return self
    
    def load_from_directory(self, directory: str, pattern: str = "*.json", 
                            labels: Dict[str, str] = None) -> 'MultiModelVisualizer':
        """
        Load all matching JSON files from a directory.
        
        Args:
            directory: Path to directory containing result files
            pattern: Glob pattern to match files (default: "*.json")
            labels: Optional dict mapping filename to custom label
            
        Returns:
            self for method chaining
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = sorted(dir_path.glob(pattern))
        
        for file_path in files:
            label = labels.get(file_path.name) if labels else None
            try:
                self._load_single_result(str(file_path), label)
            except Exception as e:
                print(f"⚠️  Skipping {file_path.name}: {e}")
        
        print(f"✓ Loaded {len(self.results)} evaluation results from {directory}")
        return self
    
    def _load_single_result(self, file_path: str, label: str = None):
        """Load a single result file and extract metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata
        provider = data.get('provider', 'unknown')
        model = data.get('model', 'unknown')
        
        # Detect method (RAG, Direct, etc.)
        if 'rag_config' in data or path.name.startswith('rag_'):
            method = 'RAG'
        else:
            method = 'Direct'
        
        # Extract metrics
        avg_metrics = data.get('average_metrics', {})
        metrics = {
            'recall': avg_metrics.get('recall', 0),
            'precision': avg_metrics.get('precision', 0),
            'f1': avg_metrics.get('f1', 0),
            'exact_match_rate': avg_metrics.get('exact_match_rate', 0),
            'mAP': avg_metrics.get('mAP', 0),
            'avg_time_per_sample': avg_metrics.get('avg_time_per_sample', 0),
            'total_samples': avg_metrics.get('total_samples', 0),
            'evaluated_samples': avg_metrics.get('evaluated_samples', 0),
        }
        
        # Extract error analysis
        error_analysis = data.get('error_analysis', {})
        
        # Create ModelResult
        result = ModelResult(
            file_path=str(path),
            label=label,
            provider=provider,
            model=model,
            method=method,
            timestamp=data.get('timestamp', 'unknown'),
            metrics=metrics,
            error_analysis=error_analysis,
            detailed_results=data.get('detailed_results', []),
            raw_data=data
        )
        
        self.results.append(result)
        print(f"  Loaded: {result.display_name} ({path.name})")
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame with all models' metrics for comparison.
        
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        data = []
        for result in self.results:
            row = {
                'model': result.display_name,
                'short_name': result.short_name,
                'provider': result.provider,
                'method': result.method,
                **result.metrics
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_rankings(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rankings for each model across specified metrics.
        
        Args:
            metrics: List of metrics to rank. Defaults to standard metrics.
            
        Returns:
            DataFrame with rankings for each metric
        """
        if metrics is None:
            metrics = self.STANDARD_METRICS
        
        df = self.get_comparison_dataframe()
        rankings = pd.DataFrame({'model': df['model']})
        
        for metric in metrics:
            if metric in df.columns:
                # Higher is better for most metrics
                rankings[f'{metric}_rank'] = df[metric].rank(ascending=False, method='min')
        
        # Calculate average rank
        rank_cols = [col for col in rankings.columns if col.endswith('_rank')]
        rankings['avg_rank'] = rankings[rank_cols].mean(axis=1)
        rankings = rankings.sort_values('avg_rank')
        
        return rankings
    
    # ==================== Visualization Methods ====================
    
    def plot_metrics_comparison(self, metrics: List[str] = None, 
                                 figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Create a grouped bar chart comparing metrics across all models.
        
        Args:
            metrics: List of metrics to compare. Defaults to standard metrics.
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        if metrics is None:
            metrics = ['recall', 'precision', 'f1', 'exact_match_rate']
        
        df = self.get_comparison_dataframe()
        n_models = len(df)
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        colors = sns.color_palette("husl", n_models)
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row.get(m, 0) for m in metrics]
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=row['short_name'], color=colors[i])
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"metrics_comparison_{self.session_timestamp}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_radar_chart(self, metrics: List[str] = None) -> Optional[go.Figure]:
        """
        Create a radar chart comparing all models across metrics.
        
        Args:
            metrics: List of metrics to include. Defaults to standard metrics.
            
        Returns:
            Plotly Figure object (or None if Plotly not available)
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available for radar chart")
            return None
        
        if metrics is None:
            metrics = ['recall', 'precision', 'f1', 'exact_match_rate']
        
        df = self.get_comparison_dataframe()
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row.get(m, 0) for m in metrics]
            # Close the radar chart
            values_closed = values + [values[0]]
            metrics_closed = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=[m.replace('_', ' ').title() for m in metrics_closed],
                fill='toself',
                name=row['short_name'],
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Model Performance Radar Chart",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        # Save
        output_path = self.output_dir / f"radar_chart_{self.session_timestamp}.html"
        fig.write_html(output_path)
        print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_performance_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a heatmap showing all metrics for all models.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        df = self.get_comparison_dataframe()
        
        # Select numeric metrics
        metric_cols = ['recall', 'precision', 'f1', 'exact_match_rate', 'mAP']
        metric_cols = [c for c in metric_cols if c in df.columns]
        
        # Create heatmap data
        heatmap_data = df[metric_cols].values
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add labels
        ax.set_xticks(np.arange(len(metric_cols)))
        ax.set_yticks(np.arange(len(df)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_cols])
        ax.set_yticklabels(df['short_name'])
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add value annotations
        for i in range(len(df)):
            for j in range(len(metric_cols)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title("Model Performance Heatmap")
        fig.colorbar(im, ax=ax, label='Score')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"heatmap_{self.session_timestamp}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_f1_distribution(self, figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Create box plots showing F1 score distribution across samples for each model.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Collect F1 scores from detailed results
        f1_data = []
        labels = []
        
        for result in self.results:
            if result.detailed_results:
                f1_scores = [r.get('metrics', {}).get('f1', 0) for r in result.detailed_results]
                f1_data.append(f1_scores)
                labels.append(result.short_name)
        
        if not f1_data:
            print("⚠️  No detailed results available for distribution plot")
            return fig
        
        # Box plot
        ax1 = axes[0]
        bp = ax1.boxplot(f1_data, labels=labels, patch_artist=True)
        
        colors = sns.color_palette("husl", len(f1_data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score Distribution by Model')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Violin plot
        ax2 = axes[1]
        positions = list(range(len(f1_data)))
        parts = ax2.violinplot(f1_data, positions=positions, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Violin Plot')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"f1_distribution_{self.session_timestamp}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_time_comparison(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a bar chart comparing average time per sample across models.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        df = self.get_comparison_dataframe()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette("husl", len(df))
        bars = ax.bar(df['short_name'], df['avg_time_per_sample'], color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Average Time per Sample (seconds)')
        ax.set_title('Inference Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"time_comparison_{self.session_timestamp}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_interactive_comparison(self) -> Optional[go.Figure]:
        """
        Create an interactive Plotly dashboard with multiple comparison views.
        
        Returns:
            Plotly Figure object (or None if Plotly not available)
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available for interactive comparison")
            return None
        
        df = self.get_comparison_dataframe()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Metrics', 'Radar Comparison',
                'Time vs F1 Trade-off', 'Error Analysis'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatterpolar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        colors = px.colors.qualitative.Set2
        metrics = ['recall', 'precision', 'f1', 'exact_match_rate']
        
        # 1. Grouped bar chart
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=df['short_name'],
                    y=df[metric],
                    marker_color=colors[i % len(colors)],
                    showlegend=True if i == 0 else False,
                    legendgroup=metric
                ),
                row=1, col=1
            )
        
        # 2. Radar chart
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row.get(m, 0) for m in metrics]
            values_closed = values + [values[0]]
            metrics_closed = [m.replace('_', ' ').title() for m in metrics + [metrics[0]]]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=metrics_closed,
                    fill='toself',
                    name=row['short_name'],
                    line_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Time vs F1 scatter
        fig.add_trace(
            go.Scatter(
                x=df['avg_time_per_sample'],
                y=df['f1'],
                mode='markers+text',
                text=df['short_name'],
                textposition='top center',
                marker=dict(
                    size=15,
                    color=list(range(len(df))),
                    colorscale='Viridis'
                ),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Error analysis (if available)
        error_types = ['api_errors', 'empty_responses', 'parse_errors']
        for i, result in enumerate(self.results):
            error_values = [result.error_analysis.get(et, 0) for et in error_types]
            fig.add_trace(
                go.Bar(
                    name=result.short_name,
                    x=error_types,
                    y=error_values,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Multi-Model Evaluation Comparison Dashboard",
            height=900,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time per Sample (s)", row=2, col=1)
        fig.update_yaxes(title_text="F1 Score", row=2, col=1)
        
        # Save
        output_path = self.output_dir / f"interactive_dashboard_{self.session_timestamp}.html"
        fig.write_html(output_path)
        print(f"✓ Saved: {output_path}")
        
        return fig
    
    def generate_markdown_report(self) -> str:
        """
        Generate a comprehensive Markdown comparison report.
        
        Returns:
            Path to the generated report file
        """
        df = self.get_comparison_dataframe()
        rankings = self.get_rankings()
        
        report_lines = [
            "# Multi-Model Evaluation Comparison Report",
            f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nNumber of models compared: {len(self.results)}",
            "",
            "## Summary",
            "",
            "### Models Evaluated",
            ""
        ]
        
        for result in self.results:
            report_lines.append(f"- **{result.display_name}** ({result.method})")
            report_lines.append(f"  - Provider: {result.provider}")
            report_lines.append(f"  - Model: {result.model}")
            report_lines.append(f"  - File: `{Path(result.file_path).name}`")
        
        # Rankings
        report_lines.extend([
            "",
            "## Overall Rankings",
            "",
            "| Rank | Model | Avg Rank | F1 | Recall | Precision |",
            "|------|-------|----------|----|----|-----------|"
        ])
        
        for i, (_, row) in enumerate(rankings.iterrows(), 1):
            model_df = df[df['model'] == row['model']].iloc[0]
            report_lines.append(
                f"| {i} | {row['model']} | {row['avg_rank']:.2f} | "
                f"{model_df['f1']:.4f} | {model_df['recall']:.4f} | {model_df['precision']:.4f} |"
            )
        
        # Detailed metrics
        report_lines.extend([
            "",
            "## Detailed Metrics",
            "",
            "| Model | Recall | Precision | F1 | Exact Match | mAP | Avg Time |",
            "|-------|--------|-----------|----|----|-------|----------|"
        ])
        
        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['short_name']} | {row['recall']:.4f} | {row['precision']:.4f} | "
                f"{row['f1']:.4f} | {row['exact_match_rate']:.4f} | {row.get('mAP', 0):.4f} | "
                f"{row['avg_time_per_sample']:.2f}s |"
            )
        
        # Best performers
        report_lines.extend([
            "",
            "## Best Performers by Metric",
            ""
        ])
        
        metrics_to_highlight = ['recall', 'precision', 'f1', 'exact_match_rate', 'mAP']
        for metric in metrics_to_highlight:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_model = df.loc[best_idx, 'short_name']
                best_value = df.loc[best_idx, metric]
                report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {best_model} ({best_value:.4f})")
        
        # Fastest model
        fastest_idx = df['avg_time_per_sample'].idxmin()
        fastest_model = df.loc[fastest_idx, 'short_name']
        fastest_time = df.loc[fastest_idx, 'avg_time_per_sample']
        report_lines.append(f"- **Fastest**: {fastest_model} ({fastest_time:.2f}s per sample)")
        
        # Key findings
        report_lines.extend([
            "",
            "## Key Findings",
            ""
        ])
        
        # Best overall (by F1)
        best_f1_idx = df['f1'].idxmax()
        best_f1_model = df.loc[best_f1_idx, 'short_name']
        best_f1_value = df.loc[best_f1_idx, 'f1']
        report_lines.append(f"1. **Best Overall Performance**: {best_f1_model} achieves the highest F1 score of {best_f1_value:.4f}")
        
        # Speed vs quality trade-off
        if len(df) > 1:
            corr = df['f1'].corr(df['avg_time_per_sample'])
            if corr > 0.5:
                report_lines.append(f"2. **Speed-Quality Trade-off**: Slower models tend to perform better (correlation: {corr:.2f})")
            elif corr < -0.5:
                report_lines.append(f"2. **Efficiency**: Faster models also tend to perform well (correlation: {corr:.2f})")
            else:
                report_lines.append("2. **Speed Independence**: Model quality is largely independent of inference time")
        
        # Save report
        report_content = "\n".join(report_lines)
        output_path = self.output_dir / f"comparison_report_{self.session_timestamp}.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ Saved: {output_path}")
        return str(output_path)
    
    def generate_csv_export(self) -> str:
        """
        Export comparison data to CSV.
        
        Returns:
            Path to the generated CSV file
        """
        df = self.get_comparison_dataframe()
        output_path = self.output_dir / f"comparison_data_{self.session_timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
        return str(output_path)
    
    def run_full_comparison(self) -> Dict[str, str]:
        """
        Run all comparison visualizations and generate complete report.
        
        Returns:
            Dictionary with paths to all generated files
        """
        if not self.results:
            raise ValueError("No results loaded. Use load_results() or load_from_directory() first.")
        
        print(f"\n{'='*60}")
        print("Running Multi-Model Comparison Analysis")
        print(f"{'='*60}")
        print(f"Models: {len(self.results)}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        outputs = {}
        
        # Generate visualizations
        print("Generating visualizations...")
        
        try:
            self.plot_metrics_comparison()
            outputs['metrics_comparison'] = str(self.output_dir / f"metrics_comparison_{self.session_timestamp}.png")
        except Exception as e:
            print(f"⚠️  Metrics comparison failed: {e}")
        
        try:
            self.plot_performance_heatmap()
            outputs['heatmap'] = str(self.output_dir / f"heatmap_{self.session_timestamp}.png")
        except Exception as e:
            print(f"⚠️  Heatmap failed: {e}")
        
        try:
            self.plot_f1_distribution()
            outputs['f1_distribution'] = str(self.output_dir / f"f1_distribution_{self.session_timestamp}.png")
        except Exception as e:
            print(f"⚠️  F1 distribution failed: {e}")
        
        try:
            self.plot_time_comparison()
            outputs['time_comparison'] = str(self.output_dir / f"time_comparison_{self.session_timestamp}.png")
        except Exception as e:
            print(f"⚠️  Time comparison failed: {e}")
        
        if PLOTLY_AVAILABLE:
            try:
                self.plot_radar_chart()
                outputs['radar_chart'] = str(self.output_dir / f"radar_chart_{self.session_timestamp}.html")
            except Exception as e:
                print(f"⚠️  Radar chart failed: {e}")
            
            try:
                self.plot_interactive_comparison()
                outputs['interactive_dashboard'] = str(self.output_dir / f"interactive_dashboard_{self.session_timestamp}.html")
            except Exception as e:
                print(f"⚠️  Interactive dashboard failed: {e}")
        
        # Generate reports
        print("\nGenerating reports...")
        
        try:
            outputs['markdown_report'] = self.generate_markdown_report()
        except Exception as e:
            print(f"⚠️  Markdown report failed: {e}")
        
        try:
            outputs['csv_export'] = self.generate_csv_export()
        except Exception as e:
            print(f"⚠️  CSV export failed: {e}")
        
        print(f"\n{'='*60}")
        print("Comparison Analysis Complete!")
        print(f"{'='*60}")
        print(f"\nAll outputs saved to: {self.output_dir}")
        
        return outputs
    
    def print_summary(self):
        """Print a quick summary of the loaded results to console."""
        if not self.results:
            print("No results loaded.")
            return
        
        df = self.get_comparison_dataframe()
        rankings = self.get_rankings()
        
        print(f"\n{'='*70}")
        print("MULTI-MODEL COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        # Header
        print(f"{'Model':<30} {'F1':>10} {'Recall':>10} {'Precision':>10} {'Time':>10}")
        print('-' * 70)
        
        # Sort by F1 (descending)
        df_sorted = df.sort_values('f1', ascending=False)
        
        for _, row in df_sorted.iterrows():
            print(f"{row['short_name']:<30} "
                  f"{row['f1']:>10.4f} "
                  f"{row['recall']:>10.4f} "
                  f"{row['precision']:>10.4f} "
                  f"{row['avg_time_per_sample']:>9.2f}s")
        
        print()
        
        # Best model
        best_idx = df['f1'].idxmax()
        best_model = df.loc[best_idx, 'short_name']
        best_f1 = df.loc[best_idx, 'f1']
        print(f"🏆 Best Model (by F1): {best_model} ({best_f1:.4f})")
        
        print(f"\n{'='*70}\n")


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Compare multiple model evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare multiple JSON files
    python multi_model_visualizer.py results/eval1.json results/eval2.json results/eval3.json
    
    # Load all JSONs from a directory
    python multi_model_visualizer.py --dir results/ --pattern "evaluation_*.json"
    
    # With custom labels
    python multi_model_visualizer.py file1.json file2.json --labels "GPT-4" "Claude-3"
    
    # Specify output directory
    python multi_model_visualizer.py *.json --output my_comparison
        """
    )
    
    parser.add_argument('files', nargs='*', help='JSON result files to compare')
    parser.add_argument('--dir', '-d', help='Directory containing result files')
    parser.add_argument('--pattern', '-p', default='*.json', help='Glob pattern for files in directory')
    parser.add_argument('--labels', '-l', nargs='*', help='Custom labels for each file')
    parser.add_argument('--output', '-o', help='Output directory for visualizations')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary, no visualizations')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.files and not args.dir:
        parser.error("Provide either file paths or --dir")
    
    # Create visualizer
    visualizer = MultiModelVisualizer(output_dir=args.output)
    
    # Load results
    if args.dir:
        visualizer.load_from_directory(args.dir, args.pattern)
    else:
        # Expand glob patterns
        files = []
        for pattern in args.files:
            expanded = glob.glob(pattern)
            if expanded:
                files.extend(expanded)
            else:
                files.append(pattern)
        
        visualizer.load_results(files, labels=args.labels)
    
    if not visualizer.results:
        print("No valid result files found.")
        return
    
    # Run comparison
    visualizer.print_summary()
    
    if not args.summary_only:
        visualizer.run_full_comparison()


if __name__ == "__main__":
    main()
