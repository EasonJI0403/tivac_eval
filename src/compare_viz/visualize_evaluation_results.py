#!/usr/bin/env python3
"""
Evaluation Results Visualization Script

This script visualizes evaluation JSON outputs from multiple models,
supporting both single file visualization and multi-model comparison.

Usage:
    # Single file visualization
    python visualize_evaluation_results.py --file results/eval1.json
    
    # Compare two files (legacy mode)
    python visualize_evaluation_results.py --file1 results/eval1.json --file2 results/eval2.json --compare
    
    # Compare multiple files
    python visualize_evaluation_results.py --files results/eval1.json results/eval2.json results/eval3.json

Features:
- Overall metrics comparison (bar charts, radar charts)
- Detailed per-sample analysis
- Error analysis
- Performance distribution histograms
- Time analysis
- Exportable HTML reports
- Support for N-model comparisons
"""

import json
import argparse
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class ModelMetadata:
    """Container for model metadata."""
    provider: str
    model: str
    method: str
    dataset_type: str
    timestamp: str
    total_samples: int
    file_path: str
    
    @property
    def display_name(self) -> str:
        return f"{self.method}:{self.model}"
    
    @property
    def short_name(self) -> str:
        return self.model


class EvaluationVisualizer:
    """Visualizer for multiple evaluation result files."""
    
    def __init__(self, file_paths: List[str] = None, file1_path: str = None, file2_path: str = None):
        """
        Initialize the visualizer with one or more result files.
        
        Args:
            file_paths: List of paths to result files
            file1_path: Path to first file (legacy mode)
            file2_path: Path to second file (legacy mode)
        """
        self.results: List[Dict] = []
        self.metadata: List[ModelMetadata] = []
        
        # Handle legacy two-file mode
        if file1_path:
            paths = [file1_path]
            if file2_path:
                paths.append(file2_path)
            file_paths = paths
        
        # Load all files
        if file_paths:
            for path in file_paths:
                self._load_file(path)
        
        # Create output directory
        workspace_root = Path(__file__).parent.parent.parent
        self.output_dir = workspace_root / 'results'
        self.output_dir.mkdir(exist_ok=True)
        
        # Legacy compatibility attributes
        self.file1_path = Path(file_paths[0]) if file_paths else None
        self.file2_path = Path(file_paths[1]) if file_paths and len(file_paths) > 1 else None
        self.data1 = self.results[0] if self.results else None
        self.data2 = self.results[1] if len(self.results) > 1 else None
        self.metadata1 = self.metadata[0] if self.metadata else None
        self.metadata2 = self.metadata[1] if len(self.metadata) > 1 else None
    
    def _load_file(self, file_path: str):
        """Load a single result file."""
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  File not found: {file_path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results.append(data)
        self.metadata.append(self._extract_metadata(data, str(path)))
        print(f"✓ Loaded: {self.metadata[-1].display_name} ({path.name})")
    
    def _extract_metadata(self, data: Dict, file_path: str) -> ModelMetadata:
        """Extract metadata from evaluation results."""
        # Detect method
        if 'rag_config' in data or Path(file_path).name.startswith('rag_'):
            method = 'RAG'
        else:
            method = 'Direct'
        
        return ModelMetadata(
            provider=data.get('provider', 'Unknown'),
            model=data.get('model', 'Unknown'),
            method=method,
            dataset_type=data.get('dataset_type', 'Unknown'),
            timestamp=data.get('timestamp', 'Unknown'),
            total_samples=data.get('average_metrics', {}).get('total_samples', 0),
            file_path=file_path
        )
    
    def create_overview_comparison(self):
        """Create overview comparison charts for multiple models."""
        if len(self.results) == 1:
            return self._create_single_overview()
        
        return self._create_multi_model_overview()
    
    def _create_multi_model_overview(self):
        """Create overview comparison for multiple models."""
        metric_names = ['recall', 'precision', 'f1', 'exact_match_rate']
        
        # Prepare data for all models
        comparison_data = {'Metric': metric_names}
        for i, (data, meta) in enumerate(zip(self.results, self.metadata)):
            metrics = data.get('average_metrics', {})
            comparison_data[meta.display_name] = [metrics.get(m, 0) for m in metric_names]
        
        df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Metrics Comparison', 'Performance Radar', 'Time Analysis', 'Error Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatterpolar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = px.colors.qualitative.Set2
        
        # Bar chart comparison
        for i, meta in enumerate(self.metadata):
            fig.add_trace(
                go.Bar(
                    name=meta.short_name,
                    x=df['Metric'],
                    y=df.iloc[:, i+1],
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=1
            )
        
        # Radar chart
        categories = ['Recall', 'Precision', 'F1', 'Exact Match']
        for i, (data, meta) in enumerate(zip(self.results, self.metadata)):
            metrics = data.get('average_metrics', {})
            values = [metrics.get(m, 0) for m in metric_names]
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill='toself',
                    name=meta.short_name,
                    line_color=colors[i % len(colors)]
                ),
                row=1, col=2
            )
        
        # Time comparison
        time_data = [data.get('average_metrics', {}).get('avg_time_per_sample', 0) for data in self.results]
        time_labels = [meta.short_name for meta in self.metadata]
        
        fig.add_trace(
            go.Bar(
                x=time_labels,
                y=time_data,
                name='Avg Time per Sample',
                marker_color=[colors[i % len(colors)] for i in range(len(self.results))]
            ),
            row=2, col=1
        )
        
        # Error analysis
        error_types = ['api_errors', 'empty_responses', 'parse_errors']
        for i, (data, meta) in enumerate(zip(self.results, self.metadata)):
            errors = data.get('error_analysis', {})
            error_values = [errors.get(et, 0) for et in error_types]
            
            fig.add_trace(
                go.Bar(
                    name=meta.short_name,
                    x=[et.replace('_', ' ').title() for et in error_types],
                    y=error_values,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        n_models = len(self.results)
        title = f"Evaluation Comparison: {n_models} Models"
        fig.update_layout(
            title_text=title,
            height=800,
            showlegend=True,
            barmode='group'
        )
        
        # Save
        output_path = self.output_dir / "comparison_overview.html"
        fig.write_html(output_path)
        print(f"✓ Comparison overview saved to: {output_path}")
        
        return fig
    
    def _create_legacy_comparison(self):
        """Legacy two-model comparison (for backward compatibility)."""
        # Extract metrics for both files
        metrics1 = self.data1.get('average_metrics', {})
        metrics2 = self.data2.get('average_metrics', {})
        
        metric_names = ['recall', 'precision', 'f1', 'exact_match_rate']
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': metric_names,
            f'{self.metadata1.display_name}': [metrics1.get(m, 0) for m in metric_names],
            f'{self.metadata2.display_name}': [metrics2.get(m, 0) for m in metric_names]
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Create side-by-side bar chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Metrics Comparison', 'Performance Radar', 'Time Analysis', 'Error Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatterpolar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Bar chart comparison
        fig.add_trace(
            go.Bar(
                name=f'{self.metadata1.method}',
                x=df['Metric'],
                y=df.iloc[:, 1],
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name=f'{self.metadata2.method}',
                x=df['Metric'],
                y=df.iloc[:, 2],
                marker_color='lightcoral'
            ),
            row=1, col=1
        )
        
        # Radar chart
        categories = ['Recall', 'Precision', 'F1', 'Exact Match']
        
        fig.add_trace(
            go.Scatterpolar(
                r=[metrics1.get(m, 0) for m in metric_names],
                theta=categories,
                fill='toself',
                name=f'{self.metadata1.method}',
                line_color='blue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatterpolar(
                r=[metrics2.get(m, 0) for m in metric_names],
                theta=categories,
                fill='toself',
                name=f'{self.metadata2.method}',
                line_color='red'
            ),
            row=1, col=2
        )
        
        # Time comparison
        time_data = [
            metrics1.get('avg_time_per_sample', 0),
            metrics2.get('avg_time_per_sample', 0)
        ]
        time_labels = [self.metadata1.method, self.metadata2.method]
        
        fig.add_trace(
            go.Bar(
                x=time_labels,
                y=time_data,
                name='Avg Time per Sample',
                marker_color=['lightblue', 'lightcoral']
            ),
            row=2, col=1
        )
        
        # Error analysis
        errors1 = self.data1.get('error_analysis', {})
        errors2 = self.data2.get('error_analysis', {})
        
        error_types = list(set(list(errors1.keys()) + list(errors2.keys())))
        error_data1 = [errors1.get(et, 0) for et in error_types]
        error_data2 = [errors2.get(et, 0) for et in error_types]
        
        fig.add_trace(
            go.Bar(
                name=f'{self.metadata1.method}',
                x=error_types,
                y=error_data1,
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                name=f'{self.metadata2.method}',
                x=error_types,
                y=error_data2,
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Evaluation Comparison: {self.metadata1.method} vs {self.metadata2.method}",
            height=800,
            showlegend=True
        )
        
        # Save
        output_path = self.output_dir / "comparison_overview.html"
        fig.write_html(output_path)
        print(f"✓ Comparison overview saved to: {output_path}")
        
        return fig
    
    def _create_single_overview(self):
        """Create overview for single file."""
        metrics = self.data1.get('average_metrics', {})
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Error Analysis', 'Database Stats', 'Sample Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Performance metrics
        metric_names = ['recall', 'precision', 'f1', 'exact_match_rate']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name='Metrics',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Error analysis
        errors = self.data1.get('error_analysis', {})
        if any(errors.values()):
            fig.add_trace(
                go.Pie(
                    labels=list(errors.keys()),
                    values=list(errors.values()),
                    name="Errors"
                ),
                row=1, col=2
            )
        
        # Database stats (if available)
        db_stats = self.data1.get('database_stats', {})
        if db_stats:
            fig.add_trace(
                go.Bar(
                    x=list(db_stats.keys()),
                    y=list(db_stats.values()),
                    name='DB Stats',
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
        
        # Performance distribution
        if 'detailed_results' in self.data1:
            f1_scores = [result.get('metrics', {}).get('f1', 0) 
                        for result in self.data1['detailed_results']]
            
            fig.add_trace(
                go.Histogram(
                    x=f1_scores,
                    name='F1 Score Distribution',
                    nbinsx=20
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text=f"Evaluation Overview: {self.metadata1.display_name}",
            height=800
        )
        
        output_path = self.output_dir / f"overview_{self.metadata1.method.lower()}.html"
        fig.write_html(output_path)
        print(f"✓ Overview saved to: {output_path}")
        
        return fig
    
    def create_detailed_analysis(self):
        """Create detailed per-sample analysis."""
        if not self.data1.get('detailed_results'):
            print("No detailed results found in the data.")
            return
        
        # Extract detailed results
        results1 = pd.DataFrame(self.data1['detailed_results'])
        
        if len(self.results) > 1 and self.data2 and self.data2.get('detailed_results'):
            results2 = pd.DataFrame(self.data2['detailed_results'])
            return self._create_comparative_detailed_analysis(results1, results2)
        else:
            return self._create_single_detailed_analysis(results1)
    
    def _create_comparative_detailed_analysis(self, results1, results2):
        """Create comparative detailed analysis."""
        # Merge results on entity/question
        merged = pd.merge(
            results1[['entity', 'metrics']].add_suffix('_1'),
            results2[['entity', 'metrics']].add_suffix('_2'),
            left_on='entity_1',
            right_on='entity_2',
            how='inner'
        )
        
        # Extract metrics
        metrics1_df = pd.json_normalize(merged['metrics_1'])
        metrics2_df = pd.json_normalize(merged['metrics_2'])
        
        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1 Score Comparison', 'Recall vs Precision', 'Performance Difference', 'Score Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # F1 Score comparison
        fig.add_trace(
            go.Scatter(
                x=metrics1_df['f1'],
                y=metrics2_df['f1'],
                mode='markers',
                name='F1 Comparison',
                text=merged['entity_1'],
                hovertemplate='<b>%{text}</b><br>Method 1 F1: %{x:.3f}<br>Method 2 F1: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add diagonal line for reference
        min_f1 = min(metrics1_df['f1'].min(), metrics2_df['f1'].min())
        max_f1 = max(metrics1_df['f1'].max(), metrics2_df['f1'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_f1, max_f1],
                y=[min_f1, max_f1],
                mode='lines',
                name='Equal Performance',
                line=dict(dash='dash', color='gray')
            ),
            row=1, col=1
        )
        
        # Recall vs Precision for both methods
        fig.add_trace(
            go.Scatter(
                x=metrics1_df['recall'],
                y=metrics1_df['precision'],
                mode='markers',
                name=f'{self.metadata1.method}',
                marker=dict(color='blue', opacity=0.7)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics2_df['recall'],
                y=metrics2_df['precision'],
                mode='markers',
                name=f'{self.metadata2.method}',
                marker=dict(color='red', opacity=0.7)
            ),
            row=1, col=2
        )
        
        # Performance difference (Method2 - Method1)
        f1_diff = metrics2_df['f1'] - metrics1_df['f1']
        entities_sorted = merged['entity_1'].iloc[f1_diff.argsort()]
        f1_diff_sorted = f1_diff.iloc[f1_diff.argsort()]
        
        colors = ['red' if x < 0 else 'green' for x in f1_diff_sorted]
        
        fig.add_trace(
            go.Bar(
                x=f1_diff_sorted,
                y=entities_sorted,
                orientation='h',
                name='F1 Difference',
                marker=dict(color=colors),
                text=[f'{x:.3f}' for x in f1_diff_sorted],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Score distribution comparison
        fig.add_trace(
            go.Histogram(
                x=metrics1_df['f1'],
                name=f'{self.metadata1.method} F1',
                opacity=0.7,
                nbinsx=15
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=metrics2_df['f1'],
                name=f'{self.metadata2.method} F1',
                opacity=0.7,
                nbinsx=15
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Detailed Performance Comparison",
            height=1000,
            showlegend=True
        )
        
        output_path = self.output_dir / "detailed_comparison.html"
        fig.write_html(output_path)
        print(f"✓ Detailed comparison saved to: {output_path}")
        
        return fig
    
    def _create_single_detailed_analysis(self, results):
        """Create detailed analysis for single file."""
        # Extract metrics
        metrics_df = pd.json_normalize(results['metrics'])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance by Entity', 'Metrics Distribution', 'Recall vs Precision', 'Error Analysis'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Performance by entity (top 20)
        top_entities = results.nlargest(20, 'metrics')
        top_f1_scores = [m.get('f1', 0) for m in top_entities['metrics']]
        
        fig.add_trace(
            go.Bar(
                x=top_entities['entity'],
                y=top_f1_scores,
                name='F1 Score',
                text=[f'{x:.3f}' for x in top_f1_scores],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Metrics distribution
        fig.add_trace(
            go.Histogram(
                x=metrics_df['f1'],
                name='F1 Distribution',
                nbinsx=20,
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=metrics_df['recall'],
                name='Recall Distribution',
                nbinsx=20,
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Recall vs Precision
        fig.add_trace(
            go.Scatter(
                x=metrics_df['recall'],
                y=metrics_df['precision'],
                mode='markers',
                name='Recall vs Precision',
                text=results['entity'],
                hovertemplate='<b>%{text}</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Box plots for all metrics
        metrics_names = ['recall', 'precision', 'f1', 'exact_match']
        for i, metric in enumerate(metrics_names):
            if metric in metrics_df.columns:
                fig.add_trace(
                    go.Box(
                        y=metrics_df[metric],
                        name=metric.replace('_', ' ').title(),
                        boxpoints='outliers'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text=f"Detailed Analysis: {self.metadata1.display_name}",
            height=1000
        )
        
        # Rotate x-axis labels for entity names
        fig.update_xaxes(tickangle=45, row=1, col=1)
        
        output_path = self.output_dir / f"detailed_{self.metadata1.method.lower()}.html"
        fig.write_html(output_path)
        print(f"✓ Detailed analysis saved to: {output_path}")
        
        return fig
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.data2:
            report_name = f"comparison_report_{timestamp}.html"
            title = "Evaluation Comparison Report"
        else:
            report_name = f"evaluation_report_{timestamp}.html"
            title = "Evaluation Report"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e6ffe6; }}
                .warning {{ background-color: #fff0e6; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """
        
        # Add metadata section
        html_content += self._generate_metadata_section()
        
        # Add metrics comparison
        html_content += self._generate_metrics_section()
        
        # Add conclusions
        html_content += self._generate_conclusions_section()
        
        html_content += """
        </body>
        </html>
        """
        
        report_path = self.output_dir / report_name
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Summary report saved to: {report_path}")
        return report_path
    
    def _generate_metadata_section(self):
        """Generate metadata section for the report."""
        html = "<div class='section'><h2>Evaluation Metadata</h2>"
        
        if len(self.results) > 1:
            # Multi-model comparison table
            html += "<table class='metrics-table'><tr><th>Aspect</th>"
            for meta in self.metadata:
                html += f"<th>{meta.short_name}</th>"
            html += "</tr>"
            
            aspects = ['Method', 'Model', 'Provider', 'Dataset Type', 'Total Samples', 'Timestamp']
            attrs = ['method', 'model', 'provider', 'dataset_type', 'total_samples', 'timestamp']
            
            for aspect, attr in zip(aspects, attrs):
                html += f"<tr><td>{aspect}</td>"
                for meta in self.metadata:
                    html += f"<td>{getattr(meta, attr, 'N/A')}</td>"
                html += "</tr>"
        else:
            html += f"""
            <table class='metrics-table'>
                <tr><th>Aspect</th><th>Value</th></tr>
                <tr><td>Method</td><td>{self.metadata1.method}</td></tr>
                <tr><td>Model</td><td>{self.metadata1.model}</td></tr>
                <tr><td>Provider</td><td>{self.metadata1.provider}</td></tr>
                <tr><td>Dataset Type</td><td>{self.metadata1.dataset_type}</td></tr>
                <tr><td>Total Samples</td><td>{self.metadata1.total_samples}</td></tr>
                <tr><td>Timestamp</td><td>{self.metadata1.timestamp}</td></tr>
            """
        
        html += "</table></div>"
        return html
    
    def _generate_metrics_section(self):
        """Generate metrics comparison section."""
        html = "<div class='section'><h2>Performance Metrics</h2>"
        
        metric_names = ['recall', 'precision', 'f1', 'exact_match_rate', 'avg_time_per_sample']
        
        if len(self.results) > 1:
            # Multi-model comparison table
            html += "<table class='metrics-table'><tr><th>Metric</th>"
            for meta in self.metadata:
                html += f"<th>{meta.short_name}</th>"
            html += "<th>Best</th></tr>"
            
            for metric in metric_names:
                html += f"<tr><td>{metric.replace('_', ' ').title()}</td>"
                
                values = []
                for data in self.results:
                    val = data.get('average_metrics', {}).get(metric, 0)
                    values.append(val)
                    html += f"<td>{val:.4f}</td>"
                
                # Find best (lowest for time, highest for others)
                if metric == 'avg_time_per_sample':
                    best_idx = values.index(min(values))
                else:
                    best_idx = values.index(max(values))
                
                html += f"<td class='highlight'>{self.metadata[best_idx].short_name}</td></tr>"
            
            html += "</table>"
        else:
            metrics1 = self.data1.get('average_metrics', {})
            html += "<table class='metrics-table'><tr><th>Metric</th><th>Value</th></tr>"
            for metric, value in metrics1.items():
                if isinstance(value, (int, float)):
                    html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _generate_conclusions_section(self):
        """Generate conclusions section."""
        html = "<div class='section'><h2>Key Findings</h2><ul>"
        
        metrics1 = self.data1.get('average_metrics', {})
        
        if len(self.results) > 1:
            # Multi-model comparison conclusions
            # Find best F1
            f1_scores = [data.get('average_metrics', {}).get('f1', 0) for data in self.results]
            best_f1_idx = f1_scores.index(max(f1_scores))
            best_f1_model = self.metadata[best_f1_idx].display_name
            best_f1_value = f1_scores[best_f1_idx]
            
            html += f"<li><strong>Best F1 Score:</strong> {best_f1_model} ({best_f1_value:.4f})</li>"
            
            # Find fastest
            times = [data.get('average_metrics', {}).get('avg_time_per_sample', float('inf')) for data in self.results]
            fastest_idx = times.index(min(times))
            fastest_model = self.metadata[fastest_idx].display_name
            fastest_time = times[fastest_idx]
            
            html += f"<li><strong>Fastest:</strong> {fastest_model} ({fastest_time:.2f}s per sample)</li>"
            
            # Speed vs quality trade-off
            if best_f1_idx != fastest_idx:
                time_diff = times[best_f1_idx] - fastest_time
                html += f"<li><strong>Trade-off:</strong> Best model is {time_diff:.1f}s slower than fastest</li>"
            else:
                html += f"<li><strong>Best Choice:</strong> {best_f1_model} is both best AND fastest!</li>"
        else:
            # Single file analysis
            f1_score = metrics1.get('f1', 0)
            exact_match = metrics1.get('exact_match_rate', 0)
            
            if f1_score > 0.9:
                html += "<li><strong>Excellent performance:</strong> F1 score above 0.9</li>"
            elif f1_score > 0.8:
                html += "<li><strong>Good performance:</strong> F1 score above 0.8</li>"
            else:
                html += "<li><strong>Room for improvement:</strong> F1 score below 0.8</li>"
                
            if exact_match < 0.3:
                html += "<li><strong>Low exact match rate:</strong> Consider improving answer precision</li>"
        
        html += "</ul></div>"
        return html
    
    def run_full_analysis(self):
        """Run complete analysis and generate all visualizations."""
        print("Starting evaluation visualization analysis...")
        print(f"Input file 1: {self.file1_path}")
        if self.file2_path:
            print(f"Input file 2: {self.file2_path}")
        
        # Create overview
        self.create_overview_comparison()
        
        # Create detailed analysis
        self.create_detailed_analysis()
        
        # Create summary report
        report_path = self.create_summary_report()
        
        print(f"\nAnalysis complete! All outputs saved to: {self.output_dir}")
        print(f"Summary report: {report_path}")
        
        return self.output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Visualize evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file visualization
    python visualize_evaluation_results.py --file results/eval1.json
    
    # Compare two files (legacy mode)
    python visualize_evaluation_results.py --file1 results/eval1.json --file2 results/eval2.json --compare
    
    # Compare multiple files
    python visualize_evaluation_results.py --files results/eval1.json results/eval2.json results/eval3.json
    
    # Use glob patterns
    python visualize_evaluation_results.py --files "results/evaluation_*.json"
        """
    )
    
    # Support multiple input modes
    parser.add_argument('--file', '--file1', dest='file1', help='Path to first/single evaluation results JSON file')
    parser.add_argument('--file2', help='Path to second evaluation results JSON file for comparison (legacy)')
    parser.add_argument('--files', '-f', nargs='+', help='Multiple JSON result files to compare')
    parser.add_argument('--compare', action='store_true', help='Enable comparison mode (legacy, auto-enabled with multiple files)')
    parser.add_argument('--output-dir', default='evaluation_visualizations', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Collect all files
    file_paths = []
    
    if args.files:
        # Expand glob patterns
        for pattern in args.files:
            expanded = glob.glob(pattern)
            if expanded:
                file_paths.extend(expanded)
            elif Path(pattern).exists():
                file_paths.append(pattern)
    
    if args.file1:
        if args.file1 not in file_paths:
            file_paths.insert(0, args.file1)
    
    if args.file2:
        if args.file2 not in file_paths:
            file_paths.append(args.file2)
    
    # Validate inputs
    if not file_paths:
        parser.error("Provide at least one file via --file, --file1, or --files")
    
    # Validate file existence
    valid_files = []
    for path in file_paths:
        if Path(path).exists():
            valid_files.append(path)
        else:
            print(f"⚠️  File not found: {path}")
    
    if not valid_files:
        print("Error: No valid files found.")
        return
    
    # Create visualizer
    print(f"\nLoading {len(valid_files)} result file(s)...")
    visualizer = EvaluationVisualizer(file_paths=valid_files)
    
    # Run analysis
    output_dir = visualizer.run_full_analysis()
    
    print(f"\n✓ Visualization complete! Check the output directory: {output_dir}")


if __name__ == "__main__":
    main()