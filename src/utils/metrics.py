"""
Evaluation metrics for the Taiwan Value Chain QA system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics of a single prediction."""
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    exact_match: float = 0.0
    correct_count: int = 0
    predicted_count: int = 0
    actual_count: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    average_precision: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'recall': self.recall,
            'precision': self.precision,
            'f1': self.f1,
            'exact_match': self.exact_match,
            'correct_count': self.correct_count,
            'predicted_count': self.predicted_count,
            'actual_count': self.actual_count,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'average_precision': self.average_precision
        }


def calculate_metrics(predicted: List[str], actual: List[str]) -> EvaluationMetrics:
    """Calculate evaluation metrics for a single prediction.
    
    Args:
        predicted: List of predicted items (chains or companies)
        actual: List of actual/ground truth items
        
    Returns:
        EvaluationMetrics object with calculated values
    """
    # Convert to sets for comparison
    pred_set = set(p.strip() for p in predicted)
    actual_set = set(a.strip() for a in actual)
    
    # Calculate intersection
    correct = pred_set & actual_set
    correct_count = len(correct)
    
    # Calculate metrics
    recall = correct_count / len(actual_set) if actual_set else 0.0
    precision = correct_count / len(pred_set) if pred_set else 0.0
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # Exact match
    exact_match = 1.0 if pred_set == actual_set else 0.0
    
    return EvaluationMetrics(
        recall=recall,
        precision=precision,
        f1=f1,
        exact_match=exact_match,
        correct_count=correct_count,
        predicted_count=len(pred_set),
        actual_count=len(actual_set),
        false_positives=len(pred_set - actual_set),
        false_negatives=len(actual_set - pred_set)
    )


def calculate_average_precision(predicted: List[str], actual: List[str]) -> float:
    """Calculate Average Precision for a single query.
    
    Args:
        predicted: Ordered list of predicted items
        actual: List of actual/ground truth items
        
    Returns:
        Average Precision score (0.0 to 1.0)
    """
    if not actual or not predicted:
        return 0.0
    
    actual_set = set(actual)
    num_correct = 0
    sum_precision = 0.0
    
    for i, pred in enumerate(predicted, 1):
        if pred in actual_set:
            num_correct += 1
            precision_at_i = num_correct / i
            sum_precision += precision_at_i
    
    return sum_precision / len(actual_set) if actual_set else 0.0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple evaluations."""
    total_samples: int = 0
    mean_recall: float = 0.0
    mean_precision: float = 0.0
    mean_f1: float = 0.0
    mean_ap: float = 0.0  # Mean Average Precision (mAP)
    exact_match_rate: float = 0.0
    
    # Standard deviations
    std_recall: float = 0.0
    std_precision: float = 0.0
    std_f1: float = 0.0
    
    # Additional statistics
    median_f1: float = 0.0
    min_f1: float = 0.0
    max_f1: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_samples': self.total_samples,
            'mean_recall': self.mean_recall,
            'mean_precision': self.mean_precision,
            'mean_f1': self.mean_f1,
            'mean_ap': self.mean_ap,
            'exact_match_rate': self.exact_match_rate,
            'std_recall': self.std_recall,
            'std_precision': self.std_precision,
            'std_f1': self.std_f1,
            'median_f1': self.median_f1,
            'min_f1': self.min_f1,
            'max_f1': self.max_f1
        }


def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> AggregatedMetrics:
    """Aggregate metrics from multiple evaluations.
    
    Args:
        metrics_list: List of EvaluationMetrics objects
        
    Returns:
        AggregatedMetrics with mean, std, and other statistics
    """
    if not metrics_list:
        return AggregatedMetrics()
    
    import numpy as np
    
    recalls = [m.recall for m in metrics_list]
    precisions = [m.precision for m in metrics_list]
    f1s = [m.f1 for m in metrics_list]
    aps = [m.average_precision for m in metrics_list]
    exact_matches = [m.exact_match for m in metrics_list]
    
    return AggregatedMetrics(
        total_samples=len(metrics_list),
        mean_recall=float(np.mean(recalls)),
        mean_precision=float(np.mean(precisions)),
        mean_f1=float(np.mean(f1s)),
        mean_ap=float(np.mean(aps)),
        exact_match_rate=float(np.mean(exact_matches)),
        std_recall=float(np.std(recalls)),
        std_precision=float(np.std(precisions)),
        std_f1=float(np.std(f1s)),
        median_f1=float(np.median(f1s)),
        min_f1=float(np.min(f1s)),
        max_f1=float(np.max(f1s))
    )
