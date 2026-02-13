"""
Configuration dataclasses for the evaluation system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 30
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval-Augmented Generation) system."""
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 30
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # RAG-specific parameters
    embedding_provider: str = "openai"  # "openai", "huggingface", "ollama"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    score_threshold: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    data_dir: str = "datasets/demo/individual_chains"
    
    extra_params: Dict[str, Any] = field(default_factory=dict)



# Output directory configuration
import os
from pathlib import Path

def get_output_dir(module: str) -> Path:
    """Get the output directory for a module.
    
    Args:
        module: One of 'evaluation', 'compare_viz', 'qa_generation'
        
    Returns:
        Path to the output directory
    """
    base_dir = Path(__file__).parent.parent / module / 'outputs'
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_evaluation_output_dir() -> Path:
    """Get the output directory for evaluation results."""
    return get_output_dir('evaluation')


def get_compare_viz_output_dir() -> Path:
    """Get the output directory for comparison and visualization results."""
    return get_output_dir('compare_viz')
