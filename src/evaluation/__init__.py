"""
Evaluation module for Taiwan Value Chain QA system.

This module provides evaluators for testing different LLM models
and RAG approaches on value chain QA tasks.

Evaluators:
    - langchain_evaluator: Multi-provider LLM evaluation using LangChain
    - rag_evaluator: RAG evaluation with vector store retrieval
    - openai_evaluator: Legacy OpenAI-specific evaluation
    
Output Directory:
    All evaluation results are saved to results/
"""

__all__ = [
    'evaluate_langchain_models',
    'evaluate_rag_models',
]
