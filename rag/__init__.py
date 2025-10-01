"""
RAG Pipeline Package
===================

This package contains modules for:
- Document ingestion and embedding (merged_pipeline.py)
- Vector database operations (qdrant.py) 
- Multithreaded query retrieval (retrieval.py)
"""

from .pipeline import run_merged_rag_pipeline, SharedEmbeddingModel
from .retrieval import run_multithreaded_retrieval, MultiThreadedRetriever

__all__ = [
    'run_merged_rag_pipeline',
    'SharedEmbeddingModel', 
    'run_multithreaded_retrieval',
    'MultiThreadedRetriever'
]