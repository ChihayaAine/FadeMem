"""
RAG (Retrieval-Augmented Generation) Module

This module implements a LangGraph-based RAG system that integrates with the
dual-layer memory architecture to provide enhanced information retrieval and
context-aware response generation.

Key Components:
- Vector store for embedding storage and similarity search
- LangGraph workflow for query processing and response generation
- Integration with text-embedding-small-3 model for embeddings
- Memory-aware retrieval combining vector search with memory importance scores
"""

from .vector_store import MemoryVectorStore
from .rag_workflow import RAGWorkflow
from .embedding_service import OpenAIEmbeddingService
from .retrieval_chain import MemoryRetrievalChain

__all__ = [
    'MemoryVectorStore',
    'RAGWorkflow', 
    'OpenAIEmbeddingService',
    'MemoryRetrievalChain'
]
