"""Retrieval system with dense, sparse, and hybrid strategies."""

from src.retrieval.sparse import SparseRetriever

__all__ = ["SparseRetriever"]

try:
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

    __all__ += ["DenseRetriever", "HybridRetriever", "CrossEncoderReranker"]
except ImportError:
    pass
