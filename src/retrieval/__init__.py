"""Retrieval system with dense, sparse, and hybrid strategies."""

from src.retrieval.sparse import SparseRetriever

__all__ = ["SparseRetriever"]

try:
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    __all__ += ["DenseRetriever", "HybridRetriever"]
except ImportError:
    pass

