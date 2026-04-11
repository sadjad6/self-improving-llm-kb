"""Tests for the retrieval system."""

import pytest

from src.retrieval.sparse import SparseRetriever, _tokenize
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.utils.models import Chunk


def _make_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(id="c1", document_id="d1", content="Machine learning is a subset of artificial intelligence", index=0, metadata={"title": "ML Basics"}),
        Chunk(id="c2", document_id="d1", content="Neural networks are inspired by biological brains", index=1, metadata={"title": "ML Basics"}),
        Chunk(id="c3", document_id="d2", content="Transformers use self-attention mechanisms for sequence processing", index=0, metadata={"title": "Transformers"}),
        Chunk(id="c4", document_id="d2", content="BERT is a bidirectional encoder transformer model", index=1, metadata={"title": "Transformers"}),
        Chunk(id="c5", document_id="d3", content="RAG combines retrieval with language model generation", index=0, metadata={"title": "RAG"}),
    ]


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("Hello, World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_empty(self) -> None:
        assert _tokenize("") == []


class TestSparseRetriever:
    def test_index_and_retrieve(self) -> None:
        retriever = SparseRetriever()
        chunks = _make_chunks()
        retriever.index(chunks)
        results = retriever.retrieve("machine learning artificial intelligence", top_k=3)
        assert len(results) > 0
        assert results[0].method == "sparse"
        # Top result should be about ML
        assert "machine" in results[0].chunk.content.lower() or "artificial" in results[0].chunk.content.lower()

    def test_retrieve_empty_index(self) -> None:
        retriever = SparseRetriever()
        results = retriever.retrieve("test query")
        assert results == []

    def test_scores_are_positive(self) -> None:
        retriever = SparseRetriever()
        retriever.index(_make_chunks())
        results = retriever.retrieve("neural networks")
        for r in results:
            assert r.score > 0


class TestDenseRetriever:
    def test_index_and_retrieve(self) -> None:
        retriever = DenseRetriever(index_path="data/indices/test_faiss.index")
        chunks = _make_chunks()
        retriever.index(chunks)
        results = retriever.retrieve("What is machine learning?", top_k=3)
        assert len(results) > 0
        assert results[0].method == "dense"

    def test_retrieve_returns_scores(self) -> None:
        retriever = DenseRetriever(index_path="data/indices/test_faiss.index")
        retriever.index(_make_chunks())
        results = retriever.retrieve("transformers attention")
        assert all(r.score is not None for r in results)


class TestHybridRetriever:
    def test_hybrid_retrieve(self) -> None:
        dense = DenseRetriever(index_path="data/indices/test_hybrid.index")
        sparse = SparseRetriever()
        hybrid = HybridRetriever(dense, sparse, dense_weight=0.6, sparse_weight=0.4)
        chunks = _make_chunks()
        hybrid.index(chunks)

        results = hybrid.retrieve("machine learning", top_k=3, method="hybrid")
        assert len(results) > 0
        assert results[0].method == "hybrid"

    def test_dense_only_mode(self) -> None:
        dense = DenseRetriever(index_path="data/indices/test_dense_only.index")
        sparse = SparseRetriever()
        hybrid = HybridRetriever(dense, sparse)
        hybrid.index(_make_chunks())
        results = hybrid.retrieve("transformers", top_k=2, method="dense")
        assert all(r.method == "dense" for r in results)

    def test_sparse_only_mode(self) -> None:
        dense = DenseRetriever(index_path="data/indices/test_sparse_only.index")
        sparse = SparseRetriever()
        hybrid = HybridRetriever(dense, sparse)
        hybrid.index(_make_chunks())
        results = hybrid.retrieve("transformers", top_k=2, method="sparse")
        assert all(r.method == "sparse" for r in results)

