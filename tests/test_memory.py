"""Tests for the memory store."""

from pathlib import Path

import pytest

from src.memory.store import MemoryStore
from src.utils.config import MemoryConfig
from src.utils.models import Chunk, QueryResult, RetrievalResult


def _make_query_result(query: str = "What is ML?", answer: str = "ML is machine learning.") -> QueryResult:
    chunk = Chunk(id="c1", document_id="d1", content="ML content", metadata={"title": "ML"})
    return QueryResult(
        query=query,
        answer=answer,
        retrieved_chunks=[RetrievalResult(chunk=chunk, score=0.9, method="dense")],
        retrieval_method="hybrid",
        latency_ms=100,
    )


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    config = MemoryConfig(
        store_path=str(tmp_path / "memory.jsonl"),
        summary_path=str(tmp_path / "summaries"),
        max_history=10,
    )
    return MemoryStore(config=config)


class TestMemoryStore:
    def test_store_interaction(self, memory_store: MemoryStore) -> None:
        result = _make_query_result()
        entry = memory_store.store_interaction(result)
        assert entry.query == "What is ML?"
        assert entry.importance_score > 0

    def test_deduplication(self, memory_store: MemoryStore) -> None:
        r1 = _make_query_result("What is ML?")
        r2 = _make_query_result("What is ML?")
        e1 = memory_store.store_interaction(r1)
        e2 = memory_store.store_interaction(r2)
        assert e2.access_count == 2
        assert e1.id == e2.id

    def test_get_relevant_history(self, memory_store: MemoryStore) -> None:
        memory_store.store_interaction(_make_query_result("What is machine learning?"))
        memory_store.store_interaction(_make_query_result("How do neural networks work?", "They use layers."))
        history = memory_store.get_relevant_history("machine learning basics")
        assert len(history) > 0

    def test_persistence(self, tmp_path: Path) -> None:
        config = MemoryConfig(
            store_path=str(tmp_path / "persist.jsonl"),
            summary_path=str(tmp_path / "summaries"),
        )
        store1 = MemoryStore(config=config)
        store1.store_interaction(_make_query_result())

        # Reload
        store2 = MemoryStore(config=config)
        assert len(store2._entries) == 1

    def test_pruning(self, tmp_path: Path) -> None:
        config = MemoryConfig(
            store_path=str(tmp_path / "prune.jsonl"),
            summary_path=str(tmp_path / "summaries"),
            max_history=3,
        )
        store = MemoryStore(config=config)
        for i in range(5):
            store.store_interaction(_make_query_result(f"Question number {i}?", f"Answer {i}"))
        assert len(store._entries) <= 3

    def test_stats(self, memory_store: MemoryStore) -> None:
        memory_store.store_interaction(_make_query_result())
        stats = memory_store.get_stats()
        assert stats["total_entries"] == 1
        assert stats["avg_importance"] > 0

    def test_get_recent(self, memory_store: MemoryStore) -> None:
        for i in range(4):
            memory_store.store_interaction(_make_query_result(f"Question {i}?", f"Answer {i}"))
        recent = memory_store.get_recent(n=2)
        assert len(recent) == 2
        # Most recent should come first
        assert "3" in recent[0].query

    def test_generate_summary_note(self, memory_store: MemoryStore) -> None:
        entry = memory_store.store_interaction(_make_query_result())
        path = memory_store.generate_summary_note(entry, "ML is a key AI technique.")
        assert path.exists()
        content = path.read_text()
        assert "ML is a key AI technique" in content

