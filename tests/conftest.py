"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


from src.utils.models import Chunk, QueryResult, RetrievalResult

SAMPLE_MD = """# Sample Document

#machine-learning #test

## Overview

Machine learning is a field of [[Artificial Intelligence]].
It enables systems to learn from data.

## Methods

Supervised learning uses labeled training data.
Unsupervised learning discovers hidden patterns.
"""


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """A small set of chunks useful across test modules."""
    return [
        Chunk(
            id="c1", document_id="d1", index=0,
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"title": "ML Basics"},
        ),
        Chunk(
            id="c2", document_id="d1", index=1,
            content="Neural networks are inspired by biological brains.",
            metadata={"title": "ML Basics"},
        ),
        Chunk(
            id="c3", document_id="d2", index=0,
            content="Transformers use self-attention mechanisms.",
            metadata={"title": "Transformers"},
        ),
    ]


@pytest.fixture
def sample_query_result() -> QueryResult:
    """A minimal QueryResult for testing downstream components."""
    chunk = Chunk(id="c1", document_id="d1", content="ML content", metadata={"title": "ML"})
    return QueryResult(
        query="What is ML?",
        answer="Machine learning is a branch of AI.",
        retrieved_chunks=[RetrievalResult(chunk=chunk, score=0.9, method="dense")],
        retrieval_method="hybrid",
        latency_ms=42,
    )


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """Write a sample markdown file and return its path."""
    fp = tmp_path / "sample.md"
    fp.write_text(SAMPLE_MD, encoding="utf-8")
    return fp

