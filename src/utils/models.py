"""Core data models used across the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    """Return current UTC time as an ISO 8601 string (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Document:
    """A parsed document from the knowledge base."""

    id: str
    title: str
    content: str
    source_path: str
    headings: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk of text extracted from a document."""

    id: str
    document_id: str
    content: str
    heading_context: str = ""
    index: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""

    chunk: Chunk
    score: float
    method: str  # "dense", "sparse", "hybrid", "reranked"


@dataclass
class QueryResult:
    """Complete result of a query pipeline execution."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievalResult]
    retrieval_method: str
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)


@dataclass
class MemoryEntry:
    """An interaction stored in memory."""

    id: str
    query: str
    answer: str
    retrieved_context: list[str]
    timestamp: str
    importance_score: float = 0.5
    access_count: int = 1
    summary: str = ""
