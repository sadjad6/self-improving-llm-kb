"""Integration tests for the KnowledgePipeline (no LLM required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.parser import MarkdownParser
from src.ingestion.chunker import SemanticChunker
from src.utils.models import Chunk, RetrievalResult


SAMPLE_DOCS = {
    "ml_basics.md": (
        "# Machine Learning Basics\n\n"
        "#machine-learning\n\n"
        "## Overview\n\n"
        "Machine learning enables systems to learn from data.\n\n"
        "## Types\n\n"
        "Supervised, unsupervised, and reinforcement learning.\n"
    ),
    "transformers.md": (
        "# Transformers\n\n"
        "#deep-learning\n\n"
        "## Attention\n\n"
        "Self-attention allows every position to attend to all positions.\n"
    ),
}


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a temporary knowledge base directory."""
    for name, content in SAMPLE_DOCS.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


class TestIngestionFlow:
    """Test the ingestion pipeline (parser + chunker) end-to-end."""

    def test_parse_and_chunk(self, knowledge_dir: Path) -> None:
        parser = MarkdownParser()
        docs = parser.parse_directory(knowledge_dir)
        assert len(docs) == 2

        chunker = SemanticChunker(max_tokens=512, preserve_headings=True)
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= 2

        # All chunks should have document IDs and non-empty content
        for chunk in chunks:
            assert chunk.document_id
            assert chunk.content.strip()

    def test_tags_and_links_extracted(self, knowledge_dir: Path) -> None:
        parser = MarkdownParser()
        docs = parser.parse_directory(knowledge_dir)
        all_tags = set()
        for doc in docs:
            all_tags.update(doc.tags)
        assert "machine-learning" in all_tags
        assert "deep-learning" in all_tags

    def test_chunk_metadata_preserved(self, knowledge_dir: Path) -> None:
        parser = MarkdownParser()
        docs = parser.parse_directory(knowledge_dir)
        chunker = SemanticChunker(max_tokens=512)
        chunks = chunker.chunk_documents(docs)
        titles = {c.metadata.get("title") for c in chunks}
        assert "Machine Learning Basics" in titles
        assert "Transformers" in titles

    def test_heading_context_attached(self, knowledge_dir: Path) -> None:
        parser = MarkdownParser()
        docs = parser.parse_directory(knowledge_dir)
        chunker = SemanticChunker(max_tokens=512, preserve_headings=True)
        chunks = chunker.chunk_documents(docs)
        headings = [c.heading_context for c in chunks if c.heading_context]
        assert len(headings) > 0


class TestRetrieveOnly:
    """Test the retrieve_only path via mocked pipeline."""

    def test_sparse_retrieve_only(self, knowledge_dir: Path) -> None:
        """Test sparse retrieval works end-to-end with real BM25."""
        pytest.importorskip("rank_bm25")

        from src.retrieval.sparse import SparseRetriever

        parser = MarkdownParser()
        docs = parser.parse_directory(knowledge_dir)
        chunker = SemanticChunker(max_tokens=512)
        chunks = chunker.chunk_documents(docs)

        retriever = SparseRetriever()
        retriever.index(chunks)
        results = retriever.retrieve("machine learning", top_k=3)
        assert len(results) > 0
        # Best result should mention machine learning
        assert "machine" in results[0].chunk.content.lower()

