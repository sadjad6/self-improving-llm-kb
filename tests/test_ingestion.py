"""Tests for the ingestion pipeline."""

from pathlib import Path

import pytest

from src.ingestion.parser import MarkdownParser
from src.ingestion.chunker import SemanticChunker

SAMPLE_MD = """# Test Document

#test-tag #sample

## Introduction

This is a test document about [[Machine Learning]] and [[Deep Learning]].
It has multiple sections to test chunking.

## Section One

Machine learning is a subset of artificial intelligence.
It focuses on building systems that learn from data.

Supervised learning uses labeled data for training.
Unsupervised learning finds patterns in unlabeled data.

## Section Two

Neural networks are computing systems inspired by biological brains.
They consist of layers of interconnected neurons.

Deep learning uses multiple hidden layers.
"""


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    fp = tmp_path / "test_doc.md"
    fp.write_text(SAMPLE_MD, encoding="utf-8")
    return fp


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    for i in range(3):
        fp = tmp_path / f"doc_{i}.md"
        fp.write_text(f"# Document {i}\n\n#tag{i}\n\nContent for document {i}.\n")
    return tmp_path


class TestMarkdownParser:
    def test_parse_file(self, sample_md_file: Path) -> None:
        parser = MarkdownParser()
        doc = parser.parse_file(sample_md_file)
        assert doc.title == "Test Document"
        assert "Machine Learning" in doc.links
        assert "Deep Learning" in doc.links
        assert "test-tag" in doc.tags
        assert "sample" in doc.tags
        assert len(doc.headings) >= 3

    def test_parse_directory(self, sample_dir: Path) -> None:
        parser = MarkdownParser()
        docs = parser.parse_directory(sample_dir)
        assert len(docs) == 3

    def test_parse_empty_directory(self, tmp_path: Path) -> None:
        parser = MarkdownParser()
        docs = parser.parse_directory(tmp_path)
        assert docs == []

    def test_extract_links_disabled(self, sample_md_file: Path) -> None:
        parser = MarkdownParser(extract_links=False)
        doc = parser.parse_file(sample_md_file)
        assert doc.links == []

    def test_extract_tags_disabled(self, sample_md_file: Path) -> None:
        parser = MarkdownParser(extract_tags=False)
        doc = parser.parse_file(sample_md_file)
        assert doc.tags == []


class TestSemanticChunker:
    def test_chunk_document(self, sample_md_file: Path) -> None:
        parser = MarkdownParser()
        doc = parser.parse_file(sample_md_file)
        chunker = SemanticChunker(max_tokens=512)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == doc.id
            assert chunk.content.strip() != ""

    def test_chunk_preserves_headings(self, sample_md_file: Path) -> None:
        parser = MarkdownParser()
        doc = parser.parse_file(sample_md_file)
        chunker = SemanticChunker(preserve_headings=True)
        chunks = chunker.chunk_document(doc)
        # At least some chunks should have heading context
        headings = [c.heading_context for c in chunks if c.heading_context]
        assert len(headings) > 0

    def test_small_max_tokens_produces_more_chunks(self, sample_md_file: Path) -> None:
        parser = MarkdownParser()
        doc = parser.parse_file(sample_md_file)
        large = SemanticChunker(max_tokens=2000).chunk_document(doc)
        small = SemanticChunker(max_tokens=50).chunk_document(doc)
        assert len(small) >= len(large)

    def test_chunk_documents_multiple(self, sample_dir: Path) -> None:
        parser = MarkdownParser()
        docs = parser.parse_directory(sample_dir)
        chunker = SemanticChunker()
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= len(docs)
