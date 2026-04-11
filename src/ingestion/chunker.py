"""Semantic document chunking with context preservation."""

from __future__ import annotations

import hashlib
import logging
import re


from src.utils.models import Chunk, Document

logger = logging.getLogger(__name__)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class SemanticChunker:
    """Chunks documents at semantic boundaries while preserving context.

    Unlike fixed-size chunking, this splits at natural boundaries
    (headings, paragraph breaks) and prepends heading context to each chunk.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        preserve_headings: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_headings = preserve_headings

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into semantically coherent chunks.

        Args:
            document: The document to chunk.

        Returns:
            List of Chunk objects with heading context preserved.
        """
        sections = self._split_by_headings(document.content)
        chunks: list[Chunk] = []

        for heading, section_text in sections:
            section_chunks = self._split_section(section_text, heading)
            for i, text in enumerate(section_chunks):
                chunk_id = hashlib.md5(
                    f"{document.id}:{heading}:{i}".encode()
                ).hexdigest()[:12]
                chunk = Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=text,
                    heading_context=heading,
                    index=len(chunks),
                    token_count=self._estimate_tokens(text),
                    metadata={
                        "source": document.source_path,
                        "title": document.title,
                    },
                )
                chunks.append(chunk)

        logger.info(
            "Chunked '%s' into %d chunks", document.title, len(chunks)
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: List of documents to chunk.

        Returns:
            Flat list of all chunks.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Total chunks created: %d from %d documents", len(all_chunks), len(documents))
        return all_chunks

    def _split_by_headings(self, content: str) -> list[tuple[str, str]]:
        """Split content into (heading, text) pairs at heading boundaries."""
        sections: list[tuple[str, str]] = []
        matches = list(HEADING_PATTERN.finditer(content))

        if not matches:
            return [("", content.strip())]

        # Text before first heading
        pre_text = content[: matches[0].start()].strip()
        if pre_text:
            sections.append(("", pre_text))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[start:end].strip()
            if text:
                sections.append((heading, text))

        return sections

    def _split_section(self, text: str, heading: str) -> list[str]:
        """Split a section into chunks respecting token limits."""
        if self._estimate_tokens(text) <= self.max_tokens:
            prefix = f"## {heading}\n\n" if heading and self.preserve_headings else ""
            return [f"{prefix}{text}"]

        paragraphs = re.split(r"\n\n+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        prefix = f"## {heading}\n\n" if heading and self.preserve_headings else ""

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if current_tokens + para_tokens > self.max_tokens and current:
                chunks.append(prefix + "\n\n".join(current))
                # Overlap: keep last paragraph
                if self.overlap_tokens > 0 and current:
                    overlap_text = current[-1]
                    current = [overlap_text]
                    current_tokens = self._estimate_tokens(overlap_text)
                else:
                    current = []
                    current_tokens = 0
            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append(prefix + "\n\n".join(current))

        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars per token for English)."""
        return len(text) // 4

