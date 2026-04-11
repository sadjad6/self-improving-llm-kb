"""Sparse retrieval using BM25."""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.utils.models import Chunk, RetrievalResult

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class SparseRetriever:
    """Sparse retrieval using BM25 (Okapi) scoring.

    BM25 excels at exact keyword matching and complements dense
    retrieval's semantic understanding in hybrid configurations.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Build a BM25 index from document chunks.

        Args:
            chunks: List of chunks to index.
        """
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required: pip install rank-bm25")
        if not chunks:
            logger.warning("No chunks to index.")
            return

        self._chunks = list(chunks)
        tokenized_corpus = [_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built with %d documents.", len(chunks))

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve chunks matching the query using BM25.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by BM25 score descending.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not initialized.")
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    method="sparse",
                )
            )
        return results

