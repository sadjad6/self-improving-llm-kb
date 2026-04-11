"""Hybrid retrieval combining dense and sparse methods."""

from __future__ import annotations

import logging
from typing import Optional

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.utils.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse scores.

    Uses Reciprocal Rank Fusion (RRF) and weighted score combination
    to merge results from dense (semantic) and sparse (keyword) retrieval,
    leveraging the strengths of both approaches.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks in both dense and sparse retrievers.

        Args:
            chunks: List of chunks to index.
        """
        self.dense.index(chunks)
        self.sparse.index(chunks)
        logger.info("Hybrid index built with %d chunks.", len(chunks))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> list[RetrievalResult]:
        """Retrieve chunks using the specified method.

        Args:
            query: The search query.
            top_k: Number of results to return.
            method: Retrieval strategy - "dense", "sparse", or "hybrid".

        Returns:
            List of RetrievalResult sorted by combined score.
        """
        if method == "dense":
            return self.dense.retrieve(query, top_k)
        elif method == "sparse":
            return self.sparse.retrieve(query, top_k)
        else:
            return self._hybrid_retrieve(query, top_k)

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Combine dense and sparse results using RRF + weighted scores."""
        # Retrieve from both sources with higher k for fusion
        fetch_k = top_k * 3
        dense_results = self.dense.retrieve(query, fetch_k)
        sparse_results = self.sparse.retrieve(query, fetch_k)

        # Reciprocal Rank Fusion
        rrf_k = 60  # Standard RRF constant
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, result in enumerate(dense_results):
            cid = result.chunk.id
            rrf_score = self.dense_weight / (rrf_k + rank + 1)
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_score
            chunk_map[cid] = result.chunk

        for rank, result in enumerate(sparse_results):
            cid = result.chunk.id
            rrf_score = self.sparse_weight / (rrf_k + rank + 1)
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_score
            chunk_map[cid] = result.chunk

        # Sort by combined score
        sorted_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:top_k]

        results = [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=chunk_scores[cid],
                method="hybrid",
            )
            for cid in sorted_ids
        ]

        logger.debug(
            "Hybrid retrieval: %d dense + %d sparse → %d fused results",
            len(dense_results), len(sparse_results), len(results),
        )
        return results

