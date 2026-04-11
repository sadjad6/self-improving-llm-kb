"""Cross-encoder reranker for improving retrieval precision."""

from __future__ import annotations

import logging
from typing import Optional

from src.utils.models import RetrievalResult

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model.

    Cross-encoders jointly encode the query and each candidate document,
    producing a more precise relevance score than bi-encoders at the cost
    of higher latency (O(k) forward passes instead of a single ANN search).

    Use this as a second-stage ranker *after* fast approximate retrieval.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        top_k: int = 3,
    ) -> None:
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.top_k = top_k
        self._model: Optional[CrossEncoder] = None  # type: ignore[type-arg]

    @property
    def model(self) -> CrossEncoder:  # type: ignore[type-arg]
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info("Loading cross-encoder reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using the cross-encoder.

        Args:
            query: The original search query.
            results: Candidate retrieval results from first-stage retrieval.

        Returns:
            Top-k results re-sorted by cross-encoder relevance score.
        """
        if not results:
            return results

        pairs = [[query, r.chunk.content] for r in results]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        top = reranked[: self.top_k]
        reranked_results = []
        for score, result in top:
            reranked_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(score),
                    method="reranked",
                )
            )

        logger.debug(
            "Reranker: %d candidates → %d results (model=%s)",
            len(results),
            len(reranked_results),
            self.model_name,
        )
        return reranked_results
