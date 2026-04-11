"""Evaluation metrics for retrieval and answer quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.utils.models import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""

    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics.

    Supports Recall@K and Mean Reciprocal Rank (MRR) for measuring
    how effectively the retrieval system finds relevant documents.
    """

    @staticmethod
    def recall_at_k(
        retrieved: list[RetrievalResult],
        relevant_ids: set[str],
        k: int,
    ) -> EvalResult:
        """Compute Recall@K.

        Args:
            retrieved: List of retrieval results.
            relevant_ids: Set of chunk IDs that are relevant.
            k: Number of top results to consider.

        Returns:
            EvalResult with recall score.
        """
        if not relevant_ids:
            return EvalResult("recall@k", 0.0, {"k": k, "note": "no relevant docs"})

        top_k_ids = {r.chunk.id for r in retrieved[:k]}
        hits = len(top_k_ids & relevant_ids)
        recall = hits / len(relevant_ids)

        return EvalResult(
            metric_name=f"recall@{k}",
            score=recall,
            details={"k": k, "hits": hits, "total_relevant": len(relevant_ids)},
        )

    @staticmethod
    def mrr(
        retrieved: list[RetrievalResult],
        relevant_ids: set[str],
    ) -> EvalResult:
        """Compute Mean Reciprocal Rank.

        Args:
            retrieved: List of retrieval results.
            relevant_ids: Set of chunk IDs that are relevant.

        Returns:
            EvalResult with MRR score.
        """
        for rank, result in enumerate(retrieved, 1):
            if result.chunk.id in relevant_ids:
                return EvalResult(
                    metric_name="mrr",
                    score=1.0 / rank,
                    details={"first_relevant_rank": rank},
                )
        return EvalResult("mrr", 0.0, {"first_relevant_rank": -1})

    def evaluate(
        self,
        retrieved: list[RetrievalResult],
        relevant_ids: set[str],
        k_values: list[int] | None = None,
    ) -> list[EvalResult]:
        """Run all retrieval metrics.

        Args:
            retrieved: List of retrieval results.
            relevant_ids: Set of relevant chunk IDs.
            k_values: List of K values for Recall@K.

        Returns:
            List of EvalResult for each metric.
        """
        k_values = k_values or [1, 3, 5, 10]
        results = []
        for k in k_values:
            results.append(self.recall_at_k(retrieved, relevant_ids, k))
        results.append(self.mrr(retrieved, relevant_ids))
        return results


class AnswerEvaluator:
    """Evaluates answer quality using heuristic and LLM-based scoring."""

    @staticmethod
    def heuristic_score(
        answer: str,
        context_texts: list[str],
        query: str,
    ) -> EvalResult:
        """Compute heuristic quality score for an answer.

        Checks: length, grounding in context, query term coverage.

        Args:
            answer: The generated answer.
            context_texts: The context chunks used.
            query: The original query.

        Returns:
            EvalResult with composite heuristic score.
        """
        scores: dict[str, float] = {}

        # Length score (prefer substantial answers, penalize very short/long)
        length = len(answer.split())
        if length < 10:
            scores["length"] = 0.2
        elif length < 30:
            scores["length"] = 0.6
        elif length <= 300:
            scores["length"] = 1.0
        else:
            scores["length"] = 0.7

        # Grounding score (check if answer terms appear in context)
        answer_terms = set(answer.lower().split())
        context_combined = " ".join(context_texts).lower()
        context_terms = set(context_combined.split())
        if answer_terms:
            grounded = len(answer_terms & context_terms) / len(answer_terms)
            scores["grounding"] = min(1.0, grounded)
        else:
            scores["grounding"] = 0.0

        # Query coverage (does answer address query terms?)
        query_terms = set(query.lower().split()) - {
            "what",
            "is",
            "the",
            "a",
            "an",
            "how",
            "why",
            "does",
        }
        if query_terms:
            coverage = len(query_terms & answer_terms) / len(query_terms)
            scores["query_coverage"] = min(1.0, coverage)
        else:
            scores["query_coverage"] = 0.5

        # Hallucination indicator (answer says it can't answer)
        cant_answer = any(
            phrase in answer.lower()
            for phrase in ["i don't have enough", "not enough information", "cannot answer"]
        )
        scores["has_answer"] = 0.3 if cant_answer else 1.0

        composite = sum(scores.values()) / len(scores)
        return EvalResult(
            metric_name="heuristic_score",
            score=composite,
            details=scores,
        )

    @staticmethod
    def llm_judge_prompt(query: str, answer: str, context: str) -> str:
        """Generate a prompt for LLM-as-judge evaluation.

        Returns the prompt string to send to an LLM for scoring.
        """
        return (
            f"Rate the following answer on a scale of 1-5 for:\n"
            f"1. Relevance to the question\n"
            f"2. Faithfulness to the provided context\n"
            f"3. Completeness\n\n"
            f"Question: {query}\n\n"
            f"Context: {context[:2000]}\n\n"
            f"Answer: {answer}\n\n"
            f'Respond with JSON: {{"relevance": X, "faithfulness": X, "completeness": X}}'
        )
