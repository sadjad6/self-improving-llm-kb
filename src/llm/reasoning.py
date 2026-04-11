"""LLM reasoning layer with intelligent context selection."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from src.utils.config import LLMConfig
from src.utils.models import RetrievalResult

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    # Fallback: no retry decorator
    def retry(**kwargs: Any):  # type: ignore[misc]
        def decorator(fn: Any) -> Any:
            return fn

        return decorator

    stop_after_attempt = None  # type: ignore[assignment]
    wait_exponential = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ANSWER_TEMPLATE = """Based on the following context from the knowledge base, answer the user's question.

## Retrieved Context

{context}

## Instructions
- Answer ONLY based on the provided context above
- If the context doesn't contain enough information, say "I don't have enough information to answer this"
- Cite specific details from the context to support your answer
- Be concise but thorough
- Do not hallucinate or add information not present in the context

## Question
{question}

## Answer"""

SUMMARY_TEMPLATE = """Based on the following Q&A interaction, generate a concise knowledge summary 
that captures the key information for future reference.

Question: {question}
Answer: {answer}

Write a 2-3 sentence summary that captures the essential knowledge:"""


class LLMReasoner:
    """LLM-powered reasoning layer with context-aware generation.

    Handles intelligent context selection, prompt construction, and
    answer generation with citation grounding.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:  # type: ignore[return-value]
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required for LLM reasoning. "
                    "Install it with: pip install openai"
                )
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. " "Set it or create a .env file."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def generate_answer(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        max_context_chunks: int = 5,
    ) -> tuple[str, dict[str, int]]:
        """Generate an answer grounded in retrieved context.

        Uses intelligent context selection (not naive top-k) by
        considering diversity and relevance.

        Args:
            query: The user's question.
            retrieval_results: Retrieved chunks with scores.
            max_context_chunks: Maximum chunks to include in context.

        Returns:
            Tuple of (answer_text, token_usage_dict).
        """
        selected = self._select_context(retrieval_results, max_context_chunks)
        context_str = self._format_context(selected)
        prompt = ANSWER_TEMPLATE.format(context=context_str, question=query)

        answer, usage = self._call_llm(prompt)
        return answer, usage

    def generate_summary(self, question: str, answer: str) -> str:
        """Generate a knowledge summary from a Q&A interaction.

        Args:
            question: The original question.
            answer: The generated answer.

        Returns:
            A concise summary string.
        """
        prompt = SUMMARY_TEMPLATE.format(question=question, answer=answer)
        summary, _ = self._call_llm(prompt, max_tokens=256)
        return summary

    def _select_context(
        self,
        results: list[RetrievalResult],
        max_chunks: int,
    ) -> list[RetrievalResult]:
        """Intelligently select context chunks.

        Beyond simple top-k, this deduplicates near-identical content
        and ensures diversity across different source documents.
        """
        if len(results) <= max_chunks:
            return results

        selected: list[RetrievalResult] = []
        deferred: list[RetrievalResult] = []
        seen_docs: set[str] = set()
        seen_content_hashes: set[str] = set()

        for result in results:
            content_key = result.chunk.content[:100].lower().strip()
            if content_key in seen_content_hashes:
                continue
            seen_content_hashes.add(content_key)

            doc_id = result.chunk.document_id
            if doc_id not in seen_docs:
                # Prioritize chunks from new documents for diversity
                seen_docs.add(doc_id)
                selected.append(result)
            else:
                deferred.append(result)

            if len(selected) >= max_chunks:
                break

        # Fill remaining slots with deferred same-doc results
        for result in deferred:
            if len(selected) >= max_chunks:
                break
            selected.append(result)

        return selected

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string."""
        parts = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("title", "Unknown")
            heading = r.chunk.heading_context
            header = f"[Source {i}: {source}"
            if heading:
                header += f" > {heading}"
            header += f"] (relevance: {r.score:.3f})"
            parts.append(f"{header}\n{r.chunk.content}")
        return "\n\n---\n\n".join(parts)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_llm(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> tuple[str, dict[str, int]]:
        """Call the LLM API with retry logic."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        answer = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        return answer, usage
