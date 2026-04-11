"""Main pipeline orchestrating all system components."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.ingestion.chunker import SemanticChunker
from src.ingestion.parser import MarkdownParser
from src.llm.reasoning import LLMReasoner
from src.memory.store import MemoryStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever
from src.utils.config import AppConfig, load_config
from src.utils.models import Chunk, QueryResult, RetrievalResult

try:
    from src.retrieval.reranker import CrossEncoderReranker
except ImportError:
    CrossEncoderReranker = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class KnowledgePipeline:
    """End-to-end pipeline for the self-improving knowledge base.

    Orchestrates: ingestion → retrieval → LLM reasoning → memory storage.
    Supports dense, sparse, and hybrid retrieval modes.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        experiment_tracker: Optional[object] = None,
    ) -> None:
        self.config = config or load_config()
        self._tracker = experiment_tracker

        # Components (lazy-initialized)
        self.parser = MarkdownParser(
            extract_links=self.config.ingestion.extract_links,
            extract_tags=self.config.ingestion.extract_tags,
        )
        self.chunker = SemanticChunker(
            max_tokens=self.config.ingestion.chunk_max_tokens,
            overlap_tokens=self.config.ingestion.chunk_overlap_tokens,
            preserve_headings=self.config.ingestion.preserve_headings,
        )
        self.dense_retriever = DenseRetriever(
            model_name=self.config.retrieval.dense.model_name,
            index_path=self.config.retrieval.dense.index_path,
        )
        self.sparse_retriever = SparseRetriever()

        # Build optional reranker
        reranker = None
        if (
            CrossEncoderReranker is not None
            and self.config.retrieval.reranker.enabled
        ):
            reranker = CrossEncoderReranker(
                model_name=self.config.retrieval.reranker.model_name,
                top_k=self.config.retrieval.reranker.top_k,
            )
            logger.info(
                "Reranker enabled: %s", self.config.retrieval.reranker.model_name
            )

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            dense_weight=self.config.retrieval.hybrid.dense_weight,
            sparse_weight=self.config.retrieval.hybrid.sparse_weight,
            reranker=reranker,
        )
        self.reasoner = LLMReasoner(config=self.config.llm)
        self.memory = MemoryStore(config=self.config.memory)

        self._chunks: list[Chunk] = []
        self._indexed = False

    def ingest(self, knowledge_dir: Optional[str] = None) -> int:
        """Ingest and index the knowledge base.

        Args:
            knowledge_dir: Path to knowledge base directory.

        Returns:
            Number of chunks created.
        """
        kb_path = Path(knowledge_dir or self.config.ingestion.knowledge_dir)
        logger.info("Ingesting knowledge base from: %s", kb_path)

        documents = self.parser.parse_directory(kb_path)
        if not documents:
            logger.warning("No documents found in %s", kb_path)
            return 0

        self._chunks = self.chunker.chunk_documents(documents)
        self.hybrid_retriever.index(self._chunks)
        self._indexed = True

        logger.info(
            "Ingestion complete: %d documents → %d chunks",
            len(documents), len(self._chunks),
        )
        return len(self._chunks)

    def query(
        self,
        question: str,
        method: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """Answer a question using the knowledge base.

        Args:
            question: The user's question.
            method: Retrieval method ("dense", "sparse", "hybrid").
            top_k: Number of chunks to retrieve.

        Returns:
            Complete QueryResult with answer and metadata.
        """
        if not self._indexed:
            raise RuntimeError("Knowledge base not indexed. Call ingest() first.")

        start = time.time()
        k = top_k or self.config.retrieval.hybrid.top_k

        # Retrieve relevant chunks
        results = self.hybrid_retriever.retrieve(question, top_k=k, method=method)

        # Generate answer
        answer, usage = self.reasoner.generate_answer(question, results)
        latency = (time.time() - start) * 1000

        query_result = QueryResult(
            query=question,
            answer=answer,
            retrieved_chunks=results,
            retrieval_method=method,
            latency_ms=latency,
            token_usage=usage,
        )

        # Store in memory (self-improving loop)
        if self.config.memory.enabled:
            entry = self.memory.store_interaction(query_result)
            # Generate and persist summary for high-importance entries
            if entry.importance_score >= 0.6:
                try:
                    summary = self.reasoner.generate_summary(question, answer)
                    self.memory.generate_summary_note(entry, summary)
                except Exception as e:
                    logger.warning("Failed to generate summary: %s", e)

        logger.info(
            "Query answered in %.0fms (method=%s, chunks=%d, tokens=%d)",
            latency, method, len(results), usage.get("total_tokens", 0),
        )

        # Log to experiment tracker if one is attached
        if self._tracker is not None:
            try:
                self._tracker.log_metrics({  # type: ignore[union-attr]
                    "latency_ms": latency,
                    "chunks_retrieved": float(len(results)),
                    "total_tokens": float(usage.get("total_tokens", 0)),
                })
            except Exception as exc:
                logger.warning("Failed to log metrics to tracker: %s", exc)

        return query_result

    def retrieve_only(
        self,
        question: str,
        method: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks without LLM generation.

        Useful for evaluation and debugging.
        """
        if not self._indexed:
            raise RuntimeError("Knowledge base not indexed. Call ingest() first.")
        k = top_k or self.config.retrieval.hybrid.top_k
        return self.hybrid_retriever.retrieve(question, top_k=k, method=method)

