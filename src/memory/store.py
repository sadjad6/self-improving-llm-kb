"""Persistent memory store for the self-improving knowledge system."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from src.utils.config import MemoryConfig
from src.utils.models import MemoryEntry, QueryResult

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persistent memory system that stores interactions and generates summaries.

    Implements the self-improving loop:
    1. Store each Q&A interaction with context
    2. Score entries by importance and frequency
    3. Generate summary notes for high-value interactions
    4. Deduplicate similar entries
    5. Persist learnings back to the knowledge base
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self.store_path = Path(self.config.store_path)
        self.summary_path = Path(self.config.summary_path)
        self._entries: list[MemoryEntry] = []
        self._load()

    def store_interaction(self, result: QueryResult) -> MemoryEntry:
        """Store a query-answer interaction in memory.

        Args:
            result: The complete query result to store.

        Returns:
            The created MemoryEntry.
        """
        entry_id = hashlib.md5(
            f"{result.query}:{result.timestamp}".encode()
        ).hexdigest()[:12]

        context_texts = [r.chunk.content for r in result.retrieved_chunks]

        # Check for deduplication
        existing = self._find_similar(result.query)
        if existing:
            existing.access_count += 1
            existing.importance_score = min(1.0, existing.importance_score + 0.1)
            logger.info("Updated existing memory entry: %s", existing.id)
            self._save()
            return existing

        entry = MemoryEntry(
            id=entry_id,
            query=result.query,
            answer=result.answer,
            retrieved_context=context_texts,
            timestamp=result.timestamp,
            importance_score=self._compute_importance(result),
        )
        self._entries.append(entry)

        # Enforce max history
        if len(self._entries) > self.config.max_history:
            self._prune()

        self._save()
        logger.info("Stored new memory entry: %s (importance=%.2f)", entry.id, entry.importance_score)
        return entry

    def get_relevant_history(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        """Find past interactions relevant to a query.

        Args:
            query: The current query.
            top_k: Number of entries to return.

        Returns:
            Most relevant past interactions.
        """
        if not self._entries:
            return []

        query_terms = set(query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []
        now = datetime.now(UTC)

        for entry in self._entries:
            entry_terms = set(entry.query.lower().split())
            overlap = len(query_terms & entry_terms)
            if overlap == 0:
                continue
            # Score based on term overlap, importance, and actual recency
            term_score = overlap / max(len(query_terms), 1)
            recency_score = self._compute_recency(entry.timestamp, now)
            combined = (
                term_score * 0.5
                + entry.importance_score * self.config.frequency_weight
                + recency_score * self.config.recency_weight
            )
            scored.append((combined, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def generate_summary_note(self, entry: MemoryEntry, summary: str) -> Path:
        """Persist a summary note to the knowledge base.

        Args:
            entry: The memory entry to summarize.
            summary: Pre-generated summary text.

        Returns:
            Path to the created summary file.
        """
        self.summary_path.mkdir(parents=True, exist_ok=True)
        filename = f"summary_{entry.id}.md"
        filepath = self.summary_path / filename

        content = (
            f"# Auto-Generated Summary\n\n"
            f"#auto-generated #memory\n\n"
            f"## Original Question\n{entry.query}\n\n"
            f"## Key Insights\n{summary}\n\n"
            f"## Generated\n{entry.timestamp}\n"
        )
        filepath.write_text(content, encoding="utf-8")
        entry.summary = summary
        self._save()

        logger.info("Generated summary note: %s", filepath)
        return filepath

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        avg_importance = (
            sum(e.importance_score for e in self._entries) / len(self._entries)
            if self._entries else 0.0
        )
        return {
            "total_entries": len(self._entries),
            "avg_importance": avg_importance,
            "average_score": avg_importance,  # alias kept for UI compatibility
            "total_accesses": sum(e.access_count for e in self._entries),
        }

    def get_recent(self, n: int = 5) -> list[MemoryEntry]:
        """Return the *n* most recent memory entries."""
        sorted_entries = sorted(
            self._entries,
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_entries[:n]

    def _compute_importance(self, result: QueryResult) -> float:
        """Compute importance score for a new interaction."""
        score = 0.5  # baseline
        # Boost if many chunks were retrieved (complex query)
        if len(result.retrieved_chunks) >= 3:
            score += 0.1
        # Boost if answer is substantial
        if len(result.answer) > 200:
            score += 0.1
        return min(1.0, score)

    def _find_similar(self, query: str) -> Optional[MemoryEntry]:
        """Find a similar existing entry for deduplication."""
        query_lower = query.lower().strip()
        query_terms = set(query_lower.split())

        for entry in self._entries:
            entry_terms = set(entry.query.lower().strip().split())
            if not query_terms or not entry_terms:
                continue
            overlap = len(query_terms & entry_terms) / max(len(query_terms | entry_terms), 1)
            if overlap >= self.config.deduplication_threshold:
                return entry
        return None

    def _prune(self) -> None:
        """Remove lowest-scoring entries to stay within max_history."""
        self._entries.sort(key=lambda e: e.importance_score, reverse=True)
        removed = len(self._entries) - self.config.max_history
        self._entries = self._entries[: self.config.max_history]
        logger.info("Pruned %d low-importance memory entries.", removed)

    @staticmethod
    def _compute_recency(timestamp: str, now: datetime) -> float:
        """Compute a 0-1 recency score; newer entries score higher.

        Uses a half-life of 7 days so entries decay to ~0.5 after one week.
        """
        try:
            # Parse both naive (legacy) and aware timestamps
            ts = datetime.fromisoformat(timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            age_days = (now - ts).total_seconds() / 86_400
            # Exponential decay: score = 2^(-age/7)
            return 2 ** (-age_days / 7)
        except (ValueError, TypeError):
            return 0.5  # Neutral score on parse failure

    def _save(self) -> None:
        """Persist entries to disk using dataclasses.asdict for safety."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(dataclasses.asdict(entry)) + "\n")

    def _load(self) -> None:
        """Load entries from disk."""
        if not self.store_path.exists():
            return
        self._entries = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self._entries.append(MemoryEntry(**data))
        logger.info("Loaded %d memory entries.", len(self._entries))

