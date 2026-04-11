"""Dense retrieval using sentence embeddings and FAISS."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.models import Chunk, RetrievalResult

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retrieval using bi-encoder embeddings and FAISS index.

    Encodes documents and queries into dense vectors using a sentence
    transformer model, then performs approximate nearest neighbor search
    via FAISS for efficient similarity retrieval.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "data/indices/faiss.index",
    ) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.chunks_path = self.index_path.with_suffix(".chunks.pkl")
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list[Chunk] = []

    @property
    def model(self) -> SentenceTransformer:  # type: ignore[return-value]
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, chunks: list[Chunk]) -> None:
        """Build a FAISS index from document chunks.

        Args:
            chunks: List of chunks to index.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info("Encoding %d chunks for dense index...", len(chunks))
        texts = [c.content for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)
        self._index.add(embeddings)
        self._chunks = list(chunks)

        logger.info("FAISS index built: %d vectors, dim=%d", self._index.ntotal, dim)
        self._save_index()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve the most similar chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        if self._index is None:
            self._load_index()
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Dense index is empty.")
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    method="dense",
                )
            )
        return results

    def _save_index(self) -> None:
        """Persist FAISS index and chunk mapping to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("Dense index saved to %s", self.index_path)

    def _load_index(self) -> None:
        """Load FAISS index and chunk mapping from disk."""
        if not self.index_path.exists():
            logger.warning("No saved dense index found at %s", self.index_path)
            return
        self._index = faiss.read_index(str(self.index_path))
        with open(self.chunks_path, "rb") as f:
            self._chunks = pickle.load(f)
        logger.info("Dense index loaded: %d vectors", self._index.ntotal)
