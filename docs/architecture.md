# System Architecture

> A module-by-module breakdown of the Self-Improving LLM Knowledge Base, covering data flow, design decisions, and component interactions.

---

## High-Level Data Flow

```
                          ┌──────────────────────────────────────────────┐
                          │            Knowledge Base (Markdown)         │
                          └──────────────────┬───────────────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  MarkdownParser  │
                                    │  (parser.py)     │
                                    └────────┬────────┘
                                             │  List[Document]
                                    ┌────────▼────────┐
                                    │ SemanticChunker  │
                                    │  (chunker.py)    │
                                    └────────┬────────┘
                                             │  List[Chunk]
                              ┌──────────────┼──────────────┐
                              ▼              ▼              │
                       DenseRetriever  SparseRetriever      │
                        (FAISS)         (BM25)              │
                              │              │              │
                              └──────┬───────┘              │
                                     ▼                      │
                             HybridRetriever                │
                           (RRF + weighted)                 │
                                     │                      │
                                     │  List[RetrievalResult]
                                     ▼                      │
                              LLMReasoner                   │
                            (OpenAI GPT)                    │
                                     │                      │
                                     │  QueryResult         │
                                     ▼                      │
                              MemoryStore ──────────────────┘
                          (score → dedupe → prune)    (summary notes
                                                       fed back)
```

---

## Module Breakdown

### 1. Ingestion Layer (`src/ingestion/`)

#### `parser.py` — MarkdownParser

Parses `.md` files into structured `Document` objects. Extracts:

- **Headings** — Builds a hierarchical heading tree
- **Links** — Captures `[text](url)` references (configurable)
- **Tags** — Extracts `#hashtags` from content (configurable)
- **Metadata** — Title derived from first `H1` or filename

**Design decision:** Markdown was chosen as the knowledge format because it's human-readable, version-controllable, and structurally rich (headings provide natural semantic boundaries).

#### `chunker.py` — SemanticChunker

Splits documents into retrieval-ready chunks with context preservation.

**Algorithm:**
1. Split document at heading boundaries (`#`, `##`, etc.)
2. For each section, further split at paragraph breaks if it exceeds `max_tokens` (default: 512)
3. Prepend the full heading path (e.g., `"Neural Networks > Training > Backpropagation"`) to each chunk
4. Apply token-based overlap (`overlap_tokens`: 64) between consecutive chunks for continuity

**Design decision:** Semantic chunking over fixed-size chunking prevents splitting mid-sentence and preserves the document's logical structure.

---

### 2. Retrieval Layer (`src/retrieval/`)

#### `dense.py` — DenseRetriever

- Encodes chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- Builds and queries a FAISS `IndexFlatIP` (inner product / cosine similarity)
- Supports saving/loading the FAISS index to disk for persistence

#### `sparse.py` — SparseRetriever

- Tokenizes chunks using simple whitespace + lowercasing
- Builds a BM25 index using the `rank-bm25` library
- Returns scores based on term frequency × inverse document frequency

#### `hybrid.py` — HybridRetriever

Combines dense and sparse results using two fusion strategies:

1. **Reciprocal Rank Fusion (RRF):**
   ```
   RRF(d) = Σ  1 / (k + rank_i(d))
   ```
   where `k = 60` (standard constant) and `rank_i(d)` is the rank of document `d` in retriever `i`.

2. **Weighted score combination:**
   ```
   final_score = dense_weight × dense_score + sparse_weight × sparse_score
   ```
   Default weights: 60% dense, 40% sparse (configurable in `config/default.yaml`).

**Design decision:** Hybrid retrieval captures both semantic similarity (dense) and exact keyword matching (sparse). Research consistently shows hybrid outperforms either method alone.

---

### 3. LLM Reasoning Layer (`src/llm/`)

#### `reasoning.py` — LLMReasoner

- Calls OpenAI's API (default: `gpt-4o-mini`) with retrieved context
- Uses a carefully engineered **answer prompt** that enforces grounding:
  - "Answer ONLY based on the provided context"
  - "If the context doesn't contain enough information, say so"
  - "Cite specific details from the context"
- Includes a **summary prompt** for the self-improving loop
- Implements retry logic via `tenacity` (exponential backoff, 3 attempts)
- Context selection: deduplicates by document and enforces a token budget

**Design decision:** Temperature is set to 0.1 (near-deterministic) to minimize hallucination in grounded Q&A.

---

### 4. Memory Layer (`src/memory/`)

#### `store.py` — MemoryStore

Persistent JSON Lines (`.jsonl`) store implementing the self-improving loop:

- **Store** — Each interaction creates a `MemoryEntry` with an importance score
- **Deduplicate** — Similar queries (Jaccard overlap ≥ 0.85) update the existing entry instead of creating a duplicate
- **Score** — Importance increases with query complexity (chunk count, answer length) and repeat access
- **Prune** — When entries exceed `max_history` (default: 1000), lowest-importance entries are removed
- **Summarize** — High-importance entries (score ≥ 0.6) trigger LLM-generated summaries
- **Persist** — Summaries are written as new `.md` files back into the knowledge base

→ Full details: [Self-Improving Loop](self_improving_loop.md)

---

### 5. Evaluation Layer (`src/evaluation/`)

#### `metrics.py` — RetrievalEvaluator & AnswerEvaluator

- **Recall@K** and **MRR** for retrieval quality
- **Heuristic scoring** for answer quality (length, grounding, query coverage, hallucination detection)
- **LLM-as-Judge prompt** for human-like evaluation on relevance, faithfulness, completeness

#### `tracker.py` — ExperimentTracker

MLflow-based experiment tracking for reproducibility. Logs parameters, metrics, and artifacts per run.

---

### 6. Orchestration (`src/pipeline.py`)

`KnowledgePipeline` ties everything together:

```python
pipeline = KnowledgePipeline(config)
pipeline.ingest()                           # Parse → Chunk → Index
result = pipeline.query("What is RAG?")     # Retrieve → Generate → Store
chunks = pipeline.retrieve_only("RAG")      # Retrieve without LLM (for eval)
```

**Lazy initialization:** All components are created in `__init__` but don't load models until first use, keeping startup fast.

---

### 7. Configuration (`src/utils/`)

#### `config.py` — Dataclass-based config

All settings live in `config/default.yaml` and are loaded into typed Python dataclasses:

```
AppConfig
├── IngestionConfig     (chunk sizes, knowledge dir)
├── RetrievalConfig
│   ├── DenseConfig     (model name, index path)
│   ├── SparseConfig    (algorithm)
│   └── HybridConfig    (weights, top_k)
├── LLMConfig           (provider, model, temperature)
├── MemoryConfig        (paths, thresholds, scoring)
├── EvaluationConfig    (metrics, k_values)
└── ExperimentConfig    (MLflow URI, experiment name)
```

#### `models.py` — Core data models

Five dataclasses used across all modules: `Document`, `Chunk`, `RetrievalResult`, `QueryResult`, `MemoryEntry`.

---

## Design Principles

1. **Modularity** — Each component (parser, chunker, retriever, reasoner, memory) is independently testable and replaceable
2. **Graceful degradation** — Heavy ML dependencies (`faiss`, `sentence-transformers`, `openai`) are soft-imported; the system provides clear error messages instead of crashing
3. **Configuration-driven** — All tunable parameters live in YAML, not hardcoded in source
4. **Observability** — Structured logging, experiment tracking, and per-query metadata (latency, token usage, retrieval method)
5. **Reproducibility** — MLflow tracks every experiment run with full parameter sets

---

*Next: [Setup →](setup.md)*

