# API Reference

> Key classes, methods, and data models across all modules.

---

## Data Models (`src/utils/models.py`)

### `Document`
```python
@dataclass
class Document:
    id: str                          # MD5 hash of file path
    title: str                       # First H1 heading or filename
    content: str                     # Full document text
    source_path: str                 # Original file path
    headings: list[str]              # All headings found
    links: list[str]                 # Wiki-links extracted
    tags: list[str]                  # Hashtags extracted
    metadata: dict[str, Any]         # Additional metadata
```

### `Chunk`
```python
@dataclass
class Chunk:
    id: str                          # Unique chunk identifier
    document_id: str                 # Parent document ID
    content: str                     # Chunk text
    heading_context: str             # Heading path (e.g., "ML > Training > SGD")
    index: int                       # Position within document
    token_count: int                 # Approximate token count
    metadata: dict[str, Any]         # Additional metadata
```

### `RetrievalResult`
```python
@dataclass
class RetrievalResult:
    chunk: Chunk                     # The retrieved chunk
    score: float                     # Relevance score (0–1)
    method: str                      # "dense", "sparse", or "hybrid"
```

### `QueryResult`
```python
@dataclass
class QueryResult:
    query: str                       # Original question
    answer: str                      # LLM-generated answer
    retrieved_chunks: list[RetrievalResult]
    retrieval_method: str            # Method used
    latency_ms: float                # End-to-end latency
    token_usage: dict[str, int]      # {"prompt_tokens": N, "completion_tokens": N, ...}
    timestamp: str                   # ISO 8601 UTC timestamp
```

### `MemoryEntry`
```python
@dataclass
class MemoryEntry:
    id: str                          # MD5 hash of query + timestamp
    query: str                       # Original question
    answer: str                      # Generated answer
    retrieved_context: list[str]     # Context chunk texts
    timestamp: str                   # ISO 8601 UTC timestamp
    importance_score: float          # 0.0–1.0
    access_count: int                # Times this entry was accessed
    summary: str                     # LLM-generated summary (if any)
```

---

## Ingestion (`src/ingestion/`)

### `MarkdownParser`
```python
class MarkdownParser:
    def __init__(self, extract_links: bool = True, extract_tags: bool = True)
    def parse_file(self, filepath: Path) -> Document
    def parse_directory(self, directory: Path) -> list[Document]
```

### `SemanticChunker`
```python
class SemanticChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64, preserve_headings: bool = True)
    def chunk_documents(self, documents: list[Document]) -> list[Chunk]
    def chunk_document(self, document: Document) -> list[Chunk]
```

---

## Retrieval (`src/retrieval/`)

### `DenseRetriever`
```python
class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "...")
    def index(self, chunks: list[Chunk]) -> None
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]
    def save_index(self) -> None
    def load_index(self) -> bool
```

### `SparseRetriever`
```python
class SparseRetriever:
    def __init__(self)
    def index(self, chunks: list[Chunk]) -> None
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]
```

### `HybridRetriever`
```python
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, dense_weight=0.6, sparse_weight=0.4)
    def index(self, chunks: list[Chunk]) -> None
    def retrieve(self, query: str, top_k: int = 5, method: str = "hybrid") -> list[RetrievalResult]
```

The `method` parameter controls which retriever is used:
- `"dense"` — Only DenseRetriever
- `"sparse"` — Only SparseRetriever
- `"hybrid"` — Both, fused with RRF + weighted scoring

---

## LLM Reasoning (`src/llm/`)

### `LLMReasoner`
```python
class LLMReasoner:
    def __init__(self, config: LLMConfig = None)
    def generate_answer(self, query: str, retrieval_results: list[RetrievalResult],
                        max_context_chunks: int = 5) -> tuple[str, dict[str, int]]
    def generate_summary(self, question: str, answer: str) -> str
```

- `generate_answer()` returns `(answer_text, token_usage_dict)`
- `generate_summary()` returns a concise knowledge summary for the memory system

---

## Memory (`src/memory/`)

### `MemoryStore`
```python
class MemoryStore:
    def __init__(self, config: MemoryConfig = None)
    def store_interaction(self, result: QueryResult) -> MemoryEntry
    def get_relevant_history(self, query: str, top_k: int = 3) -> list[MemoryEntry]
    def generate_summary_note(self, entry: MemoryEntry, summary: str) -> Path
    def get_stats(self) -> dict          # {total_entries, avg_importance, total_accesses}
    def get_recent(self, n: int = 5) -> list[MemoryEntry]
```

---

## Evaluation (`src/evaluation/`)

### `RetrievalEvaluator`
```python
class RetrievalEvaluator:
    @staticmethod
    def recall_at_k(retrieved, relevant_ids: set[str], k: int) -> EvalResult
    @staticmethod
    def mrr(retrieved, relevant_ids: set[str]) -> EvalResult
    def evaluate(self, retrieved, relevant_ids, k_values=[1,3,5,10]) -> list[EvalResult]
```

### `AnswerEvaluator`
```python
class AnswerEvaluator:
    @staticmethod
    def heuristic_score(answer: str, context_texts: list[str], query: str) -> EvalResult
    @staticmethod
    def llm_judge_prompt(query: str, answer: str, context: str) -> str
```

### `ExperimentTracker`
```python
class ExperimentTracker:
    def __init__(self, config: ExperimentConfig = None)
    def start_run(self, run_name: str = None) -> str   # Returns run ID
    def log_params(self, params: dict[str, Any]) -> None
    def log_metrics(self, metrics: dict[str, float]) -> None
    def end_run(self) -> None
```

---

## Pipeline (`src/pipeline.py`)

### `KnowledgePipeline`
```python
class KnowledgePipeline:
    def __init__(self, config: AppConfig = None)
    def ingest(self, knowledge_dir: str = None) -> int          # Returns chunk count
    def query(self, question: str, method="hybrid", top_k=None) -> QueryResult
    def retrieve_only(self, question: str, method="hybrid", top_k=None) -> list[RetrievalResult]
```

---

*Back to: [Index →](index.md)*

