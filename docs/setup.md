# Setup & Configuration Guide

> Step-by-step instructions to install, configure, and run the system.

---

## Prerequisites

- **Python 3.10+** (tested on 3.11–3.13)
- **pip** (bundled with Python)
- An **OpenAI API key** (for LLM reasoning; retrieval works without it)

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/self-improving-llm-kb.git
cd self-improving-llm-kb
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv

# Activate it:
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate

# Windows (cmd)
.venv\Scripts\activate.bat
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Embedding model (all-MiniLM-L6-v2) |
| `faiss-cpu` | Dense vector index |
| `rank-bm25` | BM25 sparse retrieval |
| `openai` | LLM API client |
| `tenacity` | Retry logic with exponential backoff |
| `pyyaml` | Configuration file parsing |
| `click` + `rich` | CLI framework |
| `streamlit` | Web UI |
| `mlflow` | Experiment tracking |
| `scikit-learn` | Utility functions |
| `pytest` + `pytest-cov` | Testing |

> **Tip:** If `sentence-transformers` installation is slow (it downloads PyTorch), the rest of the system still works for sparse-only retrieval and evaluation.

---

## 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set:

```env
# Required for LLM reasoning
OPENAI_API_KEY=sk-your-key-here

# Optional: MLflow tracking directory
MLFLOW_TRACKING_URI=mlruns
```

The system loads `.env` automatically via `python-dotenv`.

---

## 5. Configuration File

All system parameters are in `config/default.yaml`. You can override with a custom file:

```bash
python cli.py --config path/to/custom.yaml ingest
```

### Key Configuration Sections

#### Ingestion
```yaml
ingestion:
  chunk_strategy: "semantic"     # "semantic" or "fixed"
  chunk_max_tokens: 512          # Max tokens per chunk
  chunk_overlap_tokens: 64       # Overlap between consecutive chunks
  preserve_headings: true        # Prepend heading path to chunks
  knowledge_dir: "data/knowledge_base"
```

#### Retrieval
```yaml
retrieval:
  dense:
    model_name: "all-MiniLM-L6-v2"   # Sentence-Transformers model
  hybrid:
    dense_weight: 0.6                 # Weight for semantic scores
    sparse_weight: 0.4                # Weight for BM25 scores
    top_k: 5                          # Chunks returned per query
```

#### LLM
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"        # Or "gpt-4o", "gpt-3.5-turbo"
  temperature: 0.1            # Low = more deterministic
  max_tokens: 1024            # Max answer length
```

#### Memory
```yaml
memory:
  enabled: true
  max_history: 1000                # Max stored interactions
  deduplication_threshold: 0.85    # Jaccard similarity for dedup
  scoring:
    frequency_weight: 0.3
    recency_weight: 0.7
```

---

## 6. Verify Installation

```bash
# Check all imports work
python -c "import faiss, rank_bm25, yaml, openai, sklearn, streamlit; print('All dependencies OK')"

# Run the test suite
pytest tests/ -v

# Ingest the sample knowledge base
python cli.py ingest
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'faiss'`
Install with: `pip install faiss-cpu`

### `sentence-transformers` installation hangs
This package installs PyTorch (~2GB). Be patient, or install PyTorch separately first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

### `OPENAI_API_KEY` not found
Make sure your `.env` file is in the project root and contains `OPENAI_API_KEY=sk-...`.

### `RuntimeError: Knowledge base not indexed. Call ingest() first.`
Run `python cli.py ingest` before querying.

### Windows-specific: pip install stalls
Try installing packages in smaller batches:
```bash
pip install numpy pyyaml click rich
pip install faiss-cpu rank-bm25
pip install sentence-transformers
```

---

*Next: [Usage →](usage.md)*

