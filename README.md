# 🧠 Self-Improving LLM Knowledge Base

![Self-Improving LLM Knowledge Base Banner](./hero_banner.png)

[![CI](https://github.com/sadjad6/self-improving-llm-kb/actions/workflows/ci.yml/badge.svg)](https://github.com/sadjad6/self-improving-llm-kb/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sadjad6/self-improving-llm-kb/branch/main/graph/badge.svg)](https://codecov.io/gh/sadjad6/self-improving-llm-kb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **production-grade Retrieval-Augmented Generation (RAG) system** with hybrid retrieval, persistent memory, and a self-improving feedback loop — inspired by Andrej Karpathy's vision of context engineering and iterative learning systems.

> This project demonstrates senior-level ML system design: clean architecture, modular components, reproducible experiments, and production-ready patterns.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Hybrid Retrieval** | Combines FAISS dense vectors with BM25 sparse scores via Reciprocal Rank Fusion (RRF) |
| 📄 **Semantic Chunking** | Splits Markdown at natural boundaries (headings, paragraphs) while preserving heading context |
| 🤖 **Grounded LLM Reasoning** | OpenAI GPT with strict anti-hallucination prompting and citation grounding |
| 🧠 **Self-Improving Memory** | Persistent store that scores interactions, deduplicates queries, and writes learnings back to the KB |
| 📊 **Evaluation Framework** | Recall@K, MRR, heuristic scoring, and LLM-as-Judge with MLflow experiment tracking |
| 🖥️ **Dual Interface** | Polished Streamlit web UI + Rich CLI |
| 🛡️ **Graceful Degradation** | Soft-imported ML dependencies with clear error messages instead of crashes |

---

## 🏗️ Architecture

```
  Markdown Files ──▶ Parser ──▶ Semantic Chunker ──▶ Chunks
                                                       │
                                     ┌─────────────────┤
                                     ▼                  ▼
                               FAISS Dense          BM25 Sparse
                                 Index               Index
                                     │                  │
                                     └──────┬───────────┘
                                            ▼
                                    Hybrid Retriever
                                     (RRF Fusion)
                                            │
                                            ▼
                                    LLM Reasoning
                                  (grounded answer)
                                            │
                                            ▼
                                     Memory Store
                                (score → dedupe → prune)
                                            │
                                            ▼
                                     Summary Notes
                               (written back to knowledge base)
```

---

## 🚀 Quick Start

### 1. Clone & Set Up

```bash
git clone https://github.com/sadjad6/self-improving-llm-kb.git
cd self-improving-llm-kb
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

```bash
# Index the knowledge base
python cli.py ingest

# Ask a question
python cli.py ask "What is retrieval-augmented generation?"

# Launch the web UI
streamlit run app/streamlit_app.py
```

---

## 🖥️ Interfaces

### CLI (Click + Rich)

```bash
python cli.py ingest                                         # Index documents
python cli.py ask "How do transformers work?" --method hybrid # Query with method selection
python cli.py memory-stats                                   # View memory statistics
python cli.py evaluate --method hybrid                       # Run evaluation benchmark
```

### Streamlit Web UI

```bash
streamlit run app/streamlit_app.py
```

Features: query panel, retrieved-context viewer with scores, performance metrics, memory & self-improvement dashboard, and architecture overview.

---

## 📁 Project Structure

```
self-improving-llm-kb/
├── app/
│   └── streamlit_app.py          # Streamlit web dashboard
├── cli.py                        # CLI interface (Click + Rich)
├── config/
│   └── default.yaml              # All system configuration
├── data/
│   └── knowledge_base/           # Source Markdown files (5 ML topics)
├── docs/                         # Comprehensive documentation
│   ├── index.md                  # Project overview
│   ├── concepts.md               # LLM & RAG fundamentals
│   ├── architecture.md           # System design deep-dive
│   ├── setup.md                  # Installation guide
│   ├── usage.md                  # CLI & UI usage guide
│   ├── self_improving_loop.md    # Memory system explained
│   ├── evaluation.md             # Metrics & experiment tracking
│   └── api_reference.md          # Module API reference
├── src/
│   ├── ingestion/                # Markdown parser + semantic chunker
│   ├── retrieval/                # Dense (FAISS), sparse (BM25), hybrid
│   ├── llm/                      # LLM reasoning with context engineering
│   ├── memory/                   # Persistent self-improving memory store
│   ├── evaluation/               # Metrics (Recall@K, MRR) + MLflow tracker
│   ├── utils/                    # Config, data models, logging
│   └── pipeline.py               # Orchestration layer
├── tests/                        # Pytest test suite
├── requirements.txt
└── .env.example
```

---

## 🧠 How the Self-Improving Loop Works

```
User asks question → System answers (retrieve + LLM) → Interaction stored
        ↑                                                       │
        │                                               Importance scored
        │                                                       │
        │                                              Score ≥ 0.6?
        │                                              ╱         ╲
        │                                           Yes           No
        │                                            │          (wait)
        │                                   LLM generates summary
        │                                            │
        └──── Next ingest picks up ◄── Summary written as .md file
```

1. **Store** — Every Q&A interaction is persisted with an importance score
2. **Deduplicate** — Similar queries (Jaccard ≥ 0.85) update existing entries instead of duplicating
3. **Score** — Importance increases with query complexity and repeat access
4. **Summarize** — High-importance entries trigger LLM-generated summaries
5. **Feed back** — Summaries become new `.md` files, retrievable on next ingestion

→ [Full details in docs/self_improving_loop.md](docs/self_improving_loop.md)

---

## 📊 Evaluation

| Metric | Type | What It Measures |
|---|---|---|
| **Recall@K** | Retrieval | Fraction of relevant chunks in top-K results |
| **MRR** | Retrieval | Rank of the first relevant result |
| **Heuristic Score** | Answer | Length, grounding, query coverage, hallucination detection |
| **LLM-as-Judge** | Answer | Relevance, faithfulness, completeness (1–5 scale) |

All experiments are tracked via **MLflow** for full reproducibility.

→ [Full details in docs/evaluation.md](docs/evaluation.md)

---

## ⚙️ Configuration

All parameters live in [`config/default.yaml`](config/default.yaml):

```yaml
ingestion:
  chunk_strategy: "semantic"
  chunk_max_tokens: 512
  preserve_headings: true

retrieval:
  hybrid:
    dense_weight: 0.6        # Semantic similarity weight
    sparse_weight: 0.4       # BM25 keyword weight
    top_k: 5

llm:
  model: "gpt-4o-mini"
  temperature: 0.1           # Low = less hallucination

memory:
  enabled: true
  max_history: 1000
  deduplication_threshold: 0.85
```

Override with: `python cli.py --config path/to/custom.yaml ingest`

---

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

Test coverage includes: ingestion parsing/chunking, sparse retrieval, memory store operations, configuration loading, evaluation metrics, and end-to-end pipeline integration.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Dense Index | FAISS |
| Sparse Search | BM25 (`rank-bm25`) |
| LLM | OpenAI GPT-4o-mini |
| Experiment Tracking | MLflow |
| CLI | Click + Rich |
| Web UI | Streamlit |
| Testing | pytest + pytest-cov |
| Config | YAML + dataclasses |

---

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) folder:

| Document | Description |
|---|---|
| [Project Overview](docs/index.md) | Features, architecture, quick start |
| [Core Concepts](docs/concepts.md) | LLMs, RAG, embeddings, vector search explained |
| [Architecture](docs/architecture.md) | Module-by-module design breakdown |
| [Setup Guide](docs/setup.md) | Installation, configuration, troubleshooting |
| [Usage Guide](docs/usage.md) | CLI commands, Streamlit UI, example workflows |
| [Self-Improving Loop](docs/self_improving_loop.md) | Memory scoring, dedup, summary generation |
| [Evaluation](docs/evaluation.md) | Metrics, LLM-as-Judge, MLflow tracking |
| [API Reference](docs/api_reference.md) | All classes, methods, and data models |

---

## 🙏 Acknowledgments

- **Andrej Karpathy** — for the vision of persistent memory and self-improving AI systems
- **Sentence-Transformers** — for accessible, high-quality embedding models
- **FAISS** — for blazing-fast vector similarity search
- **OpenAI** — for the GPT API powering grounded generation

---

## 📄 License

This project is licensed under the MIT License.

