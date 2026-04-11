# 🧠 Self-Improving LLM Knowledge Base

> A production-grade, end-to-end Retrieval-Augmented Generation (RAG) system with persistent memory, hybrid retrieval, and a self-improving feedback loop — inspired by Andrej Karpathy's vision of context engineering and iterative learning.

---

## What Is This Project?

This system ingests a Markdown knowledge base, indexes it using **hybrid retrieval** (dense semantic search + sparse BM25 keyword matching), and answers questions by grounding an LLM's responses in the retrieved context. Every interaction is stored in a persistent memory layer that scores, deduplicates, and prunes knowledge over time — making the system **learn from its own usage**.

It is designed to demonstrate senior-level ML engineering skills: clean architecture, modular design, reproducible experiments, and production-ready patterns.

---

## Key Features

| Feature | Description |
|---|---|
| **Semantic Chunking** | Splits Markdown at natural boundaries (headings, paragraphs) and preserves heading context in every chunk |
| **Hybrid Retrieval** | Combines FAISS dense vectors with BM25 sparse scores via Reciprocal Rank Fusion (RRF) |
| **LLM Reasoning** | Grounded answer generation using OpenAI GPT models with strict anti-hallucination prompting |
| **Self-Improving Memory** | Persistent JSONL store that scores interactions, deduplicates queries, generates summary notes, and feeds learnings back into the knowledge base |
| **Evaluation Framework** | Recall@K, MRR, heuristic scoring, and LLM-as-Judge evaluation with MLflow experiment tracking |
| **CLI + Web UI** | Rich terminal interface via Click and a polished Streamlit dashboard |
| **Graceful Degradation** | All heavy ML dependencies are soft-imported — the system provides clear error messages instead of crashing |

---

## Architecture at a Glance

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

## Project Structure

```
self-improving-llm-kb/
├── app/
│   └── streamlit_app.py          # Web UI (Streamlit dashboard)
├── cli.py                        # Command-line interface (Click + Rich)
├── config/
│   └── default.yaml              # All system configuration
├── data/
│   └── knowledge_base/           # Source Markdown files
├── docs/                         # ← You are here
│   ├── index.md                  # Project overview (this file)
│   ├── concepts.md               # LLM & RAG fundamentals
│   ├── architecture.md           # System design deep-dive
│   ├── setup.md                  # Installation & configuration
│   ├── usage.md                  # CLI & Streamlit usage guide
│   ├── self_improving_loop.md    # Memory system explained
│   ├── evaluation.md             # Metrics & experiment tracking
│   └── api_reference.md          # Module-level API reference
├── src/
│   ├── ingestion/                # Parsing & chunking
│   ├── retrieval/                # Dense, sparse, hybrid retrievers
│   ├── llm/                      # LLM reasoning layer
│   ├── memory/                   # Persistent memory store
│   ├── evaluation/               # Metrics & MLflow tracker
│   ├── utils/                    # Config, models, logging
│   └── pipeline.py               # Orchestration layer
├── tests/                        # Pytest test suite
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment variable template
```

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Ingest the knowledge base
python cli.py ingest

# 5. Ask a question
python cli.py ask "What is retrieval-augmented generation?"

# 6. Launch the web UI
streamlit run app/streamlit_app.py
```

---

## Documentation Guide

| Document | What You'll Learn |
|---|---|
| [Concepts](concepts.md) | What are LLMs, RAG, embeddings, and vector search? |
| [Architecture](architecture.md) | How every module connects and why each design decision was made |
| [Setup](setup.md) | Step-by-step installation, configuration, and troubleshooting |
| [Usage](usage.md) | How to use the CLI commands and Streamlit UI |
| [Self-Improving Loop](self_improving_loop.md) | How the memory system scores, deduplicates, and feeds back knowledge |
| [Evaluation](evaluation.md) | Recall@K, MRR, LLM-as-Judge, and MLflow experiment tracking |
| [API Reference](api_reference.md) | Classes, methods, and data models for every module |

---

## Who Is This For?

- **Recruiters & hiring managers** evaluating ML engineering portfolios
- **ML engineers** looking for a clean RAG reference implementation
- **Students** learning about retrieval systems, context engineering, and self-improving AI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Dense index | FAISS |
| Sparse search | BM25 (rank-bm25) |
| LLM | OpenAI GPT-4o-mini |
| Experiment tracking | MLflow |
| CLI | Click + Rich |
| Web UI | Streamlit |
| Testing | pytest + pytest-cov |

