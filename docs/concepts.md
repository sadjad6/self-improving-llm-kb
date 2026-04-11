# Core Concepts: LLMs, Knowledge Bases, and RAG

> This guide introduces the fundamental ideas behind this project. If you already know what RAG is, skip to [Architecture](architecture.md).

---

## 1. What Is a Large Language Model (LLM)?

A Large Language Model is a neural network trained on massive text corpora to predict the next token in a sequence. Models like GPT-4, Claude, and LLaMA have billions of parameters and can generate fluent, contextually appropriate text.

**Key limitations of LLMs:**

- **Knowledge cutoff** — They only know what was in their training data. They can't answer questions about events or documents published after training.
- **Hallucination** — They sometimes generate plausible-sounding but factually incorrect statements, especially when asked about niche or specialized topics.
- **No persistent memory** — Each conversation starts from scratch. The model doesn't remember previous interactions unless explicitly provided as context.

These limitations are exactly what this project addresses.

---

## 2. What Is a Knowledge Base?

A knowledge base is a structured or semi-structured collection of information that a system can query. In this project, the knowledge base consists of **Markdown files** covering ML topics (neural networks, transformers, evaluation metrics, etc.).

Unlike training data baked into model weights, a knowledge base is:

- **External** — Lives outside the model and can be updated independently
- **Auditable** — You can inspect exactly what information the system has access to
- **Extensible** — Add a new `.md` file and re-ingest; no retraining needed

---

## 3. What Is RAG (Retrieval-Augmented Generation)?

RAG is a pattern that combines **information retrieval** with **language model generation**:

```
User Question
     │
     ▼
 ┌────────────┐     ┌──────────────┐     ┌─────────────┐
 │  Retriever  │────▶│  Top-K Chunks │────▶│     LLM      │
 └────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                          Grounded Answer
```

1. **Retrieve** — Search the knowledge base for the most relevant passages
2. **Augment** — Inject those passages into the LLM's prompt as context
3. **Generate** — The LLM produces an answer grounded in the retrieved context

**Why RAG instead of fine-tuning?**

| Aspect | Fine-Tuning | RAG |
|---|---|---|
| Knowledge updates | Requires retraining | Just update documents |
| Cost | Expensive GPU hours | Only inference cost |
| Traceability | Opaque (in weights) | Transparent (cited sources) |
| Hallucination control | Hard | Easier (constrained context) |

---

## 4. Embeddings and Vector Search

### What Are Embeddings?

An embedding is a dense numerical vector (e.g., 384 dimensions) that captures the *meaning* of a piece of text. Similar texts have vectors that are close together in this high-dimensional space.

```
"What is gradient descent?"  →  [0.12, -0.34, 0.56, ...]  (384 dims)
"How does gradient descent work?"  →  [0.13, -0.33, 0.55, ...]  (very close!)
"Best pizza in New York"  →  [-0.78, 0.91, -0.12, ...]  (far away)
```

This project uses `all-MiniLM-L6-v2` from Sentence-Transformers to generate embeddings.

### What Is FAISS?

FAISS (Facebook AI Similarity Search) is a library for efficient nearest-neighbor search over dense vectors. Given a query embedding, FAISS finds the K most similar chunk embeddings in milliseconds, even with millions of vectors.

---

## 5. Sparse vs. Dense Retrieval

### Dense Retrieval (Semantic)
- Converts text to embedding vectors
- Finds chunks with similar *meaning*
- Good at understanding paraphrases and synonyms
- Example: "neural network training" matches "how to optimize model weights"

### Sparse Retrieval (Keyword — BM25)
- Counts term frequencies and inverse document frequencies
- Finds chunks with matching *words*
- Good at exact keyword matching and rare terms
- Example: "LSTM" matches documents containing "LSTM" exactly

### Hybrid Retrieval (This Project)
Combines both using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(chunk) = 1/(k + rank_dense) + 1/(k + rank_sparse)
```

Then applies configurable weights (default: 60% dense, 40% sparse) to get the final score. This captures both semantic understanding and keyword precision.

---

## 6. Semantic Chunking

Before retrieval, documents must be split into smaller pieces (chunks). Naive approaches split at fixed character counts, which can break sentences mid-word or separate a heading from its content.

**Semantic chunking** (used in this project) splits at natural boundaries:

- Headings (`#`, `##`, `###`)
- Paragraph breaks (double newlines)
- Respects a maximum token limit (default: 512 tokens)

Each chunk also carries its **heading context** — the path of headings above it — so a chunk under `## Training > ### Backpropagation` retains that hierarchy even when read in isolation.

---

## 7. Context Engineering

A concept emphasized by Andrej Karpathy: the art of carefully selecting *what context* to put in the LLM's prompt window. Not all retrieved chunks are equally useful. This project implements context engineering through:

- **Score-based ranking** — Only the top-K highest-scoring chunks are included
- **Diversity selection** — Preferring chunks from different documents to avoid redundancy
- **Token budget management** — Ensuring the context fits within the model's token limit
- **Anti-hallucination prompting** — Explicit instructions to answer only from context

---

## 8. Self-Improving Systems

The most novel aspect of this project. Traditional RAG is stateless — ask a question, get an answer, forget everything. This system **remembers**:

1. Every Q&A interaction is stored with an importance score
2. Repeated similar questions boost the importance of that knowledge
3. High-importance interactions trigger automatic summary generation
4. Summaries are written back to the knowledge base as new Markdown files
5. On the next ingestion, these summaries become retrievable knowledge

This creates a **positive feedback loop**: the more the system is used, the better its knowledge base becomes.

→ For a deep dive, see [Self-Improving Loop](self_improving_loop.md).

---

## 9. Evaluation: How Do We Know It Works?

Two categories of metrics:

### Retrieval Metrics
- **Recall@K** — What fraction of the relevant chunks appear in the top K results?
- **MRR (Mean Reciprocal Rank)** — How high is the first relevant result?

### Answer Quality Metrics
- **Heuristic scoring** — Automated checks for length, grounding in context, query coverage
- **LLM-as-Judge** — A separate LLM rates the answer on relevance, faithfulness, and completeness (1–5 scale)

→ For details, see [Evaluation](evaluation.md).

---

## Glossary

| Term | Definition |
|---|---|
| **Chunk** | A segment of text extracted from a document, the basic unit of retrieval |
| **Dense retrieval** | Finding similar text by comparing embedding vectors |
| **Embedding** | A numerical vector representation of text meaning |
| **FAISS** | Facebook's library for fast nearest-neighbor search on vectors |
| **BM25** | A classic keyword-matching algorithm based on term frequency |
| **RRF** | Reciprocal Rank Fusion — a method to merge ranked lists from multiple retrievers |
| **RAG** | Retrieval-Augmented Generation — grounding LLM answers in retrieved evidence |
| **Top-K** | The K highest-scoring results returned by a retriever |
| **Token** | The smallest unit of text processed by an LLM (roughly ¾ of a word) |
| **Hallucination** | When an LLM generates factually incorrect information not in its context |

---

*Next: [Architecture →](architecture.md)*

