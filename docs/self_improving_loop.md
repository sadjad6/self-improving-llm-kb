# The Self-Improving Memory Loop

> How the system learns from its own usage and writes knowledge back into the knowledge base.

---

## Why Self-Improvement?

Traditional RAG systems are **stateless**: they retrieve, generate, and forget. Every query starts from scratch. This project implements a **persistent memory layer** that creates a positive feedback loop:

```
 User asks a question
        │
        ▼
 System answers (retrieve + LLM)
        │
        ▼
 Interaction stored in memory ──────┐
        │                           │
        ▼                           │
 Importance scored                  │
        │                           │
        ▼                           │  Feedback
 High-importance? ──Yes──▶ Generate │  Loop
        │                  summary  │
        No                    │     │
        │                     ▼     │
        ▼              Write .md    │
    (wait for              file ────┘
     next query)            │
                            ▼
                    Next ingest picks
                    up the summary as
                    new knowledge
```

The more the system is used, the richer its knowledge base becomes — without any manual curation.

---

## Step-by-Step: How It Works

### Step 1: Store the Interaction

After every `query()` call, the system creates a `MemoryEntry`:

```python
MemoryEntry(
    id="a1b2c3d4e5f6",           # MD5 hash of query + timestamp
    query="What is backpropagation?",
    answer="Backpropagation is...",
    retrieved_context=["chunk1 text", "chunk2 text"],
    timestamp="2025-01-15T14:30:00",
    importance_score=0.6,
    access_count=1,
    summary=""
)
```

Entries are persisted as JSON Lines in `data/memory/interactions.jsonl`.

### Step 2: Compute Importance Score

The importance score (0.0–1.0) is computed based on:

| Signal | Score Boost | Rationale |
|---|---|---|
| Baseline | 0.5 | Every interaction has minimum importance |
| ≥ 3 chunks retrieved | +0.1 | Complex queries that needed multiple sources |
| Answer > 200 chars | +0.1 | Substantial answers contain more knowledge |
| Repeated similar query | +0.1 per repeat | Frequent topics deserve higher priority |

The score is capped at 1.0.

### Step 3: Deduplicate

Before creating a new entry, the system checks for similar existing entries using **Jaccard term overlap**:

```
Jaccard(Q1, Q2) = |terms(Q1) ∩ terms(Q2)| / |terms(Q1) ∪ terms(Q2)|
```

If the overlap ≥ `deduplication_threshold` (default: **0.85**), the existing entry is updated:
- `access_count += 1`
- `importance_score += 0.1` (capped at 1.0)

This prevents the memory from filling up with near-identical queries while boosting genuinely popular topics.

### Step 4: Generate Summary

When an entry's importance score reaches **≥ 0.6**, the system calls the LLM to generate a concise knowledge summary:

```
Prompt: "Based on the following Q&A interaction, generate a concise
knowledge summary that captures the key information..."

Input: question + answer
Output: 2-3 paragraph summary of the key knowledge
```

### Step 5: Write Back to Knowledge Base

The summary is saved as a new Markdown file in `data/memory/summaries/`:

```markdown
# Auto-Generated Summary

#auto-generated #memory

## Original Question
What is backpropagation?

## Key Insights
Backpropagation is the primary algorithm for training neural networks...

## Generated
2025-01-15T14:30:00
```

On the **next ingestion** (`python cli.py ingest`), these summary files can be included in the knowledge base, making their insights available for retrieval.

### Step 6: Prune Low-Value Entries

When the memory store exceeds `max_history` (default: **1000**):

1. All entries are sorted by importance score (descending)
2. The bottom entries are removed
3. This ensures limited storage is used for the most valuable knowledge

---

## Memory Retrieval: Using Past Interactions

The `get_relevant_history()` method finds past interactions relevant to a new query:

```python
# Score each past entry against the current query
term_score  = overlap(query_terms, entry_terms)  # Jaccard-like
combined    = term_score × 0.5
            + importance_score × frequency_weight
            + (access_count / 10) × recency_weight
```

This allows the system to recall past interactions and potentially use them as additional context. The weighting scheme balances:

- **Relevance** (term overlap) — Is this past entry about the same topic?
- **Importance** (frequency_weight: 0.3) — Is this entry high-value?
- **Popularity** (recency_weight: 0.7) — Has this entry been accessed often?

---

## Configuration

All memory parameters are tunable in `config/default.yaml`:

```yaml
memory:
  enabled: true                      # Toggle the entire memory system
  store_path: "data/memory/interactions.jsonl"
  summary_path: "data/memory/summaries"
  max_history: 1000                  # Max stored entries
  deduplication_threshold: 0.85      # Jaccard overlap for dedup
  scoring:
    importance_decay: 0.95           # Future: time-based decay
    frequency_weight: 0.3            # Weight for importance in retrieval
    recency_weight: 0.7              # Weight for access count in retrieval
```

---

## Observing the Loop in Action

```bash
# 1. Ask a question
python cli.py ask "What are transformers?"

# 2. Check memory
python cli.py memory-stats
# → Total entries: 1, Avg importance: 0.600

# 3. Ask a similar question (deduplication triggers)
python cli.py ask "How do transformers work?"
python cli.py memory-stats
# → Total entries: 1, Avg importance: 0.700, Total accesses: 2

# 4. Ask a different question
python cli.py ask "What is gradient descent?"
python cli.py memory-stats
# → Total entries: 2, Avg importance: 0.650

# 5. Check for generated summaries
ls data/memory/summaries/
# → summary_a1b2c3d4e5f6.md
```

---

## Design Inspiration

This approach is directly inspired by Andrej Karpathy's concept of **"persistent memory for AI systems"** — the idea that an AI should learn from its interactions and improve its knowledge over time, rather than treating every conversation as a clean slate. The key insight is that a system's *usage patterns* contain valuable signal about what knowledge matters most.

---

*Next: [Evaluation →](evaluation.md)*

