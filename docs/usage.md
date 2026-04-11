# Usage Guide

> How to interact with the system via CLI and the Streamlit web UI.

---

## CLI (Command Line Interface)

The CLI is built with [Click](https://click.palletsprojects.com/) and uses [Rich](https://rich.readthedocs.io/) for formatted terminal output. All commands support `--config path/to/config.yaml` to override the default configuration.

### `ingest` — Index the Knowledge Base

```bash
python cli.py ingest
```

Parses all Markdown files in `data/knowledge_base/`, chunks them semantically, and builds both the FAISS dense index and BM25 sparse index.

Options:
- `--knowledge-dir PATH` — Override the knowledge base directory

Example output:
```
✓ Indexed 47 chunks successfully.
```

---

### `ask` — Query the Knowledge Base

```bash
python cli.py ask "What is retrieval-augmented generation?"
```

Ingests the knowledge base (if not already indexed), retrieves relevant chunks, generates an LLM answer, and stores the interaction in memory.

Options:
- `--method [dense|sparse|hybrid]` — Retrieval strategy (default: `hybrid`)
- `--top-k N` — Number of chunks to retrieve (default: `5`)

Example output:
```
┌──────────── Retrieved Context ────────────┐
│ #  Source                Score    Preview  │
│ 1  RAG Overview          0.8234  ...      │
│ 2  Transformers          0.7891  ...      │
│ 3  Neural Networks       0.6543  ...      │
└───────────────────────────────────────────┘

╭──────────────── Answer ────────────────╮
│ Retrieval-Augmented Generation (RAG)   │
│ is a technique that combines...        │
╰────────────────────────────────────────╯

Method: hybrid | Latency: 1234ms | Tokens: 456
```

---

### `memory-stats` — View Memory Statistics

```bash
python cli.py memory-stats
```

Displays the current state of the self-improving memory system.

Example output:
```
╭──── Memory Statistics ────╮
│ Total entries: 12         │
│ Avg importance: 0.583     │
│ Total accesses: 28        │
╰───────────────────────────╯
```

---

### `evaluate` — Run Evaluation Benchmark

```bash
python cli.py evaluate --method hybrid
```

Runs a set of sample queries and evaluates answer quality using heuristic scoring.

Options:
- `--method [dense|sparse|hybrid]` — Retrieval strategy to evaluate

Example output:
```
┌── Evaluation Results (hybrid) ────────────────────────┐
│ Query                              Score   Latency    │
│ What are the main types of ML?     0.812   1102       │
│ How do transformers use attention?  0.756   987        │
│ What is RAG and how does it work?   0.834   1245       │
│ What metrics evaluate retrieval?    0.789   1056       │
└───────────────────────────────────────────────────────┘
```

---

## Streamlit Web UI

The web interface provides an interactive, visual way to explore the system.

### Launch

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501` by default.

### Features

#### 1. Query Panel
- Type a question in the input field
- Select retrieval method (Dense / Sparse / Hybrid) from the sidebar
- Adjust top-K slider
- Click **"Ask"** to get a grounded answer

#### 2. Retrieved Context Viewer
Each retrieved chunk is displayed with:
- **Source document** name (color-coded badge)
- **Relevance score** (0–1)
- **Full content** expandable section
- **Heading context** showing where the chunk came from in the original document

#### 3. Performance Metrics
Real-time display of:
- Response latency (ms)
- Token usage (prompt + completion)
- Retrieval method used

#### 4. Memory & Self-Improvement Section
- Total memory entries and average importance score
- Recent interactions with timestamps
- Visual indicator of the self-improving feedback loop

#### 5. "How It Works" Panel
When no query has been submitted, the UI shows an architecture overview explaining the system's components — useful for demos and portfolio presentations.

---

## Example Workflows

### Workflow 1: Compare Retrieval Methods

```bash
python cli.py ask "What is attention?" --method dense
python cli.py ask "What is attention?" --method sparse
python cli.py ask "What is attention?" --method hybrid
```

Compare which chunks are retrieved and how answer quality differs.

### Workflow 2: Observe Self-Improvement

```bash
# Ask the same question multiple times
python cli.py ask "What is backpropagation?"
python cli.py ask "How does backpropagation work?"  # Similar question
python cli.py memory-stats  # Watch importance scores increase
```

The second query is deduplicated — the existing entry's importance score rises instead of creating a duplicate.

### Workflow 3: Add New Knowledge

```bash
# Add a new Markdown file
echo "# Reinforcement Learning\n\nRL is a type of ML where..." > data/knowledge_base/reinforcement_learning.md

# Re-ingest
python cli.py ingest

# Query the new topic
python cli.py ask "What is reinforcement learning?"
```

### Workflow 4: Run Evaluation After Changes

```bash
python cli.py evaluate --method hybrid
# Adjust config/default.yaml (e.g., change weights)
python cli.py evaluate --method hybrid
# Compare results
```

---

## Programmatic Usage

You can also use the system as a Python library:

```python
from src.pipeline import KnowledgePipeline

pipeline = KnowledgePipeline()
pipeline.ingest()

# Full query with LLM
result = pipeline.query("What is a neural network?", method="hybrid")
print(result.answer)
print(f"Latency: {result.latency_ms:.0f}ms")

# Retrieval only (no LLM call)
chunks = pipeline.retrieve_only("neural network", top_k=3)
for r in chunks:
    print(f"[{r.score:.3f}] {r.chunk.content[:80]}...")
```

---

*Next: [Self-Improving Loop →](self_improving_loop.md)*

