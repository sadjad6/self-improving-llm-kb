# Evaluation & Experiment Tracking

> How the system measures retrieval quality, answer quality, and tracks experiments for reproducibility.

---

## Why Evaluation Matters

A RAG system without evaluation is a black box. You can't tell if:
- The retriever is finding the right chunks
- The LLM is faithfully using the context
- A configuration change improved or degraded performance

This project implements **two categories of metrics** plus **MLflow experiment tracking** so every change is measurable and reproducible.

---

## Retrieval Metrics

These measure how well the retriever finds relevant documents.

### Recall@K

**Definition:** What fraction of all relevant documents appear in the top-K results?

```
Recall@K = |relevant ∩ top_K| / |relevant|
```

| K | Interpretation |
|---|---|
| Recall@1 | Does the single best result contain the answer? |
| Recall@3 | Can the answer be found in the top 3 results? |
| Recall@5 | (Default top_k) — Covers most use cases |
| Recall@10 | Upper bound on retrieval quality |

**Example:** If there are 3 relevant chunks and 2 appear in the top 5:
```
Recall@5 = 2/3 = 0.667
```

### Mean Reciprocal Rank (MRR)

**Definition:** How high is the *first* relevant result?

```
MRR = 1 / rank_of_first_relevant
```

| First relevant at rank | MRR |
|---|---|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| Not found | 0.000 |

MRR rewards systems that place the most relevant chunk at the very top.

### Usage

```python
from src.evaluation.metrics import RetrievalEvaluator

evaluator = RetrievalEvaluator()
results = evaluator.evaluate(
    retrieved=retrieval_results,
    relevant_ids={"chunk_001", "chunk_007", "chunk_012"},
    k_values=[1, 3, 5, 10],
)
for r in results:
    print(f"{r.metric_name}: {r.score:.3f}")
```

---

## Answer Quality Metrics

These measure how good the LLM's generated answer is.

### Heuristic Scoring (Automated)

Four automated checks combined into a composite score (0–1):

| Sub-metric | What it checks | Score logic |
|---|---|---|
| **Length** | Is the answer a reasonable length? | < 10 words → 0.2, 10–30 → 0.6, 30–300 → 1.0, > 300 → 0.7 |
| **Grounding** | Are answer terms present in the context? | `|answer_terms ∩ context_terms| / |answer_terms|` |
| **Query coverage** | Does the answer address the question? | `|query_terms ∩ answer_terms| / |query_terms|` (excluding stop words) |
| **Has answer** | Did the model actually answer (vs. "I can't answer")? | Contains refusal phrase → 0.3, otherwise → 1.0 |

**Composite = mean of all four sub-metrics.**

```python
from src.evaluation.metrics import AnswerEvaluator

evaluator = AnswerEvaluator()
result = evaluator.heuristic_score(
    answer="Backpropagation is an algorithm...",
    context_texts=["Neural networks use backprop...", ...],
    query="What is backpropagation?"
)
print(f"Score: {result.score:.3f}")
print(f"Details: {result.details}")
# → {'length': 1.0, 'grounding': 0.82, 'query_coverage': 0.75, 'has_answer': 1.0}
```

### LLM-as-Judge (Model-Based)

A separate LLM call evaluates the answer on three dimensions (1–5 scale):

| Dimension | What it measures |
|---|---|
| **Relevance** | Does the answer address the question? |
| **Faithfulness** | Is the answer consistent with the provided context? (No hallucination) |
| **Completeness** | Does the answer cover all relevant aspects? |

The system generates a structured prompt and expects JSON output:

```json
{"relevance": 4, "faithfulness": 5, "completeness": 3}
```

```python
prompt = AnswerEvaluator.llm_judge_prompt(
    query="What is attention?",
    answer="Attention is a mechanism...",
    context="Transformers use self-attention..."
)
# Send `prompt` to an LLM and parse the JSON response
```

---

## MLflow Experiment Tracking

The `ExperimentTracker` class wraps MLflow to log every experiment run with full reproducibility.

### What Gets Tracked

| Category | Examples |
|---|---|
| **Parameters** | Retrieval method, top_k, chunk_max_tokens, dense_weight, model_name |
| **Metrics** | Recall@1, Recall@5, MRR, heuristic_score, latency_ms |
| **Artifacts** | Config YAML, evaluation results, answer samples |

### Usage

```python
from src.evaluation.tracker import ExperimentTracker

tracker = ExperimentTracker()
run_id = tracker.start_run("hybrid_vs_dense")

tracker.log_params({
    "retrieval_method": "hybrid",
    "dense_weight": 0.6,
    "top_k": 5,
})
tracker.log_metrics({
    "recall@5": 0.82,
    "mrr": 0.67,
    "heuristic_score": 0.78,
    "latency_ms": 1234,
})
tracker.end_run()
```

### Viewing Results

```bash
mlflow ui --backend-store-uri mlflow_runs
```

Opens at `http://localhost:5000` with a dashboard showing all runs, metrics comparison, and parameter search.

---

## Running the Built-in Evaluation

The CLI includes a quick evaluation command:

```bash
python cli.py evaluate --method hybrid
```

This runs four sample queries, computes heuristic scores, and displays results in a formatted table. To compare methods:

```bash
python cli.py evaluate --method dense
python cli.py evaluate --method sparse
python cli.py evaluate --method hybrid
```

---

## Evaluation Best Practices

1. **Create a ground-truth test set** — Manually label which chunks are relevant for each query; this enables accurate Recall@K and MRR measurement
2. **Compare methods side-by-side** — Always evaluate dense, sparse, and hybrid to understand each method's strengths
3. **Track config changes** — Use MLflow to log every parameter change so you can reproduce your best configuration
4. **Use both heuristic and LLM-as-Judge** — Heuristics are fast and free; LLM-as-Judge catches nuance that heuristics miss

---

*Next: [API Reference →](api_reference.md)*

