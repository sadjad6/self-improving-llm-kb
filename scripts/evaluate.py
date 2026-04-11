"""Standalone evaluation script for the Self-Improving LLM Knowledge Base.

Runs a predefined query set against the knowledge base, computes retrieval
and answer quality metrics, and logs all results to MLflow.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --method dense --top-k 10
    python scripts/evaluate.py --config config/default.yaml --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import AnswerEvaluator, RetrievalEvaluator
from src.pipeline import KnowledgePipeline
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation dataset
# Each entry has a query and optional relevant chunk IDs for retrieval metrics.
# When relevant_ids is empty, only answer quality is evaluated.
# ---------------------------------------------------------------------------
EVAL_DATASET: list[dict] = [
    {
        "query": "What are the main types of machine learning?",
        "relevant_ids": set(),  # Fill in after first indexing run
    },
    {
        "query": "How do transformers use self-attention?",
        "relevant_ids": set(),
    },
    {
        "query": "What is retrieval-augmented generation and why is it useful?",
        "relevant_ids": set(),
    },
    {
        "query": "What metrics are used to evaluate retrieval systems?",
        "relevant_ids": set(),
    },
    {
        "query": "Explain the difference between supervised and unsupervised learning.",
        "relevant_ids": set(),
    },
    {
        "query": "What is overfitting and how can it be prevented?",
        "relevant_ids": set(),
    },
    {
        "query": "How does FAISS enable fast vector similarity search?",
        "relevant_ids": set(),
    },
]

K_VALUES = [1, 3, 5]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the Self-Improving LLM Knowledge Base")
    parser.add_argument(
        "--method",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval strategy (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging (useful when mlflow is not available)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results as JSON",
    )
    return parser.parse_args()


def _try_start_mlflow(config: object, run_name: str) -> object | None:
    """Attempt to start an MLflow run; return tracker or None on failure."""
    try:
        from src.evaluation.tracker import ExperimentTracker

        tracker = ExperimentTracker(config=config.experiment)  # type: ignore[union-attr]
        tracker.start_run(run_name=run_name)
        return tracker
    except Exception as exc:
        logger.warning("MLflow unavailable, skipping tracking: %s", exc)
        return None


def run_evaluation(
    pipeline: KnowledgePipeline,
    method: str,
    top_k: int,
    use_mlflow: bool,
) -> dict:
    """Run the full evaluation loop and return aggregated metrics."""
    retrieval_evaluator = RetrievalEvaluator()
    answer_evaluator = AnswerEvaluator()

    tracker = None
    if use_mlflow:
        run_name = f"eval_{method}_k{top_k}_{int(time.time())}"
        tracker = _try_start_mlflow(pipeline.config, run_name)
        if tracker:
            tracker.log_params(
                {
                    "method": method,
                    "top_k": top_k,
                    "model": pipeline.config.llm.model,
                }
            )

    all_results = []
    recall_scores: dict[int, list[float]] = {k: [] for k in K_VALUES}
    mrr_scores: list[float] = []
    answer_scores: list[float] = []
    latencies: list[float] = []

    print(f"\n{'='*60}")
    print(f"  Evaluation — method={method}, top_k={top_k}")
    print(f"{'='*60}\n")

    for i, entry in enumerate(EVAL_DATASET, 1):
        query = entry["query"]
        relevant_ids: set[str] = entry["relevant_ids"]

        print(f"[{i}/{len(EVAL_DATASET)}] {query[:60]}...")

        try:
            result = pipeline.query(query, method=method, top_k=top_k)
        except Exception as exc:
            logger.error("Query failed: %s — %s", query, exc)
            continue

        latencies.append(result.latency_ms)

        # Retrieval metrics (only when relevant_ids are provided)
        if relevant_ids:
            for k in K_VALUES:
                r_eval = retrieval_evaluator.recall_at_k(result.retrieved_chunks, relevant_ids, k)
                recall_scores[k].append(r_eval.score)

            mrr_eval = retrieval_evaluator.mrr(result.retrieved_chunks, relevant_ids)
            mrr_scores.append(mrr_eval.score)

        # Answer quality
        ctx_texts = [r.chunk.content for r in result.retrieved_chunks]
        a_eval = answer_evaluator.heuristic_score(result.answer, ctx_texts, query)
        answer_scores.append(a_eval.score)

        entry_result = {
            "query": query,
            "answer_score": a_eval.score,
            "latency_ms": result.latency_ms,
            "chunks_retrieved": len(result.retrieved_chunks),
            "token_usage": result.token_usage,
            "answer_details": a_eval.details,
        }
        all_results.append(entry_result)

        print(
            f"    ✓ answer_score={a_eval.score:.3f}  "
            f"latency={result.latency_ms:.0f}ms  "
            f"tokens={result.token_usage.get('total_tokens', 'N/A')}"
        )

    # Aggregate
    avg_answer = sum(answer_scores) / len(answer_scores) if answer_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else None

    aggregated = {
        "method": method,
        "top_k": top_k,
        "num_queries": len(EVAL_DATASET),
        "avg_answer_score": avg_answer,
        "avg_latency_ms": avg_latency,
    }
    if avg_mrr is not None:
        aggregated["avg_mrr"] = avg_mrr
        for k in K_VALUES:
            if recall_scores[k]:
                aggregated[f"avg_recall@{k}"] = sum(recall_scores[k]) / len(recall_scores[k])

    print(f"\n{'='*60}")
    print("  Aggregated Results")
    print(f"{'='*60}")
    for key, val in aggregated.items():
        if isinstance(val, float):
            print(f"  {key:<30} {val:.4f}")
        else:
            print(f"  {key:<30} {val}")

    if tracker:
        tracker.log_metrics({k: v for k, v in aggregated.items() if isinstance(v, float)})
        tracker.end_run()
        print("\n  ✓ Results logged to MLflow")

    return {"aggregated": aggregated, "per_query": all_results}


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    pipeline = KnowledgePipeline(config=config)

    print("Ingesting knowledge base...")
    n_chunks = pipeline.ingest()
    print(f"✓ Indexed {n_chunks} chunks\n")

    results = run_evaluation(
        pipeline=pipeline,
        method=args.method,
        top_k=args.top_k,
        use_mlflow=not args.no_mlflow,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
