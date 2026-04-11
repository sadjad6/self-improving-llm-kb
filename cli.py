"""CLI interface for the Self-Improving LLM Knowledge Base."""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import KnowledgePipeline
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

console = Console()


@click.group()
@click.option("--config", default=None, help="Path to config file")
@click.pass_context
def main(ctx: click.Context, config: str | None) -> None:
    """Self-Improving LLM Knowledge Base System."""
    setup_logging()
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@main.command()
@click.option("--knowledge-dir", default=None, help="Path to knowledge base")
@click.pass_context
def ingest(ctx: click.Context, knowledge_dir: str | None) -> None:
    """Ingest and index the knowledge base."""
    cfg = ctx.obj["config"]
    pipeline = KnowledgePipeline(config=cfg)
    n_chunks = pipeline.ingest(knowledge_dir)
    console.print(f"[green]✓[/green] Indexed {n_chunks} chunks successfully.")


@main.command()
@click.argument("question")
@click.option("--method", default="hybrid", type=click.Choice(["dense", "sparse", "hybrid"]))
@click.option("--top-k", default=5, help="Number of chunks to retrieve")
@click.pass_context
def ask(ctx: click.Context, question: str, method: str, top_k: int) -> None:
    """Ask a question against the knowledge base."""
    cfg = ctx.obj["config"]
    pipeline = KnowledgePipeline(config=cfg)

    with console.status("[bold blue]Ingesting knowledge base..."):
        pipeline.ingest()

    with console.status("[bold blue]Thinking..."):
        result = pipeline.query(question, method=method, top_k=top_k)

    # Display retrieved context
    if result.retrieved_chunks:
        table = Table(title="Retrieved Context", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Score", width=8)
        table.add_column("Preview", max_width=60)

        for i, r in enumerate(result.retrieved_chunks, 1):
            source = r.chunk.metadata.get("title", "Unknown")
            preview = r.chunk.content[:120].replace("\n", " ") + "..."
            table.add_row(str(i), source, f"{r.score:.4f}", preview)

        console.print(table)

    # Display answer
    console.print(Panel(result.answer, title="Answer", border_style="green"))

    # Display metadata
    meta = (
        f"Method: {result.retrieval_method} | "
        f"Latency: {result.latency_ms:.0f}ms | "
        f"Tokens: {result.token_usage.get('total_tokens', 'N/A')}"
    )
    console.print(f"[dim]{meta}[/dim]")


@main.command()
@click.pass_context
def memory_stats(ctx: click.Context) -> None:
    """Show memory system statistics."""
    from src.memory.store import MemoryStore

    cfg = ctx.obj["config"]
    store = MemoryStore(config=cfg.memory)
    stats = store.get_stats()
    console.print(
        Panel(
            f"Total entries: {stats['total_entries']}\n"
            f"Avg importance: {stats['avg_importance']:.3f}\n"
            f"Total accesses: {stats['total_accesses']}",
            title="Memory Statistics",
        )
    )


@main.command()
@click.option("--method", default="hybrid", type=click.Choice(["dense", "sparse", "hybrid"]))
@click.pass_context
def evaluate(ctx: click.Context, method: str) -> None:
    """Run evaluation on sample queries."""
    from src.evaluation.metrics import AnswerEvaluator

    cfg = ctx.obj["config"]
    pipeline = KnowledgePipeline(config=cfg)

    with console.status("[bold blue]Ingesting..."):
        pipeline.ingest()

    sample_queries = [
        "What are the main types of machine learning?",
        "How do transformers use attention?",
        "What is RAG and how does it work?",
        "What metrics are used to evaluate retrieval?",
    ]

    evaluator = AnswerEvaluator()
    table = Table(title=f"Evaluation Results ({method})", show_lines=True)
    table.add_column("Query", max_width=40)
    table.add_column("Score", width=8)
    table.add_column("Latency (ms)", width=12)

    for q in sample_queries:
        result = pipeline.query(q, method=method)
        ctx_texts = [r.chunk.content for r in result.retrieved_chunks]
        score = evaluator.heuristic_score(result.answer, ctx_texts, q)
        table.add_row(q, f"{score.score:.3f}", f"{result.latency_ms:.0f}")

    console.print(table)


if __name__ == "__main__":
    main()
