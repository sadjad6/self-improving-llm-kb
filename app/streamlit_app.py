"""Streamlit UI for the Self-Improving LLM Knowledge Base."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.pipeline import KnowledgePipeline
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Self-Improving Knowledge Base",
    page_icon="🧠",
    layout="wide",
)

setup_logging()


# ──────────────────────────────────────────────
# Custom CSS for polish
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stMetric > div { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
    .source-badge {
        display: inline-block; background: #e3f2fd; color: #1565c0;
        padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 600;
    }
    .score-badge {
        display: inline-block; background: #e8f5e9; color: #2e7d32;
        padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Pipeline initialization (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_pipeline() -> KnowledgePipeline:
    """Initialize and ingest the pipeline once."""
    config = load_config()
    pipeline = KnowledgePipeline(config=config)
    pipeline.ingest()
    return pipeline


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🧠 Self-Improving LLM Knowledge Base")
st.caption(
    "Ask questions grounded in a local knowledge base · "
    "Hybrid retrieval (Dense + Sparse) · Self-improving memory loop"
)
st.divider()

# ──────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    retrieval_method = st.selectbox(
        "Retrieval Method",
        options=["hybrid", "dense", "sparse"],
        index=0,
        help="Dense = semantic similarity, Sparse = keyword matching, Hybrid = both combined",
    )
    top_k = st.slider("Top-K Documents", min_value=1, max_value=15, value=5)
    st.divider()

    st.header("📊 System Info")
    try:
        pipeline = get_pipeline()
        st.metric("Indexed Chunks", len(pipeline._chunks))
        mem_stats = pipeline.memory.get_stats()
        st.metric("Memory Entries", mem_stats["total_entries"])
    except ValueError as exc:
        st.warning(f"⚠️ {exc}")
    except Exception:
        st.info("Pipeline not yet loaded.")

# ──────────────────────────────────────────────
# Query input
# ──────────────────────────────────────────────
st.subheader("💬 Query")
query = st.text_input(
    "Ask a question about the knowledge base",
    placeholder="e.g. How do transformers use self-attention?",
    label_visibility="collapsed",
)

if query:
    try:
        pipeline = get_pipeline()
    except ValueError as exc:
        st.error(
            f"⚠️ **Configuration error:** {exc}\n\n"
            "Set `OPENAI_API_KEY` in your `.env` file and restart the app."
        )
        st.stop()
    else:
        with st.spinner("🔍 Retrieving & generating answer..."):
            start = time.time()
            result = pipeline.query(query, method=retrieval_method, top_k=top_k)
            wall_time = (time.time() - start) * 1000

    # ── Metadata bar ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Latency", f"{result.latency_ms:.0f} ms")
    col2.metric("📄 Chunks Retrieved", len(result.retrieved_chunks))
    col3.metric("🔍 Method", result.retrieval_method.title())
    col4.metric("🪙 Tokens Used", result.token_usage.get("total_tokens", "N/A"))

    st.divider()

    # ── Answer section ──
    st.subheader("✅ Answer")
    st.markdown(result.answer)

    st.divider()

    # ── Retrieved context ──
    st.subheader("📚 Retrieved Context")
    for i, r in enumerate(result.retrieved_chunks, 1):
        source = r.chunk.metadata.get("title", "Unknown")
        heading = r.chunk.heading_context
        label = f"**{source}**"
        if heading:
            label += f" › {heading}"

        with st.expander(f"#{i}  {source} — Score: {r.score:.4f}", expanded=(i <= 2)):
            st.markdown(
                f'<span class="source-badge">{source}</span> '
                f'<span class="score-badge">Score: {r.score:.4f}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(r.chunk.content)

    # ── Memory section ──
    st.divider()
    st.subheader("🧠 Memory & Self-Improvement")
    mem_stats = pipeline.memory.get_stats()
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Total Memory Entries", mem_stats["total_entries"])
    mcol2.metric("Avg Score", f'{mem_stats.get("average_score", 0):.2f}')

    recent = pipeline.memory.get_recent(n=5)
    if recent:
        with st.expander("🕐 Recent interactions", expanded=False):
            for entry in recent:
                st.markdown(f"**Q:** {entry.query}")
                st.caption(f"Importance: {entry.importance_score:.2f} · {entry.timestamp}")
    else:
        st.caption("No memory entries yet. Ask questions to build memory.")

else:
    # Empty state
    st.info("👆 Enter a question above to get started.")
    with st.expander("💡 Example questions"):
        st.markdown(
            "- What are the main types of machine learning?\n"
            "- How does self-attention work in transformers?\n"
            "- What is RAG and why is it useful?\n"
            "- What metrics are used for evaluating retrieval systems?\n"
            "- Explain overfitting and underfitting"
        )

    # ── Architecture overview ──
    with st.expander("🏗️ How it works"):
        st.markdown(
            "1. **Ingestion** — Markdown documents are parsed and semantically chunked\n"
            "2. **Indexing** — Chunks are indexed with FAISS (dense) and BM25 (sparse)\n"
            "3. **Retrieval** — Hybrid search combines both signals via RRF\n"
            "4. **Reasoning** — LLM generates grounded answers from retrieved context\n"
            "5. **Memory** — Query-answer pairs are scored and stored for self-improvement"
        )

