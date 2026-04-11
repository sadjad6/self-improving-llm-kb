"""Configuration management for the knowledge base system."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass
class IngestionConfig:
    chunk_strategy: str = "semantic"
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    preserve_headings: bool = True
    extract_links: bool = True
    extract_tags: bool = True
    knowledge_dir: str = "data/knowledge_base"


@dataclass
class DenseRetrievalConfig:
    model_name: str = "all-MiniLM-L6-v2"
    index_path: str = "data/indices/faiss.index"
    top_k: int = 10


@dataclass
class SparseRetrievalConfig:
    algorithm: str = "bm25"
    top_k: int = 10


@dataclass
class HybridConfig:
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    top_k: int = 5


@dataclass
class RerankerConfig:
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 3


@dataclass
class RetrievalConfig:
    dense: DenseRetrievalConfig = field(default_factory=DenseRetrievalConfig)
    sparse: SparseRetrievalConfig = field(default_factory=SparseRetrievalConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    system_prompt: str = (
        "You are a knowledgeable assistant that answers questions based strictly on "
        "the provided context. If the context does not contain enough information "
        "to answer, say so explicitly. Do not hallucinate or invent information."
    )


@dataclass
class MemoryConfig:
    enabled: bool = True
    store_path: str = "data/memory/interactions.jsonl"
    summary_path: str = "data/memory/summaries"
    max_history: int = 1000
    deduplication_threshold: float = 0.85
    importance_decay: float = 0.95
    frequency_weight: float = 0.3
    recency_weight: float = 0.7


@dataclass
class EvaluationConfig:
    retrieval_metrics: list[str] = field(default_factory=lambda: ["recall@k", "mrr"])
    answer_metrics: list[str] = field(default_factory=lambda: ["llm_judge", "heuristic"])
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class ExperimentConfig:
    tracking_uri: str = "mlflow_runs"
    experiment_name: str = "knowledge_base"


@dataclass
class AppConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict."""
    if not data:
        return cls()
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in field_types and isinstance(value, dict):
            # Nested dataclass
            nested_cls = cls.__dataclass_fields__[key].type
            if isinstance(nested_cls, str):
                nested_cls = globals().get(nested_cls) or locals().get(nested_cls)
            if nested_cls and hasattr(nested_cls, "__dataclass_fields__"):
                kwargs[key] = _build_dataclass(nested_cls, value)
                continue
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to the config file. Uses default if None.

    Returns:
        Populated AppConfig dataclass.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file %s not found, using defaults.", config_path)
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = AppConfig()
    if "ingestion" in raw:
        config.ingestion = _build_dataclass(IngestionConfig, raw["ingestion"])
    if "retrieval" in raw:
        r = raw["retrieval"]
        config.retrieval = RetrievalConfig(
            dense=_build_dataclass(DenseRetrievalConfig, r.get("dense", {})),
            sparse=_build_dataclass(SparseRetrievalConfig, r.get("sparse", {})),
            hybrid=_build_dataclass(HybridConfig, r.get("hybrid", {})),
            reranker=_build_dataclass(RerankerConfig, r.get("reranker", {})),
        )
    if "llm" in raw:
        config.llm = _build_dataclass(LLMConfig, raw["llm"])
    if "memory" in raw:
        mem = dict(raw["memory"])  # Copy to avoid mutating original
        scoring = mem.pop("scoring", {})
        mem.update(scoring)
        config.memory = _build_dataclass(MemoryConfig, mem)
    if "evaluation" in raw:
        config.evaluation = _build_dataclass(EvaluationConfig, raw["evaluation"])
    if "experiment" in raw:
        config.experiment = _build_dataclass(ExperimentConfig, raw["experiment"])
    return config

