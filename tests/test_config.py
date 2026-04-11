"""Tests for configuration management."""

from pathlib import Path

from src.utils.config import AppConfig, load_config


class TestConfig:
    def test_default_config(self) -> None:
        config = AppConfig()
        assert config.ingestion.chunk_strategy == "semantic"
        assert config.retrieval.dense.model_name == "all-MiniLM-L6-v2"
        assert config.llm.provider == "openai"
        assert config.memory.enabled is True

    def test_load_config_from_file(self) -> None:
        config = load_config("config/default.yaml")
        assert config.ingestion.chunk_max_tokens == 512
        assert config.retrieval.hybrid.dense_weight == 0.6

    def test_load_missing_config_uses_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(config, AppConfig)

    def test_retrieval_config_structure(self) -> None:
        config = AppConfig()
        assert config.retrieval.dense.top_k == 10
        assert config.retrieval.sparse.algorithm == "bm25"
        assert config.retrieval.reranker.enabled is False
