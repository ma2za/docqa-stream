import pytest

from src.docqa_stream import settings


def test_validate_config_reports_missing_backend_setting(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.delenv("PGVECTOR_CONNECTION", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "local-test-model")

    with pytest.raises(RuntimeError) as exc_info:
        settings.validate_config()

    assert "PGVECTOR_CONNECTION is required" in str(exc_info.value)


def test_validate_config_reports_invalid_int(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "many")

    with pytest.raises(RuntimeError) as exc_info:
        settings.validate_config()

    assert "MAX_UPLOAD_BYTES must be an integer" in str(exc_info.value)


def test_validate_config_accepts_fully_local_settings(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.setenv(
        "PGVECTOR_CONNECTION",
        "postgresql+psycopg://docqa:docqa@pgvector:5432/docqa",
    )
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "local-test-model")

    settings.validate_config()


def test_validate_config_does_not_require_embeddings_for_weaviate(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "weaviate")
    monkeypatch.setenv("WEAVIATE_SERVICE_NAME", "weaviate")
    monkeypatch.setenv("WEAVIATE_PORT", "8080")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)

    settings.validate_config()
