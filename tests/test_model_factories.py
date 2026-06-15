from src.docqa_stream.services import embeddings, llms


def test_vertexai_chat_model_uses_env(monkeypatch):
    captured = {}

    class FakeChatVertexAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llms, "get_chat_vertexai_class", lambda: FakeChatVertexAI)
    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    monkeypatch.setenv("VERTEXAI_MODEL", "gemini-test")
    monkeypatch.setenv("VERTEXAI_PROJECT", "docqa-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "europe-west4")

    llms.get_chat_model(0.2)

    assert captured == {
        "model": "gemini-test",
        "temperature": 0.2,
        "streaming": True,
        "project": "docqa-project",
        "location": "europe-west4",
    }


def test_vertexai_embeddings_model_uses_env(monkeypatch):
    captured = {}

    class FakeVertexAIEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        embeddings,
        "get_vertexai_embeddings_class",
        lambda: FakeVertexAIEmbeddings,
    )
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "vertexai")
    monkeypatch.setenv("VERTEXAI_EMBEDDING_MODEL", "embedding-test")
    monkeypatch.setenv("VERTEXAI_PROJECT", "docqa-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

    embeddings.get_embeddings_model()

    assert captured == {
        "model": "embedding-test",
        "project": "docqa-project",
        "location": "us-central1",
    }
