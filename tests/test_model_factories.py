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


def test_openai_embeddings_model_uses_dimensions(monkeypatch):
    captured = {}

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        embeddings,
        "get_openai_embeddings_class",
        lambda: FakeOpenAIEmbeddings,
    )
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embedding-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")

    embeddings.get_embeddings_model()

    assert captured == {
        "model": "embedding-test",
        "api_key": "test-key",
        "dimensions": 384,
    }


def test_local_embeddings_model_uses_sentence_transformer(monkeypatch):
    captured = {}

    class FakeArray:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def encode(self, texts, convert_to_numpy=True):
            captured["texts"] = texts
            captured["convert_to_numpy"] = convert_to_numpy
            return FakeArray([[0.1, 0.2], [0.3, 0.4]])

    monkeypatch.setattr(
        embeddings,
        "get_sentence_transformer_class",
        lambda: FakeSentenceTransformer,
    )
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "local-test-model")
    monkeypatch.setenv("LOCAL_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("LOCAL_EMBEDDING_CACHE_FOLDER", ".cache/models")
    monkeypatch.setenv("LOCAL_EMBEDDING_LOCAL_FILES_ONLY", "True")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "2")

    model = embeddings.get_embeddings_model()

    assert model.embed_documents(["hello", "world"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert captured == {
        "args": ("local-test-model",),
        "kwargs": {
            "device": "cpu",
            "cache_folder": ".cache/models",
            "local_files_only": True,
            "truncate_dim": 2,
        },
        "texts": ["hello", "world"],
        "convert_to_numpy": True,
    }
