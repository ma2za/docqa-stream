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


def test_ollama_chat_model_uses_env(monkeypatch):
    captured = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llms, "get_chat_ollama_class", lambda: FakeChatOllama)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    monkeypatch.setenv("OLLAMA_NUM_THREAD", "8")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "256")

    llms.get_chat_model(0.1)

    assert captured == {
        "model": "qwen2.5:0.5b",
        "temperature": 0.1,
        "base_url": "http://ollama:11434",
        "keep_alive": "30m",
        "num_ctx": 2048,
        "num_thread": 8,
        "num_predict": 256,
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
    calls = []

    class FakeArray:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
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
    embeddings.get_local_embeddings_model.cache_clear()

    model = embeddings.get_embeddings_model()
    cached = embeddings.get_embeddings_model()

    assert model.embed_documents(["hello", "world"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert cached is model
    assert len(calls) == 1
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
