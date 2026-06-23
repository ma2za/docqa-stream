import os


def get_int(name, default):
    return int(os.getenv(name, str(default)))


def _is_set(name):
    return bool(os.getenv(name))


def _require(errors, name):
    if not _is_set(name):
        errors.append(f"{name} is required")


def _require_one(errors, names):
    if not any(_is_set(name) for name in names):
        errors.append(f"one of {', '.join(names)} is required")


def _validate_int(errors, name, minimum=None):
    value = os.getenv(name)
    if value is None or value == "":
        return
    try:
        parsed = int(value)
    except ValueError:
        errors.append(f"{name} must be an integer")
        return
    if minimum is not None and parsed < minimum:
        errors.append(f"{name} must be at least {minimum}")


def _validate_provider(errors, name, allowed, default):
    value = os.getenv(name, default).lower()
    if value not in allowed:
        errors.append(f"{name} must be one of {', '.join(sorted(allowed))}")
    return value


def validate_config():
    errors = []
    vector_store = _validate_provider(
        errors,
        "VECTOR_STORE",
        {"weaviate", "pgvector"},
        "weaviate",
    )
    llm_provider = _validate_provider(
        errors,
        "LLM_PROVIDER",
        {"azure", "openai", "vertexai", "ollama"},
        "azure",
    )
    embeddings_provider = None
    if vector_store == "pgvector" or _is_set("EMBEDDINGS_PROVIDER"):
        embeddings_provider = _validate_provider(
            errors,
            "EMBEDDINGS_PROVIDER",
            {"azure", "openai", "vertexai", "local"},
            llm_provider,
        )

    for name in POSITIVE_INT_SETTINGS:
        _validate_int(errors, name, minimum=1)
    _validate_int(errors, "CHUNK_OVERLAP", minimum=0)
    for name in OPTIONAL_INT_SETTINGS:
        _validate_int(errors, name, minimum=1)

    if vector_store == "weaviate":
        _require(errors, "WEAVIATE_SERVICE_NAME")
        _require(errors, "WEAVIATE_PORT")
    elif vector_store == "pgvector":
        _require(errors, "PGVECTOR_CONNECTION")

    if llm_provider == "azure":
        _require(errors, "OPENAI_DEPLOYMENT_NAME")
        _require_one(errors, ["AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE"])
        _require_one(errors, ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"])
    elif llm_provider == "openai":
        _require(errors, "OPENAI_MODEL")
    elif llm_provider == "vertexai":
        _require(errors, "VERTEXAI_MODEL")
    elif llm_provider == "ollama":
        _require(errors, "OLLAMA_MODEL")

    if vector_store == "pgvector":
        if embeddings_provider == "azure":
            _require(errors, "OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            _require_one(errors, ["AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE"])
            _require_one(errors, ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"])
        elif embeddings_provider == "openai":
            _require(errors, "OPENAI_EMBEDDING_MODEL")
        elif embeddings_provider == "vertexai":
            _require(errors, "VERTEXAI_EMBEDDING_MODEL")
        elif embeddings_provider == "local":
            _require(errors, "LOCAL_EMBEDDING_MODEL")

    if errors:
        raise RuntimeError("Invalid configuration: " + "; ".join(errors))


POSITIVE_INT_SETTINGS = [
    "FASTAPI_PORT",
    "MAX_UPLOAD_BYTES",
    "UPLOAD_READ_CHUNK_BYTES",
    "MAX_CHUNKS_PER_UPLOAD",
    "VECTORSTORE_ADD_BATCH_SIZE",
    "CITATION_PREVIEW_CHARS",
    "LIST_DOCUMENT_CHUNK_SCAN_LIMIT",
    "MAX_QUESTION_CHARS",
    "WEAVIATE_PORT",
    "WEAVIATE_GRPC_PORT",
]

OPTIONAL_INT_SETTINGS = [
    "EMBEDDING_DIMENSIONS",
    "OLLAMA_NUM_CTX",
    "OLLAMA_NUM_THREAD",
    "OLLAMA_NUM_PREDICT",
]


MAX_UPLOAD_BYTES = get_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
UPLOAD_READ_CHUNK_BYTES = get_int("UPLOAD_READ_CHUNK_BYTES", 1024 * 1024)
MAX_CHUNKS_PER_UPLOAD = get_int("MAX_CHUNKS_PER_UPLOAD", 5000)
VECTORSTORE_ADD_BATCH_SIZE = get_int("VECTORSTORE_ADD_BATCH_SIZE", 128)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 20)
CITATION_PREVIEW_CHARS = get_int("CITATION_PREVIEW_CHARS", 240)
LIST_DOCUMENT_CHUNK_SCAN_LIMIT = get_int("LIST_DOCUMENT_CHUNK_SCAN_LIMIT", 10000)
MAX_QUESTION_CHARS = get_int("MAX_QUESTION_CHARS", 2000)
