import os

from langchain_core.embeddings import Embeddings


def get_azure_openai_embeddings_class():
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings


def get_openai_embeddings_class():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings


def get_vertexai_embeddings_class():
    from langchain_google_vertexai import VertexAIEmbeddings

    return VertexAIEmbeddings


def get_sentence_transformer_class():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def get_embedding_dimensions():
    if not os.getenv("EMBEDDING_DIMENSIONS"):
        return None
    return int(os.environ["EMBEDDING_DIMENSIONS"])


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_name,
        device=None,
        cache_folder=None,
        local_files_only=False,
        dimensions=None,
    ):
        self.model = get_sentence_transformer_class()(
            model_name,
            device=device,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
            truncate_dim=dimensions,
        )

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings_model():
    provider = os.getenv("EMBEDDINGS_PROVIDER") or os.getenv("LLM_PROVIDER", "azure")
    provider = provider.lower()

    if provider == "azure":
        kwargs = {
            "azure_deployment": os.environ["OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        }
        dimensions = get_embedding_dimensions()
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        if os.getenv("OPENAI_API_VERSION"):
            kwargs["api_version"] = os.environ["OPENAI_API_VERSION"]
        if os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE"):
            kwargs["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv(
                "OPENAI_API_BASE"
            )
        if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv(
                "OPENAI_API_KEY"
            )
        return get_azure_openai_embeddings_class()(**kwargs)

    if provider == "openai":
        kwargs = {
            "model": os.environ["OPENAI_EMBEDDING_MODEL"],
        }
        dimensions = get_embedding_dimensions()
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        if os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        return get_openai_embeddings_class()(**kwargs)

    if provider == "vertexai":
        kwargs = {
            "model": os.environ["VERTEXAI_EMBEDDING_MODEL"],
        }
        dimensions = get_embedding_dimensions()
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        if os.getenv("VERTEXAI_PROJECT"):
            kwargs["project"] = os.environ["VERTEXAI_PROJECT"]
        if os.getenv("VERTEXAI_LOCATION"):
            kwargs["location"] = os.environ["VERTEXAI_LOCATION"]
        return get_vertexai_embeddings_class()(**kwargs)

    if provider == "local":
        return LocalSentenceTransformerEmbeddings(
            model_name=os.environ["LOCAL_EMBEDDING_MODEL"],
            device=os.getenv("LOCAL_EMBEDDING_DEVICE"),
            cache_folder=os.getenv("LOCAL_EMBEDDING_CACHE_FOLDER"),
            local_files_only=os.getenv("LOCAL_EMBEDDING_LOCAL_FILES_ONLY", "False")
            == "True",
            dimensions=get_embedding_dimensions(),
        )

    raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {provider}")
