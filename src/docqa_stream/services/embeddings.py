import os


def get_azure_openai_embeddings_class():
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings


def get_openai_embeddings_class():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings


def get_vertexai_embeddings_class():
    from langchain_google_vertexai import VertexAIEmbeddings

    return VertexAIEmbeddings


def get_embeddings_model():
    provider = os.getenv("EMBEDDINGS_PROVIDER") or os.getenv("LLM_PROVIDER", "azure")
    provider = provider.lower()

    if provider == "azure":
        kwargs = {
            "azure_deployment": os.environ["OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        }
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
        if os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        return get_openai_embeddings_class()(**kwargs)

    if provider == "vertexai":
        kwargs = {
            "model": os.environ["VERTEXAI_EMBEDDING_MODEL"],
        }
        if os.getenv("VERTEXAI_PROJECT"):
            kwargs["project"] = os.environ["VERTEXAI_PROJECT"]
        if os.getenv("VERTEXAI_LOCATION"):
            kwargs["location"] = os.environ["VERTEXAI_LOCATION"]
        return get_vertexai_embeddings_class()(**kwargs)

    raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {provider}")
