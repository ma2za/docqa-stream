import os


def get_azure_chat_openai_class():
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI


def get_chat_openai_class():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI


def get_chat_vertexai_class():
    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI


def get_chat_model(temperature):
    provider = os.getenv("LLM_PROVIDER", "azure").lower()

    if provider == "azure":
        kwargs = {
            "azure_deployment": os.environ["OPENAI_DEPLOYMENT_NAME"],
            "temperature": temperature,
            "streaming": True,
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
        return get_azure_chat_openai_class()(**kwargs)

    if provider == "openai":
        kwargs = {
            "model": os.environ["OPENAI_MODEL"],
            "temperature": temperature,
            "streaming": True,
        }
        if os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        return get_chat_openai_class()(**kwargs)

    if provider == "vertexai":
        kwargs = {
            "model": os.environ["VERTEXAI_MODEL"],
            "temperature": temperature,
            "streaming": True,
        }
        if os.getenv("VERTEXAI_PROJECT"):
            kwargs["project"] = os.environ["VERTEXAI_PROJECT"]
        if os.getenv("VERTEXAI_LOCATION"):
            kwargs["location"] = os.environ["VERTEXAI_LOCATION"]
        return get_chat_vertexai_class()(**kwargs)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
