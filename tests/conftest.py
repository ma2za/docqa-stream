import os

import pytest
import yaml
from fastapi.testclient import TestClient

os.environ.setdefault("WEAVIATE_SERVICE_NAME", "weaviate")
os.environ.setdefault("WEAVIATE_PORT", "8080")
os.environ.setdefault("LLM_PROVIDER", "azure")
os.environ.setdefault("OPENAI_DEPLOYMENT_NAME", "chat-test")
os.environ.setdefault("OPENAI_API_BASE", "https://example.openai.azure.com")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from src.docqa_stream.server import app


@pytest.fixture(scope="session")
def test_client():
    client = TestClient(app)
    yield client


@pytest.fixture(autouse=False)
def test_files():
    with open("tests/data/test_files.yml", "r") as f:
        output = yaml.safe_load(f)
    yield output
