import pytest
import yaml
from fastapi.testclient import TestClient

from src.docqa_stream.server import app


@pytest.fixture(scope="session")
def test_client():
    client = TestClient(app)
    yield client


@pytest.fixture(autouse=False)
def test_files():
    with open('tests/data/test_files.yml', 'r') as f:
        output = yaml.safe_load(f)
    yield output
