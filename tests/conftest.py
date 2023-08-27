import pytest
from fastapi.testclient import TestClient

from src.docqa_stream.server import app


@pytest.fixture(scope="session")
def test_client():
    client = TestClient(app)
    yield client
