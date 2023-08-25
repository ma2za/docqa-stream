from fastapi.testclient import TestClient

from src.docqa_stream.server import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}


def test_upload():
    # TODO complete
    response = client.get("/upload")
    assert response.status_code == 200


def test_query():
    # TODO complete
    response = client.get("/query")
    assert response.status_code == 200
