import os

from fastapi.testclient import TestClient

from src.docqa_stream.server import app, weaviate_client
from src.docqa_stream.utils.schema import create_class

test_client = TestClient(app)


def test_health():
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}


def test_upload():
    create_class(
        weaviate_client,
        os.environ.get("WEAVIATE_DROP_COLLECTION", True),
        os.environ.get("WEAVIATE_COLLECTION", "Document"),
    )

    response = test_client.post(
        "/upload",
        files={"file": ("rome_guide.pdf", open("tests/data/rome_guide.pdf", "rb"))},
        params={"chunk_size": 1000},
    )
    assert response.status_code == 200
    uuids = response.json().get("response")
    assert isinstance(uuids, list) and len(uuids) == 73


def test_query():
    _ = test_client.post(
        "/upload",
        files={"file": ("rome_guide.pdf", open("tests/data/rome_guide.pdf", "rb"))},
    )
    response = test_client.get(
        "/query",
        params={
            "question": "Be concise, can you give me a food suggestion?",
            "temperature": 1,
            "n_docs": 1,
        },
    )
    assert response.status_code == 200
    assert (
            response.text
            == "Paolo Mazza's email address is mazzapaolo2019@gmail.com and his phone number is +393518339474."
    )
