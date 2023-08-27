def test_health(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}


def test_upload(test_client, monkeypatch):
    monkeypatch.setenv("WEAVIATE_COLLECTION", "Test_Document")
    monkeypatch.setenv("WEAVIATE_DROP_COLLECTION", "True")

    response = test_client.post(
        "/upload",
        files={"file": ("rome_guide.pdf", open("tests/data/rome_guide.pdf", "rb"))},
        params={"chunk_size": 1000},
    )
    assert response.status_code == 200
    uuids = response.json().get("response")
    assert isinstance(uuids, list) and len(uuids) == 73


def test_query(test_client, monkeypatch):
    monkeypatch.setenv("WEAVIATE_COLLECTION", "Test_Document")
    monkeypatch.setenv("WEAVIATE_DROP_COLLECTION", "True")

    _ = test_client.post(
        "/upload",
        files={"file": ("rome_guide.pdf", open("tests/data/rome_guide.pdf", "rb"))},
        params={"chunk_size": 500}
    )
    response = test_client.get(
        "/query",
        params={
            "question": "when was rome founded?",
            "temperature": 1,
            "n_docs": 3,
        },
    )
    assert response.status_code == 200
    assert (
            response.text
            == "Sure! My mad urges for eating whatever I want at dinnertime suggest that I try a delicious and juicy steak tonight. How does that sound?"
    )
