import pytest


@pytest.mark.usefixtures("test_files")
def test_upload(test_client, test_files, monkeypatch):
    monkeypatch.setenv("WEAVIATE_COLLECTION", "Test_Document")
    monkeypatch.setenv("WEAVIATE_DROP_COLLECTION", "True")

    for test_example in test_files.get("test_upload"):
        inputs = test_example.get("inputs")
        outputs = test_example.get("outputs")

        response = test_client.post(
            "/files/upload",
            files={"file": (inputs.get("filename"), open(inputs.get("file"), "rb"))},
            params={"chunk_size": inputs.get("chunk_size")},
        )
        assert response.status_code == 200
        uuids = response.json().get("response")
        assert isinstance(uuids, list) and len(uuids) == outputs.get("length")


def test_query(test_client, test_files, monkeypatch):
    monkeypatch.setenv("WEAVIATE_COLLECTION", "Test_Document")
    monkeypatch.setenv("WEAVIATE_DROP_COLLECTION", "True")

    for test_example in test_files.get("test_query"):
        inputs = test_example.get("inputs")
        outputs = test_example.get("outputs")

        _ = test_client.post(
            "/files/upload",
            files={"file": (inputs.get("filename"), open(inputs.get("file"), "rb"))},
            params={"chunk_size": inputs.get("chunk_size")},
        )
        response = test_client.get(
            "/files/query",
            params={
                "question": inputs.get("question"),
                "temperature": inputs.get("temperature"),
                "n_docs": inputs.get("n_docs"),
            },
        )
        assert response.status_code == 200
        assert response.text == outputs.get("response")
