import pytest

from src.docqa_stream.routers.files import get_store
from src.docqa_stream.server import app
from src.docqa_stream.services.files import FilesService


class FakeVectorStore:
    pass


def get_fake_store():
    yield FakeVectorStore()


@pytest.fixture(autouse=True)
def override_store():
    app.dependency_overrides[get_store] = get_fake_store
    yield
    app.dependency_overrides.clear()


@pytest.mark.usefixtures("test_files")
def test_upload(test_client, test_files, monkeypatch):
    for test_example in test_files.get("test_upload"):
        inputs = test_example.get("inputs")
        outputs = test_example.get("outputs")

        async def upload(file, chunk_size, vectorstore):
            assert file.filename == inputs.get("filename")
            assert chunk_size == inputs.get("chunk_size")
            assert isinstance(vectorstore, FakeVectorStore)
            return ["uuid"] * outputs.get("length")

        monkeypatch.setattr(FilesService, "upload", upload)

        with open(inputs.get("file"), "rb") as file:
            response = test_client.post(
                "/files/upload",
                files={"file": (inputs.get("filename"), file)},
                params={"chunk_size": inputs.get("chunk_size")},
            )
        assert response.status_code == 200
        uuids = response.json().get("response")
        assert isinstance(uuids, list) and len(uuids) == outputs.get("length")


def test_query(test_client, test_files, monkeypatch):
    for test_example in test_files.get("test_query"):
        inputs = test_example.get("inputs")
        outputs = test_example.get("outputs")

        async def query(question, temperature, n_docs, vectorstore):
            assert question == inputs.get("question")
            assert temperature == inputs.get("temperature")
            assert n_docs == inputs.get("n_docs")
            assert isinstance(vectorstore, FakeVectorStore)
            return iter([outputs.get("response")])

        monkeypatch.setattr(FilesService, "query", query)

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
