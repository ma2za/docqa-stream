import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.docqa_stream.routers.files import get_store
from src.docqa_stream.server import app
from src.docqa_stream.services.files import FilesService


class FakeVectorStore:
    def __init__(self):
        self.docs = []
        self.deleted = {}
        self.documents = [
            {
                "document_id": "doc-1",
                "filename": "rome_guide.pdf",
                "chunks": 2,
                "pages": [1],
            }
        ]

    def add_documents(self, docs):
        self.docs.extend(docs)
        return ["uuid"] * len(docs)

    def similarity_search_with_score(self, query, k):
        self.query = query
        self.k = k
        return [
            (
                Document(
                    page_content="Rome was founded in 753 BC according to tradition.",
                    metadata={
                        "filename": "rome_guide.pdf",
                        "page": 1,
                        "chunk_index": 0,
                    },
                ),
                0.82,
            )
        ]

    def list_documents(self, limit=50, offset=0):
        self.list_limit = limit
        self.list_offset = offset
        return self.documents[offset : offset + limit]

    def delete_document(self, document_id):
        return self.deleted.get(document_id, 0)


@pytest.fixture()
def fake_store():
    return FakeVectorStore()


@pytest.fixture(autouse=True)
def override_store(fake_store):
    def get_fake_store():
        yield fake_store

    app.dependency_overrides[get_store] = get_fake_store
    yield
    app.dependency_overrides.clear()


class FakeUpload:
    def __init__(self, filename, data):
        self.filename = filename
        self.data = data

    async def read(self, size=-1):
        if size == -1:
            output = self.data
            self.data = b""
            return output
        output = self.data[:size]
        self.data = self.data[size:]
        return output


def test_upload_api_returns_document_summary(test_client, test_files, monkeypatch):
    test_example = test_files.get("test_upload")[0]
    inputs = test_example.get("inputs")

    async def upload(file, chunk_size, vectorstore):
        assert file.filename == inputs.get("filename")
        assert chunk_size == inputs.get("chunk_size")
        assert isinstance(vectorstore, FakeVectorStore)
        return {
            "document_id": "doc-1",
            "filename": file.filename,
            "chunks_added": 73,
        }

    monkeypatch.setattr(FilesService, "upload", upload)

    with open(inputs.get("file"), "rb") as file:
        response = test_client.post(
            "/files/upload",
            files={"file": (inputs.get("filename"), file)},
            params={"chunk_size": inputs.get("chunk_size")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-1",
        "filename": inputs.get("filename"),
        "chunks_added": 73,
    }


def test_upload_rejects_non_pdf(test_client):
    response = test_client.post(
        "/files/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        params={"chunk_size": 1000},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF uploads are supported."


def test_upload_rejects_empty_pdf(test_client):
    response = test_client.post(
        "/files/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        params={"chunk_size": 1000},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_upload_rejects_oversized_pdf(fake_store, monkeypatch):
    monkeypatch.setattr("src.docqa_stream.settings.MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr("src.docqa_stream.settings.UPLOAD_READ_CHUNK_BYTES", 2)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            FilesService.upload(
                FakeUpload("large.pdf", b"12345"),
                1000,
                fake_store,
            )
        )

    assert exc_info.value.status_code == 413


def test_upload_stores_document_metadata(fake_store, monkeypatch):
    element = SimpleNamespace(
        text="Rome was founded in 753 BC according to tradition.",
        metadata=SimpleNamespace(page_number=1),
    )
    monkeypatch.setattr(
        "src.docqa_stream.services.files.partition_pdf_file",
        lambda file: [element],
    )

    response = asyncio.run(
        FilesService.upload(
            FakeUpload("rome_guide.pdf", b"%PDF-1.4"),
            1000,
            fake_store,
        )
    )

    assert response["filename"] == "rome_guide.pdf"
    assert response["chunks_added"] == 1
    assert len(response["document_id"]) == 36
    assert fake_store.docs[0].metadata["document_id"] == response["document_id"]
    assert fake_store.docs[0].metadata["filename"] == "rome_guide.pdf"
    assert fake_store.docs[0].metadata["page"] == 1
    assert fake_store.docs[0].metadata["chunk_index"] == 0


def test_query_api_streams_answer_and_citations(test_client, test_files, monkeypatch):
    test_example = test_files.get("test_query")[0]
    inputs = test_example.get("inputs")

    async def query(question, temperature, n_docs, vectorstore):
        assert question == inputs.get("question")
        assert temperature == inputs.get("temperature")
        assert n_docs == inputs.get("n_docs")
        assert isinstance(vectorstore, FakeVectorStore)
        return iter(
            [
                FilesService._event("token", {"text": "Rome was founded in 753 BC."}),
                FilesService._event(
                    "citations",
                    {
                        "citations": [
                            {
                                "filename": "rome_guide.pdf",
                                "page": 1,
                                "chunk_index": 0,
                                "score": 0.82,
                                "preview": "Rome was founded in 753 BC.",
                            }
                        ]
                    },
                ),
            ]
        )

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
    assert "event: token" in response.text
    assert "Rome was founded in 753 BC." in response.text
    assert "event: citations" in response.text
    assert "rome_guide.pdf" in response.text


def test_query_uses_bounded_inputs_and_returns_citations(fake_store, monkeypatch):
    class FakeLLM:
        def stream(self, messages):
            self.messages = messages
            return iter([SimpleNamespace(content="Rome"), SimpleNamespace(content=".")])

    llm = FakeLLM()
    monkeypatch.setattr(
        "src.docqa_stream.services.files.get_chat_model",
        lambda temperature: llm,
    )

    stream = asyncio.run(
        FilesService.query("when was rome founded?", 0, 3, fake_store)
    )
    body = "".join(stream)
    events = [event for event in body.split("\n\n") if event]
    citations_payload = json.loads(events[-1].split("data: ", 1)[1])

    assert fake_store.query == "when was rome founded?"
    assert fake_store.k == 3
    assert "Source 1:" in llm.messages[1][1]
    assert "event: token" in body
    assert citations_payload["citations"][0]["filename"] == "rome_guide.pdf"
    assert citations_payload["citations"][0]["page"] == 1
    assert citations_payload["citations"][0]["chunk_index"] == 0


def test_list_files(test_client):
    response = test_client.get("/files")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {
                "document_id": "doc-1",
                "filename": "rome_guide.pdf",
                "chunks": 2,
                "pages": [1],
            }
        ],
        "limit": 50,
        "offset": 0,
        "count": 1,
    }


def test_delete_file(test_client, fake_store):
    fake_store.deleted["doc-1"] = 2

    response = test_client.delete("/files/doc-1")

    assert response.status_code == 200
    assert response.json() == {"document_id": "doc-1", "chunks_deleted": 2}


def test_delete_missing_file_returns_404(test_client):
    response = test_client.delete("/files/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."
