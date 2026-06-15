from types import SimpleNamespace

from langchain_core.documents import Document

from src.docqa_stream.utils.store import PGVectorDocumentStore, WeaviateDocumentStore


class FakeQuery:
    def __init__(self, objects):
        self.objects = objects
        self.filters = None

    def fetch_objects(self, **kwargs):
        self.filters = kwargs.get("filters")
        limit = kwargs.get("limit", len(self.objects))
        return SimpleNamespace(objects=self.objects[:limit])


class FakeData:
    def __init__(self, successful):
        self.successful = successful
        self.where = None
        self.verbose = None

    def delete_many(self, where, verbose=False):
        self.where = where
        self.verbose = verbose
        return SimpleNamespace(results=SimpleNamespace(successful=self.successful))


class FakeCollections:
    def __init__(self, collection):
        self.collection = collection

    def use(self, name):
        self.name = name
        return self.collection


def test_list_documents_groups_chunks_by_document_id():
    objects = [
        SimpleNamespace(
            properties={
                "document_id": "doc-1",
                "filename": "rome.pdf",
                "page": 2,
            }
        ),
        SimpleNamespace(
            properties={
                "document_id": "doc-1",
                "filename": "rome.pdf",
                "page": 1,
            }
        ),
        SimpleNamespace(
            properties={
                "document_id": "doc-2",
                "filename": "athens.pdf",
                "page": None,
            }
        ),
    ]
    collection = SimpleNamespace(query=FakeQuery(objects))
    client = SimpleNamespace(collections=FakeCollections(collection))
    store = WeaviateDocumentStore(client, "Document", None)

    assert store.list_documents() == [
        {
            "document_id": "doc-2",
            "filename": "athens.pdf",
            "chunks": 1,
            "pages": [],
        },
        {
            "document_id": "doc-1",
            "filename": "rome.pdf",
            "chunks": 2,
            "pages": [1, 2],
        },
    ]


def test_delete_document_returns_successful_delete_count():
    collection = SimpleNamespace(
        query=FakeQuery([SimpleNamespace(properties={"document_id": "doc-1"})]),
        data=FakeData(3),
    )
    client = SimpleNamespace(collections=FakeCollections(collection))
    store = WeaviateDocumentStore(client, "Document", None)

    assert store.delete_document("doc-1") == 3
    assert collection.data.verbose is True


def test_delete_document_returns_zero_when_missing():
    collection = SimpleNamespace(query=FakeQuery([]), data=FakeData(0))
    client = SimpleNamespace(collections=FakeCollections(collection))
    store = WeaviateDocumentStore(client, "Document", None)

    assert store.delete_document("missing") == 0


class FakePGVector:
    def add_documents(self, docs, ids):
        self.docs = docs
        self.ids = ids
        return ids


def test_pgvector_add_documents_uses_document_chunk_ids():
    vectorstore = FakePGVector()
    store = PGVectorDocumentStore(vectorstore, stores=(None, None))
    docs = [
        Document(
            page_content="First chunk",
            metadata={"document_id": "doc-1", "chunk_index": 0},
        ),
        Document(
            page_content="Second chunk",
            metadata={"document_id": "doc-1", "chunk_index": 1},
        ),
    ]

    assert store.add_documents(docs) == ["doc-1:0", "doc-1:1"]
    assert vectorstore.ids == ["doc-1:0", "doc-1:1"]
