import os

from ..services.embeddings import get_embeddings_model
from .. import settings


def create_class(client, drop: bool = False, class_name: str = "Document"):
    from weaviate.classes.config import Configure, DataType, Property

    if drop:
        client.collections.delete(class_name)
    if not client.collections.exists(class_name):
        client.collections.create(
            name=class_name,
            vector_config=Configure.Vectors.text2vec_transformers(
                vectorize_collection_name=False
            ),
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )


class WeaviateDocumentStore:
    def __init__(self, client, collection_name, vectorstore, filter_builder=None):
        self.client = client
        self.collection_name = collection_name
        self.vectorstore = vectorstore
        self.filter_builder = filter_builder or get_weaviate_document_filter

    def add_documents(self, docs):
        return self.vectorstore.add_documents(docs)

    def similarity_search_with_score(self, query, k):
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def list_documents(self, limit=50, offset=0):
        collection = self.client.collections.use(self.collection_name)
        response = collection.query.fetch_objects(
            limit=settings.LIST_DOCUMENT_CHUNK_SCAN_LIMIT
        )
        documents = {}

        for obj in response.objects:
            props = obj.properties
            document_id = props.get("document_id")
            if not document_id:
                continue

            document = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "filename": props.get("filename"),
                    "chunks": 0,
                    "pages": set(),
                },
            )
            document["chunks"] += 1
            if props.get("page") is not None:
                document["pages"].add(props.get("page"))

        output = []
        for document in documents.values():
            output.append(
                {
                    "document_id": document["document_id"],
                    "filename": document["filename"],
                    "chunks": document["chunks"],
                    "pages": sorted(document["pages"]),
                }
            )
        return sorted(output, key=lambda item: item["filename"] or "")[
            offset : offset + limit
        ]

    def delete_document(self, document_id):
        collection = self.client.collections.use(self.collection_name)
        where = self.filter_builder(document_id)
        matched = collection.query.fetch_objects(filters=where, limit=1)
        if not matched.objects:
            return 0

        result = collection.data.delete_many(where=where, verbose=True)
        results = getattr(result, "results", None)
        successful = getattr(results, "successful", None)
        return successful if successful is not None else 1


class PGVectorDocumentStore:
    def __init__(self, vectorstore, stores=None):
        self.vectorstore = vectorstore
        self.EmbeddingStore, self.CollectionStore = stores or get_pgvector_tables()

    def add_documents(self, docs):
        ids = [
            f"{doc.metadata['document_id']}:{doc.metadata['chunk_index']}"
            for doc in docs
        ]
        return self.vectorstore.add_documents(docs, ids=ids)

    def similarity_search_with_score(self, query, k):
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def _collection(self, session):
        return self.CollectionStore.get_by_name(session, self.vectorstore.collection_name)

    def list_documents(self, limit=50, offset=0):
        from sqlalchemy import select

        with self.vectorstore.session_maker() as session:
            collection = self._collection(session)
            if collection is None:
                return []

            rows = session.execute(
                select(self.EmbeddingStore.cmetadata).where(
                    self.EmbeddingStore.collection_id == collection.uuid
                ).limit(settings.LIST_DOCUMENT_CHUNK_SCAN_LIMIT)
            ).all()

        documents = {}
        for row in rows:
            metadata = row[0] or {}
            document_id = metadata.get("document_id")
            if not document_id:
                continue

            document = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "filename": metadata.get("filename"),
                    "chunks": 0,
                    "pages": set(),
                },
            )
            document["chunks"] += 1
            if metadata.get("page") is not None:
                document["pages"].add(metadata.get("page"))

        output = []
        for document in documents.values():
            output.append(
                {
                    "document_id": document["document_id"],
                    "filename": document["filename"],
                    "chunks": document["chunks"],
                    "pages": sorted(document["pages"]),
                }
            )
        return sorted(output, key=lambda item: item["filename"] or "")[
            offset : offset + limit
        ]

    def delete_document(self, document_id):
        from sqlalchemy import delete

        with self.vectorstore.session_maker() as session:
            collection = self._collection(session)
            if collection is None:
                return 0

            result = session.execute(
                delete(self.EmbeddingStore).where(
                    self.EmbeddingStore.collection_id == collection.uuid,
                    self.EmbeddingStore.cmetadata["document_id"].astext == document_id,
                )
            )
            session.commit()
            return result.rowcount or 0


def get_weaviate_document_filter(document_id):
    from weaviate.classes.query import Filter

    return Filter.by_property("document_id").equal(document_id)


def get_pgvector_tables():
    from langchain_postgres.vectorstores import _get_embedding_collection_store

    return _get_embedding_collection_store()


def get_weaviate_store():
    import weaviate
    from langchain_weaviate import WeaviateVectorStore

    client = weaviate.connect_to_custom(
        http_host=os.environ["WEAVIATE_SERVICE_NAME"],
        http_port=int(os.environ["WEAVIATE_PORT"]),
        http_secure=False,
        grpc_host=os.environ["WEAVIATE_SERVICE_NAME"],
        grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=False,
    )

    create_class(
        client,
        os.environ.get("WEAVIATE_DROP_COLLECTION", "False") == "True",
        os.environ.get("WEAVIATE_COLLECTION", "Document"),
    )
    collection_name = os.environ.get("WEAVIATE_COLLECTION", "Document")

    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="content",
    )

    try:
        yield WeaviateDocumentStore(client, collection_name, vectorstore)
    finally:
        client.close()


def get_pgvector_store():
    from langchain_postgres import PGVector

    embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    vectorstore = PGVector(
        embeddings=get_embeddings_model(),
        collection_name=os.environ.get("PGVECTOR_COLLECTION", "docqa_stream"),
        connection=os.environ["PGVECTOR_CONNECTION"],
        embedding_length=int(embedding_dimensions) if embedding_dimensions else None,
        use_jsonb=True,
        pre_delete_collection=os.environ.get("PGVECTOR_DROP_COLLECTION", "False")
        == "True",
    )
    yield PGVectorDocumentStore(vectorstore)


def get_store():
    backend = os.getenv("VECTOR_STORE", "weaviate").lower()

    if backend == "weaviate":
        yield from get_weaviate_store()
        return

    if backend == "pgvector":
        yield from get_pgvector_store()
        return

    raise ValueError(f"Unsupported VECTOR_STORE: {backend}")
