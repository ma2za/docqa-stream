import os

import weaviate
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.config import Configure, DataType, Property


def create_class(client, drop: bool = False, class_name: str = "Document"):
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
            ],
        )


def get_store():
    weaviate_client = weaviate.connect_to_custom(
        http_host=os.environ["WEAVIATE_SERVICE_NAME"],
        http_port=int(os.environ["WEAVIATE_PORT"]),
        http_secure=False,
        grpc_host=os.environ["WEAVIATE_SERVICE_NAME"],
        grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=False,
    )

    create_class(
        weaviate_client,
        os.environ.get("WEAVIATE_DROP_COLLECTION", "False") == "True",
        os.environ.get("WEAVIATE_COLLECTION", "Document"),
    )

    vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        index_name=os.environ.get("WEAVIATE_COLLECTION", "Document"),
        text_key="content",
    )

    try:
        yield vectorstore
    finally:
        weaviate_client.close()
