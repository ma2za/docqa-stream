import weaviate


def create_class(
    client: weaviate.Client, drop: bool = False, class_name: str = "Document"
):
    if drop:
        client.schema.delete_class(class_name)
    schema = {
        "class": class_name,
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {"text2vec-transformers": {"vectorizeClassName": "false"}},
        "properties": [
            {"name": "title", "dataType": ["text"]},
            {"name": "content", "dataType": ["text"]},
        ],
    }
    if not client.schema.exists(class_name):
        client.schema.create_class(schema)
