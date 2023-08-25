import weaviate


def create_classes(client: weaviate.Client, drop: bool = False):
    # client.schema.delete_all()
    class_name = "Document"
    schema = {
        "class": class_name,
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": "false"
            }
        },
        "properties": [
            {
                "name": "title",
                "dataType": ["text"]
            },
            {
                "name": "content",
                "dataType": ["text"]
            },
        ],

    }
    if not client.schema.exists(class_name) or drop:
        client.schema.create_class(schema)
