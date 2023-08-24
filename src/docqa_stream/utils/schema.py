import os

import weaviate
from dotenv import load_dotenv

load_dotenv()

client = weaviate.Client(os.environ["WEAVIATE_URL"])

client.schema.delete_all()

schema = {
    "class": "Document",
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

client.schema.create_class(schema)
