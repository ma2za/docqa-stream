import io
import logging
import os

import uvicorn
import weaviate
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from langchain.chains import RetrievalQA
from langchain.vectorstores import Weaviate
from unstructured.partition.pdf import partition_pdf

from src.docqa_stream.utils.llm import get_model

load_dotenv()

app = FastAPI()

client = weaviate.Client(os.environ["WEAVIATE_URL"])
vectorstore = Weaviate(client=client, index_name="Document", text_key="content")
logger = logging.getLogger(__name__)
llm = get_model()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/query")
async def query(question: str):
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    return qa_chain({"query": question})


@app.post("/upload")
async def create_upload_file(file: UploadFile):
    data = await file.read()
    elements = partition_pdf(file=io.BytesIO(data))
    text = [ele.text for ele in elements]
    vectorstore.add_texts(["\n".join(text)])
    # data_object = {
    #     'title': file.filename,
    #     'content': "\n".join(text),
    # }
    #
    # uuid = generate_uuid5(data_object)
    # existing_data_object = client.data_object.get_by_id(
    #     uuid,
    #     class_name='Document',
    # )
    # if existing_data_object is not None:
    #     return existing_data_object
    #
    # uuid = client.data_object.create(
    #     data_object=data_object,
    #     class_name='Document',
    #     uuid=uuid,
    # )
    out = vectorstore.search("paolo", "similarity")

    return {"out": out}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
