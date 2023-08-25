import io
import logging
import os

import uvicorn
import weaviate
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from langchain import LLMChain
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from starlette.responses import StreamingResponse
from unstructured.partition.pdf import partition_pdf

from src.docqa_stream.utils.schema import create_classes

load_dotenv()

app = FastAPI()

client = weaviate.Client(os.environ["WEAVIATE_URL"])

create_classes(client)

vectorstore = Weaviate(client=client, index_name="Document", text_key="content")
logger = logging.getLogger(__name__)


@app.get("/health")
async def health():
    return {"message": "OK"}


@app.get("/query")
async def query(question: str, temperature: int = 0.7, n_docs: int = 10):
    llm = AzureChatOpenAI(deployment_name="finetuner", streaming=True, temperature=temperature)
    messages = [
        SystemMessage(
            content=(
                "You are a world class algorithm to answer "
                "questions in a specific format."
            )
        ),
        HumanMessage(content="Answer question using the following context"),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt = ChatPromptTemplate(messages=messages)

    qa_chain = LLMChain(llm=llm, prompt=prompt)
    doc_prompt = PromptTemplate(
        template="Content: {page_content}",
        input_variables=["page_content"],
    )
    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context",
        document_prompt=doc_prompt,
    )
    retrieval_qa = RetrievalQA(
        retriever=vectorstore.as_retriever(search_kwargs={"k": n_docs}), combine_documents_chain=final_qa_chain
    )

    def openai_streamer(retr_qa: RetrievalQA, text: str):
        for resp in retr_qa.run(text):
            yield resp

    return StreamingResponse(openai_streamer(retrieval_qa, question), media_type='text/event-stream')


@app.post("/upload")
async def create_upload_file(file: UploadFile, chunk_size: int = 200):
    data = await file.read()
    elements = partition_pdf(file=io.BytesIO(data))
    text = [ele.text for ele in elements]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )

    docs = text_splitter.create_documents(["\n".join(text)])
    response = vectorstore.add_documents(docs)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["FASTAPI_PORT"]))
