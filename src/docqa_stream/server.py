import io

from fastapi import FastAPI, UploadFile
from unstructured.partition.pdf import partition_pdf

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    data = await file.read()
    elements = partition_pdf(file=io.BytesIO(data))
    titles = [elem for elem in elements if elem.category == "Title"]
    return {"filename": titles}
