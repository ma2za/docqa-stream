from typing import Annotated

from fastapi import APIRouter, Depends, Query, UploadFile
from starlette.responses import StreamingResponse

from ..services.files import FilesService
from ..utils.store import get_store

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/query")
async def query(
    question: str,
    temperature: Annotated[float, Query(ge=0, le=2)] = 0.7,
    n_docs: Annotated[int, Query(ge=1, le=50)] = 10,
    vectorstore=Depends(get_store),
):
    response = await FilesService.query(question, temperature, n_docs, vectorstore)
    return StreamingResponse(response, media_type="text/event-stream")


@router.post("/upload")
async def upload(
    file: UploadFile,
    chunk_size: Annotated[int, Query(ge=100, le=4000)] = 200,
    vectorstore=Depends(get_store),
):
    return await FilesService.upload(file, chunk_size, vectorstore)


@router.get("")
async def list_files(vectorstore=Depends(get_store)):
    return {"documents": await FilesService.list(vectorstore)}


@router.delete("/{document_id}")
async def delete_file(document_id: str, vectorstore=Depends(get_store)):
    return await FilesService.delete(document_id, vectorstore)
