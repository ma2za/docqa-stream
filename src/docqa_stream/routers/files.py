from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    UploadFile,
)
from starlette.responses import StreamingResponse

from .. import settings
from ..services.files import FilesService
from ..utils.store import get_store

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/query")
async def query(
    question: Annotated[
        str,
        Query(min_length=1, max_length=settings.MAX_QUESTION_CHARS),
    ],
    temperature: Annotated[float, Query(ge=0, le=2)] = 0.7,
    n_docs: Annotated[int, Query(ge=1, le=50)] = 10,
    vectorstore=Depends(get_store),
):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    response = await FilesService.query(question, temperature, n_docs, vectorstore)
    return StreamingResponse(response, media_type="text/event-stream")


@router.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    chunk_size: Annotated[int, Query(ge=100, le=4000)] = 200,
):
    return await FilesService.upload(file, chunk_size, background_tasks, get_store)


@router.get("/uploads/{document_id}")
async def upload_status(document_id: str):
    return FilesService.upload_status(document_id)


@router.get("")
async def list_files(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    vectorstore=Depends(get_store),
):
    return await FilesService.list(vectorstore, limit, offset)


@router.delete("/{document_id}")
async def delete_file(document_id: str, vectorstore=Depends(get_store)):
    return await FilesService.delete(document_id, vectorstore)
