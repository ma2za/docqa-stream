from fastapi import APIRouter, Depends, UploadFile
from starlette.responses import StreamingResponse

from ..services.files import FilesService
from ..utils.store import get_store

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/query")
async def query(
    question: str,
    temperature: float = 0.7,
    n_docs: int = 10,
    vectorstore=Depends(get_store),
):
    response = await FilesService.query(question, temperature, n_docs, vectorstore)
    return StreamingResponse(response, media_type="text/event-stream")


@router.post("/upload")
async def upload(
    file: UploadFile, chunk_size: int = 200, vectorstore=Depends(get_store)
):
    response = await FilesService.upload(file, chunk_size, vectorstore)
    return {"response": response}
