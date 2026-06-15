import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse

from .routers import files

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(files.router)
DEMO_PATH = Path(__file__).parent / "static" / "demo.html"


@app.get("/")
async def root():
    return HTMLResponse(DEMO_PATH.read_text(encoding="utf-8"))


@app.get("/demo")
async def demo():
    return HTMLResponse(DEMO_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"message": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["FASTAPI_PORT"]))
