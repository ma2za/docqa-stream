import logging
import os

import uvicorn
from fastapi import FastAPI

from .routers import files

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(files.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"message": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["FASTAPI_PORT"]))
