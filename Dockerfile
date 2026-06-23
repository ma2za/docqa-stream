# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim AS base
ARG INSTALL_EXTRAS=""

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
        libxcb1 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

ENV POETRY_HOME=/opt/poetry \
    POETRY_VENV=/opt/poetry-venv \
    POETRY_CACHE_DIR=/opt/.cache

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==2.3.2

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    if [ -n "$INSTALL_EXTRAS" ]; then \
        poetry install --without dev --extras "$INSTALL_EXTRAS" --no-root; \
    else \
        poetry install --without dev --no-root; \
    fi \
    && poetry run python -m pip install --force-reinstall "protobuf==6.33.6"

COPY . .

CMD ["sh", "-c", "poetry run uvicorn src.docqa_stream.server:app --host 0.0.0.0 --port ${FASTAPI_PORT:-8000}"]
