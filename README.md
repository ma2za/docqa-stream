# DocQA Stream

Self-hosted PDF Q&A API with FastAPI, Weaviate or pgvector, streaming responses, and Azure OpenAI, OpenAI-compatible models, or Vertex AI.

DocQA Stream is a small RAG starter for uploading PDFs, indexing them in a local vector store, and asking questions over the indexed content. Answers stream back over server-sent events and finish with citation metadata.

## Architecture

```text
Browser or curl
  -> FastAPI
  -> Unstructured PDF parsing
  -> LangChain text splitting
  -> Weaviate text2vec-transformers or PostgreSQL pgvector
  -> Azure OpenAI, OpenAI-compatible, or Vertex AI chat model
```

## Features

- Docker Compose setup for FastAPI, Weaviate, local transformer embeddings, and optional pgvector.
- PDF upload with chunk metadata: document ID, filename, page, and chunk index.
- Streaming Q&A endpoint with citation metadata.
- Document list and delete endpoints.
- Minimal browser demo at `http://localhost:8000`.
- Azure OpenAI by default, with OpenAI-compatible and Vertex AI provider options.
- `VECTOR_STORE=weaviate` by default, or `VECTOR_STORE=pgvector` for PostgreSQL-backed vectors.

## Quickstart

### 1. Prerequisites

- Docker and Docker Compose
- Azure OpenAI, OpenAI-compatible, or Vertex AI model credentials

### 2. Install options

For local development with the default Weaviate and Azure/OpenAI-compatible path:

```shell
poetry install --extras "weaviate azure"
```

For pgvector and Vertex AI:

```shell
poetry install --extras "pgvector vertexai"
```

For all supported backends and model providers:

```shell
poetry install --extras all
```

### 3. Configure environment

```shell
cp .env.example .env
```

Fill in these values in `.env` for the default Weaviate backend.

```text
OPENAI_DEPLOYMENT_NAME=
OPENAI_API_KEY=
OPENAI_API_BASE=
```

The default `.env.example` values run FastAPI on `8000`, Weaviate on `8080`, and Weaviate gRPC on `50051`.

For pgvector, also set an embeddings deployment and switch the vector store backend.

```text
VECTOR_STORE=pgvector
OPENAI_EMBEDDING_DEPLOYMENT_NAME=
PGVECTOR_CONNECTION=postgresql+psycopg://docqa:docqa@pgvector:5432/docqa
```

### 4. Start the stack

```shell
docker compose up --build
```

To run with the pgvector backend, set `VECTOR_STORE=pgvector` and include the pgvector profile:

```shell
docker compose --profile pgvector up --build
```

Check the API health endpoint.

```shell
curl http://localhost:8000/health
```

Expected response:

```json
{"message":"OK"}
```

Open the demo UI:

```text
http://localhost:8000
```

Open the API docs:

```text
http://localhost:8000/docs
```

## API Usage

Upload a PDF.

```shell
curl -X POST "http://localhost:8000/files/upload?chunk_size=1000" \
  -F "file=@tests/data/rome_guide.pdf;type=application/pdf"
```

Expected response:

```json
{
  "document_id": "8f50a3f9-8a95-4c72-b4ad-3e17613d1219",
  "filename": "rome_guide.pdf",
  "chunks_added": 73
}
```

Ask a question.

```shell
curl -N "http://localhost:8000/files/query?question=when%20was%20rome%20founded%3F&temperature=0&n_docs=3"
```

The response is an event stream.

```text
event: token
data: {"text":"Rome"}

event: citations
data: {"citations":[{"filename":"rome_guide.pdf","page":1,"chunk_index":0,"score":0.82,"preview":"..."}]}
```

List indexed documents.

```shell
curl http://localhost:8000/files
```

Delete one indexed document.

```shell
curl -X DELETE http://localhost:8000/files/{document_id}
```

## Configuration

Runtime integrations are optional dependencies:

```text
azure       Azure OpenAI chat and embeddings
openai      OpenAI-compatible chat and embeddings
vertexai    Google Vertex AI chat and embeddings
weaviate    Weaviate vector store
pgvector    PostgreSQL pgvector store
all         All model providers and vector stores
```

Required Azure OpenAI values:

```text
LLM_PROVIDER=azure
OPENAI_DEPLOYMENT_NAME=
OPENAI_API_VERSION=2023-07-01-preview
OPENAI_API_KEY=
OPENAI_API_BASE=
```

Pgvector uses client-side embeddings, so it also needs an embeddings deployment:

```text
EMBEDDINGS_PROVIDER=azure
OPENAI_EMBEDDING_DEPLOYMENT_NAME=
```

Optional OpenAI-compatible provider:

```text
LLM_PROVIDER=openai
EMBEDDINGS_PROVIDER=openai
OPENAI_MODEL=
OPENAI_EMBEDDING_MODEL=
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

Optional Vertex AI provider:

```text
LLM_PROVIDER=vertexai
EMBEDDINGS_PROVIDER=vertexai
VERTEXAI_MODEL=
VERTEXAI_EMBEDDING_MODEL=
VERTEXAI_PROJECT=
VERTEXAI_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=
```

If `GOOGLE_APPLICATION_CREDENTIALS` is not set, Vertex AI uses Google Application Default Credentials from the environment.

Vector store backends:

```text
VECTOR_STORE=weaviate
```

```text
VECTOR_STORE=pgvector
PGVECTOR_CONNECTION=postgresql+psycopg://docqa:docqa@pgvector:5432/docqa
PGVECTOR_COLLECTION=docqa_stream
```

## Troubleshooting

- `GET /health` fails: check that the API container is running and `FASTAPI_PORT` matches the port mapping.
- Upload fails with a Weaviate connection error: check that both `WEAVIATE_PORT` and `WEAVIATE_GRPC_PORT` are exposed.
- Upload or query fails with pgvector: check that `docker compose --profile pgvector up --build` started the `pgvector` service and that `PGVECTOR_CONNECTION` uses the `postgresql+psycopg://` driver.
- Query fails with an authentication error: check the model deployment name, endpoint, API key, and API version.
- Pgvector embedding fails: check `EMBEDDINGS_PROVIDER` and the embedding model or deployment variable.
- Vertex AI fails before model invocation: check `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, and Google Application Default Credentials or `GOOGLE_APPLICATION_CREDENTIALS`.
- Empty PDF extraction: try a text-based PDF first. Scanned PDFs may require OCR-related system dependencies.

## Tests

```shell
docker exec fastapi-application poetry run pytest .
```

When dependencies are installed locally:

```shell
poetry run pytest .
```
