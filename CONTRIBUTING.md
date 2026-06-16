# Contributing

## Local setup

Copy the example environment file and fill in the model credentials.

```shell
poetry install --extras all
cp .env.example .env
docker compose up --build
```

Use `poetry install --extras "pgvector local"` when working only on the self-hosted pgvector/local-embeddings path.

The API runs at `http://localhost:8000`, the demo UI is served from `/`, and the OpenAPI docs are served from `/docs`.

## Tests

Run the test suite inside the API container.

```shell
docker exec fastapi-application poetry run pytest .
```

For small local changes, running `poetry run pytest .` is also fine when dependencies are installed on the host.

## Pull requests

Keep changes focused. Include tests for API behavior, document ingestion, citations, and regressions. Update `README.md` when user-facing setup or response shapes change.
