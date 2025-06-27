# PlainRAG

PlainRAG is RAG application without LLM orchestration frameworks like Langchain or Haystack but with `LiteLLM`, `Transformers`, `FastAPI`.

## Quick Start

0. You need to install `uv` first
1. Copy `.env.example` to `.env` and fill in your values.
2. Run the following command to build and start the services:

```bash
make install-deps # or uv sync
make create-tables
make start
```

or

```bash
docker compoes up -d
```

This will use Docker Compose to start the API and database services.

## Environment Variables

- `DB_SERVER`: Database server hostname (default: localhost)
- `DB_USER`: Database username (default: postgres)
- `DB_PASSWORD`: Database password
- `DB_NAME`: Database name (default: chat_hub)
- `DB_PORT`: Database port (default: 5432)

Other variables are documented in `.env.example`.

## Components

- **API** (`app/api/endpoints.py`): FastAPI endpoints for file ingestion, querying, and configuration.
- **Services** (`app/services/`):
  - `rag_service.py`: Core RAG pipeline (ingest, query, stream answers).
  - `config_service.py`: Manages model and vector store selection.
  - `embedding_providers.py`: Embedding model integration (e.g., MiniLM).
  - `rerankers.py`: Reranker model integration (e.g., CrossEncoder).
  - `vector_stores.py`: Vector database integration (PostgreSQL/pgvector).
- **Core** (`app/core/`):
  - `config.py`: Application settings and environment management.
  - `registry.py`: Registry pattern for models and stores.
  - `utils.py`: Text splitting utilities.
  - `exception.py`: Custom exceptions.
  - `interfaces.py`: Abstract interfaces for models and stores.
- **Schemas** (`app/schemas/`):
  - `models.py`: Pydantic models for API requests/responses.
  - `enums.py`: Enum definitions for model/store selection.

