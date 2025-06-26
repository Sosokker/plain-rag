import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from structlog import get_logger

from app.schemas.models import (
    ConfigUpdateRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.config_service import ConfigService
from app.services.rag_service import RAGService

logger = get_logger()

router = APIRouter()


# Dependency function to get the RAG service instance
def get_rag_service():
    from app.main import app_state

    return app_state["rag_service"]


def get_config_service():
    from app.main import app_state

    return app_state["config_service"]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
):
    # Save the uploaded file temporarily
    temp_dir = Path("temp_files")
    Path.mkdir(temp_dir, exist_ok=True)
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")
    file_path = temp_dir / Path(file.filename)

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Add the ingestion task to run in the background
    background_tasks.add_task(rag_service.ingest_document, file_path, file.filename)

    # Immediately return a response to the user
    return {
        "message": "File upload successful. Ingestion has started in the background.",
        "filename": file.filename,
    }


@router.post("/query", response_model=QueryResponse)
async def query_index(
    request: QueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
):
    try:
        result = rag_service.answer_query(request.question)

        answer = result.get("answer", "No answer generated")
        sources = result.get("sources", ["No sources"])

        return QueryResponse(answer=answer, sources=sources)

    except Exception:
        logger.exception("Failed to answer query")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query/stream")
async def query_index_stream(
    request: QueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
):
    try:
        return StreamingResponse(
            rag_service.answer_query_stream(request.question),
            media_type="text/event-stream",
        )

    except Exception:
        logger.exception("Failed to answer query")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/config")
async def update_configuration(
    request: ConfigUpdateRequest,
    config_service: Annotated[ConfigService, Depends(get_config_service)],
):
    responses = []
    if request.embedding_model:
        response = await config_service.set_embedding_model(request.embedding_model)
        responses.append(response)
    if request.reranker_model:
        response = await config_service.set_reranker_model(request.reranker_model)
        responses.append(response)
    # Add similar logic for LLM models and providers when implemented

    if not responses:
        return {"message": "No configuration changes requested."}

    return {"message": " ".join(responses)}
