from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from structlog import get_logger

from app.api import endpoints
from app.services.config_service import ConfigService
from app.services.rag_service import RAGService
from app.services.vector_stores import PGVectorStore

logger = get_logger()

load_dotenv()

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    config_service = ConfigService()
    await config_service.initialize_models()
    app_state["config_service"] = config_service

    logger.info("Application starting up...")
    embedding_model = config_service.get_current_embedding_model()
    reranker = config_service.get_current_reranker_model()
    if embedding_model is None:
        raise RuntimeError("Embedding model failed to initialize")
    app_state["rag_service"] = RAGService(
        embedding_model=embedding_model,
        vector_db=PGVectorStore(),
        reranker=reranker,
    )
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(endpoints.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Custom RAG API"}
