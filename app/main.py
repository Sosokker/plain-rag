from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from structlog import get_logger

from app.api import endpoints
from app.services.config_service import ConfigService
from app.services.rag_service import RAGService
from app.services.vector_stores import PGVectorStore

logger = get_logger()

# Load environment variables from .env file
load_dotenv()

# Dictionary to hold our application state, including the RAG service instance
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_service = ConfigService()
    await config_service.initialize_models()
    app_state["config_service"] = config_service

    # This code runs on startup
    logger.info("Application starting up...")
    # Initialize the RAG Service and store it in the app_state
    app_state["rag_service"] = RAGService(
        embedding_model=config_service.get_current_embedding_model(),
        vector_db=PGVectorStore(),
        reranker=config_service.get_current_reranker_model(),
    )
    yield


app = FastAPI(lifespan=lifespan)

# Include the API router
app.include_router(endpoints.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Custom RAG API"}
