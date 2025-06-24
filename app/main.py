from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from structlog import get_logger

from app.api import endpoints
from app.services.embedding_providers import MiniLMEmbeddingModel
from app.services.rag_service import RAGService
from app.services.vector_stores import PGVectorStore

logger = get_logger()

# Load environment variables from .env file
load_dotenv()

# Dictionary to hold our application state, including the RAG service instance
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_provider = MiniLMEmbeddingModel()
    vector_store_provider = PGVectorStore()

    # This code runs on startup
    logger.info("Application starting up...")
    # Initialize the RAG Service and store it in the app_state
    app_state["rag_service"] = RAGService(
        embedding_model=embedding_provider, vector_db=vector_store_provider
    )
    yield


app = FastAPI(lifespan=lifespan)

# Include the API router
app.include_router(endpoints.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Custom RAG API"}
