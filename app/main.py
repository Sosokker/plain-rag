from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from structlog import get_logger

from app.api import endpoints
from app.services.rag_service import RAGService

logger = get_logger()

# Load environment variables from .env file
load_dotenv()

# Dictionary to hold our application state, including the RAG service instance
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    logger.info("Application starting up...")
    # Initialize the RAG Service and store it in the app_state
    app_state["rag_service"] = RAGService()
    yield
    # This code runs on shutdown
    logger.info("Application shutting down...")
    app_state["rag_service"].db_conn.close() # Clean up DB connection
    app_state.clear()

app = FastAPI(lifespan=lifespan)

# Include the API router
app.include_router(endpoints.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Custom RAG API"}
