import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Application metadata
APP_NAME = "Chat Hub API"
API_VERSION = "0.1.0"
APP_DESCRIPTION = """
Chat Hub API - A modern chat application with AI capabilities.
"""


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources
    print("Starting application...")

    yield

    # Shutdown: Clean up resources
    print("Shutting down application...")


# Initialize FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=dict[str, Any])
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "service": APP_NAME, "version": API_VERSION}


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint that provides basic information about the API.
    """
    return {
        "message": f"Welcome to {APP_NAME}",
        "version": API_VERSION,
        "docs": "/docs",
        "health_check": "/health",
    }


# This allows running the app with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", 8001))
    uvicorn.run(
        "main:app", host="0.0.0.0", port=port, reload=True, reload_dirs=["./app"]
    )
