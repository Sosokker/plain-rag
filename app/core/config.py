import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog import get_logger

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT / ".env"

logger = get_logger()


class Settings(BaseSettings):
    # Database
    DB_SERVER: str = "localhost"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_NAME: str = "chat_hub"
    DB_PORT: str = "5432"

    # Vector Store
    VECTOR_STORE_TYPE: str = "pgvector"  # or "chroma", "faiss", etc.
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # File uploads
    ALLOWED_DOCUMENT_TYPES: list[str] = Field(default=["pdf", "txt", "md"])

    # LLM Models
    GEMINI_API_KEY: str = "secret"

    # Prompt
    # may be use prompt.pys
    SYSTEM_PROMPT: str = "You are a helpful assistant that answers questions based on the provided context."

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This function uses lru_cache to prevent re-reading the environment on each call.
    """
    try:
        return Settings()
    except ValidationError:
        logger.exception("Error loading settings")
        raise


settings = get_settings()

# Set environment variables for third-party libraries
os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"
