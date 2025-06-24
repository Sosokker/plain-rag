import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, PostgresDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT / ".env"


class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Chat Hub"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Security
    ALGORITHM: str = "HS256"

    # Backend Server
    BACKEND_CORS_ORIGINS: list[str | AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(self, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        if isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "chat_hub"
    POSTGRES_PORT: str = "5432"
    DATABASE_URI: PostgresDsn | None = None

    @model_validator(mode="after")
    def assemble_db_connection(self) -> "Settings":
        if self.DATABASE_URI is None:
            self.DATABASE_URI = PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD,
                host=self.POSTGRES_SERVER,
                port=int(self.POSTGRES_PORT),
                path=f"/{self.POSTGRES_DB or ''}",
            )
        return self

    # LLM Configuration
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    # Vector Store
    VECTOR_STORE_TYPE: str = "pgvector"  # or "chroma", "faiss", etc.
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    TESTING: bool = False

    # Rate Limiting
    RATE_LIMIT: str = "100/minute"

    # Caching
    CACHE_TTL: int = 300  # 5 minutes

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
    return Settings()


# Create settings instance
settings = get_settings()

# Set environment variables for third-party libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
