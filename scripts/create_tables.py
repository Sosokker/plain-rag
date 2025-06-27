#!/usr/bin/env python3
"""
Database table creation script with pgvector support.

This script initializes the database with the required tables and extensions
for vector similarity search using pgvector.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import psycopg2
import structlog
from dotenv import load_dotenv
from psycopg2.extensions import connection as pg_connection
from psycopg2.extensions import cursor as pg_cursor

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
logger = structlog.get_logger()


def get_db_config() -> dict:
    """
    Retrieve database configuration from environment variables.

    Returns:
        dict: Database connection parameters

    """
    return {
        "host": os.getenv("DB_SERVER", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "user"),
        "password": os.getenv("DB_PASSWORD", "password"),
        "database": os.getenv("DB_NAME", "mydatabase"),
    }


@contextmanager
def get_db_connection() -> Generator[tuple[pg_connection, pg_cursor], None, None]:
    """
    Context manager for database connection handling.

    Yields:
        Tuple containing connection and cursor objects

    """
    conn = None
    try:
        db_config = get_db_config()
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        logger.info(
            "Successfully connected to PostgreSQL database",
            database=db_config["database"],
            host=db_config["host"],
        )
        yield conn, cursor
    except Exception:
        logger.exception("Database connection failed")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()


def create_vector_extension(conn: pg_connection, cursor: pg_cursor) -> None:
    """Create pgvector extension if it doesn't exist."""
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        logger.info("pgvector extension is enabled")
    except Exception:
        logger.exception("Failed to create pgvector extension")
        conn.rollback()
        raise


def create_documents_table(conn: pg_connection, cursor: pg_cursor) -> None:
    """Create documents table with vector support."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(384),  -- Match the dimension of your embedding model
                source VARCHAR(255),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()
        logger.info("Table 'documents' created successfully")
    except Exception:
        logger.exception("Failed to create documents table")
        conn.rollback()
        raise


def create_vector_index(
    conn: pg_connection, cursor: pg_cursor, dimensions: int = 384
) -> None:
    """Create HNSW index on the vector column."""
    try:
        logger.info("Creating HNSW index on vectors of dimension %s", dimensions)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents
            USING HNSW (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        conn.commit()
        logger.info("HNSW index created successfully")
    except Exception:
        logger.exception("Failed to create vector index")
        conn.rollback()
        raise


def main() -> None:
    """Main function to set up the database schema."""
    load_dotenv()
    logger.info("Starting database setup")

    try:
        with get_db_connection() as (conn, cursor):
            create_vector_extension(conn, cursor)
            create_documents_table(conn, cursor)
            create_vector_index(conn, cursor)

        logger.info("Database setup completed successfully")
    except Exception:
        logger.exception("Database setup failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
