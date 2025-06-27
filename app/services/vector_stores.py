import numpy as np
import psycopg2
from psycopg2.extensions import AsIs, register_adapter
from psycopg2.extras import execute_values
from structlog import get_logger

from app.core.config import settings
from app.core.interfaces import SearchResult, VectorDB

# Register NumPy array and float32 adapters for psycopg2
register_adapter(np.ndarray, AsIs)
register_adapter(np.float32, AsIs)

logger = get_logger()


class PGVectorStore(VectorDB):
    """PostgreSQL vector store implementation for document storage and retrieval."""

    def __init__(self):
        pass

    def _get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(
            host=settings.DB_SERVER,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            dbname=settings.DB_NAME,
        )

    def upsert_documents(self, documents: list[dict]) -> None:
        """
        Upsert documents into the vector store.

        Args:
            documents: List of document dictionaries containing 'content', 'embedding', and 'source'.

        Raises:
            ValueError: If required fields are missing from documents.
            psycopg2.Error: For database-related errors.

        """
        if not documents:
            logger.warning("No documents provided for upsert.")
            return

        # Validate document structure
        for doc in documents:
            if not all(key in doc for key in ["content", "embedding", "source"]):
                err = "Document must contain 'content', 'embedding', and 'source' keys"
                logger.error(f"Invalid document structure: {doc}")
                raise ValueError(err)

        data_to_insert = [
            (doc["content"], np.array(doc["embedding"]), doc["source"])
            for doc in documents
        ]

        query = """
            INSERT INTO documents (content, embedding, source)
            VALUES %s
            ON CONFLICT (content, source) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            RETURNING id
        """

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                execute_values(
                    cursor,
                    query,
                    data_to_insert,
                    template="(%s, %s::vector, %s)",
                    page_size=100,
                )
                conn.commit()
        except psycopg2.Error as db_err:
            logger.exception(f"Database error during upsert: {db_err}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during upsert: {e}")
            raise

    def search(self, vector: list, top_k: int = 5) -> list[SearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            vector: The query vector to search with.
            top_k: Maximum number of results to return.

        Returns:
            List of search results with content and source.

        Raises:
            psycopg2.Error: For database-related errors.

        """
        if len(vector) == 0:
            logger.warning("Empty vector provided for search.")
            return []

        query = """
            SELECT content, source
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, (np.array(vector).tolist(), top_k))
                results = [
                    SearchResult(content=row[0], source=row[1])
                    for row in cursor.fetchall()
                ]
                return results
        except psycopg2.Error as db_err:
            logger.exception(f"Database error during search: {db_err}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during search: {e}")
            raise
