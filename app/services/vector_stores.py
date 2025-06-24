import numpy as np
import psycopg2
from psycopg2.extensions import AsIs, register_adapter
from psycopg2.extras import execute_values

from app.core.config import settings
from app.core.interfaces import SearchResult, VectorDB

# Register NumPy array and float32 adapters for psycopg2
register_adapter(np.ndarray, AsIs)
register_adapter(np.float32, AsIs)


class PGVectorStore(VectorDB):
    """PostgreSQL vector store implementation for document storage and retrieval."""

    def __init__(self):
        pass

    def _get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(
            host=settings.POSTGRES_SERVER,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DB,
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
            return

        # Validate document structure
        for doc in documents:
            if not all(key in doc for key in ["content", "embedding", "source"]):
                err = "Document must contain 'content', 'embedding', and 'source' keys"
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

        with self._get_connection() as conn, conn.cursor() as cursor:
            try:
                execute_values(
                    cursor,
                    query,
                    data_to_insert,
                    template="(%s, %s::vector, %s)",
                    page_size=100,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
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
        if not vector:
            return []

        query = """
            SELECT content, source
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """

        with self._get_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, (np.array(vector).tolist(), top_k))
                return [
                    SearchResult(content=row[0], source=row[1])
                    for row in cursor.fetchall()
                ]
            except Exception:
                conn.rollback()
                raise
