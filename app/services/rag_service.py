import os
from pathlib import Path
from typing import TypedDict

import litellm
import numpy as np
import psycopg2
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from structlog import get_logger

from app.core.config import settings
from app.core.exception import DocumentExtractionError, DocumentInsertionError
from app.core.utils import RecursiveCharacterTextSplitter

logger = get_logger()

# pyright: reportArgumentType=false

# Load environment variables
load_dotenv()

# Initialize the embedding model globally to load it only once
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # Dimension of the all-MiniLM-L6-v2 model

os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY


class AnswerResult(TypedDict):
    answer: str
    sources: list[str]


class RAGService:
    def __init__(self):
        logger.info("Initializing RAGService...")
        # Load the embedding model ONCE
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cpu"
        )  # Use 'cuda' if GPU is available
        self.db_conn = psycopg2.connect(
            host=settings.POSTGRES_SERVER,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DB,
        )
        logger.info("RAGService initialized.")
        self.prompt = """Answer the question based on the following context. 
If you don't know the answer, say you don't know. Don't make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    def _split_text(
        self, text: str, chunk_size: int = 500, chunk_overlap: int = 100
    ) -> list[str]:
        """
        Split text into chunks with specified size and overlap.

        Args:
            text: Input text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks

        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return text_splitter.split_text(text)

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text chunk.

        Args:
            text: Input text to embed

        Returns:
            Numpy array containing the embedding vector

        """
        return EMBEDDING_MODEL.encode(text, convert_to_numpy=True)

    def _store_document(self, content: str, embedding: np.ndarray, source: str) -> int:
        """
        Store a document chunk in the database.

        Args:
            content: Text content of the chunk
            embedding: Embedding vector of the chunk
            source: Source file path

        Returns:
            ID of the inserted document

        """
        with self.db_conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (content, embedding, source)
                VALUES (%s, %s, %s)
                RETURNING id
            """,
                (content, embedding.tolist(), source),
            )
            doc_id = cursor.fetchone()
            if doc_id is None:
                err = "Failed to insert document into database"
                raise DocumentInsertionError(err)
            self.db_conn.commit()
            return doc_id[0]

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as a single string

        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise DocumentExtractionError(
                "Error extracting text from PDF: " + str(e)
            ) from e

    def _get_relevant_context(self, question: str, top_k: int) -> list[tuple[str, str]]:
        """Get the most relevant document chunks for a given question"""
        question_embedding = self.embedding_model.encode(
            question, convert_to_numpy=True
        )

        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT content, source 
                    FROM documents 
                    ORDER BY embedding <-> %s::vector 
                    LIMIT %s
                    """,
                    (question_embedding.tolist(), top_k),
                )
                results = cursor.fetchall()
                return results
        except Exception as e:
            logger.exception("Error retrieving context: %s", e)
            return []

    def ingest_document(self, file_path: str, filename: str):
        logger.info("Ingesting %s...", filename)
        if not Path(file_path).exists():
            err = f"File not found: {filename}"
            raise FileNotFoundError(err)

        logger.info("Processing PDF: %s : %s", filename, file_path)

        text = self._extract_text_from_pdf(file_path)
        if not text.strip():
            err = "No text could be extracted from the PDF"
            raise ValueError(err)

        chunks = self._split_text(text)
        logger.info("Split PDF into %d chunks", len(chunks))

        for i, chunk in enumerate(chunks, 1):
            logger.info("Processing chunk %d/%d", i, len(chunks))
            embedding = self._get_embedding(chunk)
            self._store_document(chunk, embedding, filename)

        logger.info("Successfully processed %d chunks from %s", len(chunks), filename)

    def answer_query(self, question: str) -> AnswerResult:
        relevant_context = self._get_relevant_context(question, 5)
        context_str = "\n\n".join([chunk[0] for chunk in relevant_context])
        sources = list({chunk[1] for chunk in relevant_context if chunk[1]})

        try:
            response = litellm.completion(
                model="gemini/gemini-2.0-flash",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context.",
                    },
                    {
                        "role": "user",
                        "content": self.prompt.format(
                            context=context_str, question=question
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=500,
            )

            answer_text = response.choices[0].message.content.strip()

            if not answer_text:
                answer_text = "No answer generated"
                sources = ["No sources"]

            return AnswerResult(answer=answer_text, sources=sources)

        except Exception:
            logger.exception("Error generating response")
            return AnswerResult(
                answer="Error generating response", sources=["No sources"]
            )
