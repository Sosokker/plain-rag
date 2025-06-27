import json
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import litellm
from PyPDF2 import PdfReader
from PyPDF2.errors import PyPdfError
from structlog import get_logger

from app.core.config import settings
from app.core.exception import FileTypeIngestionError
from app.core.interfaces import EmbeddingModel, Reranker, VectorDB
from app.core.utils import RecursiveCharacterTextSplitter
from app.schemas.enums import LLMModelName

logger = get_logger()


class AnswerResult(TypedDict):
    answer: str
    sources: list[str]


class RAGService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDB,
        reranker: Reranker | None = None,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.reranker = reranker
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

    def _ingest_document(self, text_chunks: list[str], source_name: str):
        embeddings = self.embedding_model.embed_documents(text_chunks)
        documents_to_upsert = [
            {"content": chunk, "embedding": emb, "source": source_name}
            for chunk, emb in zip(text_chunks, embeddings, strict=False)
        ]
        self.vector_db.upsert_documents(documents_to_upsert)

    def ingest_document(self, file_path: Path, source_name: str):
        path = Path(file_path)
        ext = path.suffix
        text = ""
        if ext[1:] not in settings.ALLOWED_DOCUMENT_TYPES:
            raise FileTypeIngestionError("Only support PDF, MD and TXT files")
        if ext == ".pdf":
            try:
                reader = PdfReader(str(file_path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except PyPdfError as e:
                logger.exception("PDF processing error for %s", file_path)
                raise ValueError(
                    f"Failed to extract text from PDF due to a PDF processing error: {e}"
                ) from e
            except Exception as e:
                logger.exception(
                    "An unexpected error occurred during PDF processing for %s",
                    file_path,
                )
                raise RuntimeError(
                    f"An unexpected error occurred during PDF processing: {e}"
                ) from e
        else:
            with Path(file_path).open("r", encoding="utf-8") as f:
                text = f.read()
        text_chunks = self._split_text(text)
        self._ingest_document(text_chunks, source_name)

    def answer_query(self, question: str) -> AnswerResult:
        query_embedding = self.embedding_model.embed_query(question)
        search_results = self.vector_db.search(query_embedding, top_k=5)

        if self.reranker:
            logger.info("Reranking search results...")
            search_results = self.reranker.rerank(search_results, question)

        sources = list({chunk["source"] for chunk in search_results if chunk["source"]})
        context_str = "\n\n".join([chunk["content"] for chunk in search_results])

        try:
            response = litellm.completion(
                model=LLMModelName.GeminiFlash.value,
                messages=[
                    {
                        "role": "system",
                        "content": settings.SYSTEM_PROMPT,
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

            answer_text = None
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                first_choice = choices[0]
                message = getattr(first_choice, "message", None)
                content = getattr(message, "content", None)
                if content:
                    answer_text = content.strip()
            if not answer_text:
                answer_text = "No answer generated"
                sources = ["No sources"]

            return AnswerResult(answer=answer_text, sources=sources)
        except Exception:
            logger.exception("Error generating response")
            return AnswerResult(
                answer="Error generating response", sources=["No sources"]
            )

    def answer_query_stream(self, question: str) -> Generator[str, None, None]:
        query_embedding = self.embedding_model.embed_query(question)
        search_results = self.vector_db.search(query_embedding, top_k=5)

        if self.reranker:
            logger.info("Reranking search results...")
            search_results = self.reranker.rerank(search_results, question)

        sources = list({chunk["source"] for chunk in search_results if chunk["source"]})
        context_str = "\n\n".join([chunk["content"] for chunk in search_results])

        try:
            response = litellm.completion(
                model=LLMModelName.GeminiFlash.value,
                messages=[
                    {
                        "role": "system",
                        "content": settings.SYSTEM_PROMPT,
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
                stream=True,
            )

            for chunk in response:
                choices = getattr(chunk, "choices", None)
                if choices and len(choices) > 0:
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None)
                    if content:
                        yield f'data: {{"token": {json.dumps(content)}}}\n\n'

            # Yield sources at the end
            yield f'data: {{"sources": {json.dumps(sources)}}}\n\n'
            yield 'data: {"end_of_stream": true}\n\n'

        except Exception:
            logger.exception("Error generating streaming response")
            yield 'data: {"error": "Error generating response"}\n\n'
