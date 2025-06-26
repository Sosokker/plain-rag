import json
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import litellm
from structlog import get_logger

from app.core.interfaces import EmbeddingModel, Reranker, VectorDB
from app.core.utils import RecursiveCharacterTextSplitter

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
                stream=True,
            )

            # Yield each chunk of the response as it's generated
            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield f'data: {{"token": "{json.dumps(delta.content)}"}}\n\n'

            # Yield sources at the end
            yield f'data: {{"sources": {json.dumps(sources)}}}\n\n'
            yield 'data: {"end_of_stream": true}\n\n'

        except Exception:
            logger.exception("Error generating streaming response")
            yield 'data: {"error": "Error generating response"}\n\n'
