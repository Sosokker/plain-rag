from typing import Protocol, TypedDict

import numpy as np


class SearchResult(TypedDict):
    """Type definition for search results."""

    content: str
    source: str


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def rerank(
        self, documents: list[SearchResult], query: str
    ) -> list[SearchResult]: ...


class VectorDB(Protocol):
    def upsert_documents(self, documents: list[dict]) -> None: ...

    def search(self, vector: np.ndarray, top_k: int) -> list[SearchResult]: ...
