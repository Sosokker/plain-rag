from enum import Enum


class EmbeddingModelName(str, Enum):
    MiniLMEmbeddingModel = "sentence-transformers/all-MiniLM-L6-v2"


class RerankerModelName(str, Enum):
    MiniLMReranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LLMModelName(str, Enum):
    GeminiFlash = "gemini/gemini-2.0-flash"


class VectorStoreType(str, Enum):
    PGVECTOR = "pgvector"
