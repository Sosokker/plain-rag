from enum import Enum


class EmbeddingModelName(str, Enum):
    MiniLMEmbeddingModel = "MiniLMEmbeddingModel"


class RerankerModelName(str, Enum):
    MiniLMReranker = "MiniLMReranker"


class LLMModelName(str, Enum):
    GeminiFlash = "gemini/gemini-2.0-flash"
