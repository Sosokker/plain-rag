from enum import Enum


class EmbeddingModelName(str, Enum):
    MiniLMEmbeddingModel = "MiniLMEmbeddingModel"


class RerankerModelName(str, Enum):
    MiniLMReranker = "MiniLMReranker"
