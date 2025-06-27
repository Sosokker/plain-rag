import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.interfaces import EmbeddingModel


class MiniLMEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0].tolist()
