# pyright: reportArgumentType=false
from sentence_transformers import CrossEncoder
from structlog import get_logger

from app.core.exception import ModelNotFoundError
from app.core.interfaces import Reranker, SearchResult

logger = get_logger()

# pyright: reportCallIssue=false


class MiniLMReranker(Reranker):
    def __init__(self, model_name: str):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as er:
            err = f"Failed to load model '{model_name}'"
            logger.exception(err)
            raise ModelNotFoundError(err) from er

    def rerank(self, documents: list[SearchResult], query: str) -> list[SearchResult]:
        if not documents:
            logger.warning("No documents to rerank.")
            return []

        # Preprocess pairs and keep track of original indexes
        pairs = []
        valid_docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            if not content:
                err = f"Document at index {i} has no content."
                logger.warning(err)
                continue
            pairs.append((query, content))
            valid_docs.append(doc)

        if not pairs:
            logger.warning("No valid document pairs to rerank.")
            return []

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            err = f"Model prediction failed: {e}"
            logger.exception(err)
            return valid_docs  # fallback: return unranked valid docs

        # Sort by score descending
        if len(scores) != len(valid_docs):
            logger.warning("Mismatch in number of scores and documents")
            return valid_docs  # or handle the mismatch appropriately

        result = sorted(
            zip(scores, valid_docs, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for _, doc in result]
