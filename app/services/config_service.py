import logging

from app.core.config import settings
from app.core.interfaces import EmbeddingModel, Reranker, VectorDB
from app.core.registry import (
    embedding_model_registry,
    reranker_registry,
    vector_store_registry,
)
from app.schemas.enums import EmbeddingModelName, RerankerModelName
from app.services.embedding_providers import MiniLMEmbeddingModel
from app.services.rerankers import MiniLMReranker
from app.services.vector_stores import PGVectorStore

logger = logging.getLogger(__name__)


class ConfigService:
    def __init__(self):
        self._current_embedding_model: EmbeddingModel | None = None
        self._current_reranker_model: Reranker | None = None
        self._current_vector_store: VectorDB | None = None
        self._loading_status: dict[str, bool] = {
            "embedding_model": False,
            "reranker_model": False,
            "vector_store": False,
        }

        self._register_models()

    def _register_models(self):
        """Register all default models"""
        embedding_model_registry.register("MiniLMEmbeddingModel", MiniLMEmbeddingModel)
        reranker_registry.register("MiniLMReranker", MiniLMReranker)
        vector_store_registry.register("PGVectorStore", PGVectorStore)

    async def initialize_models(self):
        """
        Initialize embedding and reranker mode,
        if not a valid name then fallback to default one.
        Will get call on first time starting the app.
        """
        logger.info("Initializing default models...")
        embedding_model_name = settings.EMBEDDING_MODEL
        if embedding_model_name not in EmbeddingModelName.__members__:
            logger.warning(
                "Embedding model '%s' is not valid. Falling back to default '%s'",
                embedding_model_name,
                EmbeddingModelName.MiniLMEmbeddingModel.value,
            )
            embedding_model_name = (
                EmbeddingModelName.MiniLMEmbeddingModel.value
            )  # use minilm as default
        await self.set_embedding_model(embedding_model_name)
        logger.info("Default embedding model initialized: %s", embedding_model_name)

        reranker_model_name = (
            getattr(settings, "RERANKER_MODEL", None)
            or RerankerModelName.MiniLMReranker.value
        )
        if reranker_model_name not in RerankerModelName.__members__:
            logger.warning(
                "Reranker model '%s' is not valid. Falling back to default '%s'",
                reranker_model_name,
                RerankerModelName.MiniLMReranker.value,
            )
            reranker_model_name = RerankerModelName.MiniLMReranker.value
        await self.set_reranker_model(reranker_model_name)
        logger.info("Default reranker model initialized: %s", reranker_model_name)

        vector_store_name = (
            getattr(settings, "VECTOR_STORE_TYPE", None) or "PGVectorStore"
        )
        if vector_store_name not in vector_store_registry.list_available():
            logger.warning(
                "Vector store '%s' is not valid. Falling back to default '%s'",
                vector_store_name,
                "PGVectorStore",
            )
            vector_store_name = "PGVectorStore"
        await self.set_vector_store(vector_store_name)
        logger.info("Default vector store initialized: %s", vector_store_name)

    async def set_embedding_model(self, model_name: str) -> str:
        """Set system embedding model based on provide model_name"""
        if (
            self._current_embedding_model
            and self._current_embedding_model.__class__.__name__ == model_name
        ):
            return f"Embedding model '{model_name}' is already in use."

        if self._loading_status["embedding_model"]:
            return "Another embedding model is currently being loaded. Please wait."

        try:
            self._loading_status["embedding_model"] = True
            logger.info("Attempting to load embedding model: %s", model_name)
            model_constructor = embedding_model_registry.get(model_name)
            self._current_embedding_model = model_constructor()
            settings.EMBEDDING_MODEL = model_name  # Update settings
        except KeyError:
            logger.warning("Embedding model '%s' not found in registry.", model_name)
            return (
                f"Embedding model '{model_name}' not available. "
                f"Current model remains '{self._current_embedding_model.__class__.__name__ if self._current_embedding_model else 'None'}'."
            )
        except Exception as e:
            logger.exception("Error loading embedding model %s: %s", model_name, e)
            return f"Failed to load embedding model '{model_name}': {e}"
        else:
            logger.info("Successfully loaded embedding model: %s", model_name)
            return f"Embedding model set to '{model_name}' successfully."
        finally:
            self._loading_status["embedding_model"] = False

    async def set_reranker_model(self, model_name: str) -> str:
        """Set system reranker model based on provide model_name"""
        if (
            self._current_reranker_model
            and self._current_reranker_model.__class__.__name__ == model_name
        ):
            return f"Reranker model '{model_name}' is already in use."

        if self._loading_status["reranker_model"]:
            return "Another reranker model is currently being loaded. Please wait."

        try:
            self._loading_status["reranker_model"] = True
            logger.info("Attempting to load reranker model: %s", model_name)
            model_constructor = reranker_registry.get(model_name)
            self._current_reranker_model = model_constructor()
            # settings.RERANKER_MODEL = model_name
        except KeyError:
            logger.warning("Reranker model '%s' not found in registry.", model_name)
            return (
                f"Reranker model '{model_name}' not available. "
                f"Current model remains '{self._current_reranker_model.__class__.__name__ if self._current_reranker_model else 'None'}'."
            )
        except Exception as e:
            logger.exception("Error loading reranker model %s: %s", model_name, e)
            return f"Failed to load reranker model '{model_name}': {e}"
        else:
            logger.info("Successfully loaded reranker model: %s", model_name)
            return f"Reranker model set to '{model_name}' successfully."
        finally:
            self._loading_status["reranker_model"] = False

    async def set_vector_store(self, store_name: str) -> str:
        """Set system vector store based on provided store_name"""
        if (
            self._current_vector_store
            and self._current_vector_store.__class__.__name__ == store_name
        ):
            return f"Vector store '{store_name}' is already in use."

        if self._loading_status["vector_store"]:
            return "Another vector store is currently being loaded. Please wait."

        try:
            self._loading_status["vector_store"] = True
            logger.info("Attempting to load vector store: %s", store_name)
            store_constructor = vector_store_registry.get(store_name)
            self._current_vector_store = store_constructor()
            settings.VECTOR_STORE_TYPE = store_name  # Update settings
        except KeyError:
            logger.warning("Vector store '%s' not found in registry.", store_name)
            return (
                f"Vector store '{store_name}' not available. "
                f"Current store remains '{self._current_vector_store.__class__.__name__ if self._current_vector_store else 'None'}'."
            )
        except Exception as e:
            logger.exception("Error loading vector store %s: %s", store_name, e)
            return f"Failed to load vector store '{store_name}': {e}"
        else:
            logger.info("Successfully loaded vector store: %s", store_name)
            return f"Vector store set to '{store_name}' successfully."
        finally:
            self._loading_status["vector_store"] = False

    def get_current_embedding_model(self) -> EmbeddingModel | None:
        return self._current_embedding_model

    def get_current_reranker_model(self) -> Reranker | None:
        return self._current_reranker_model

    def get_current_vector_store(self) -> VectorDB | None:
        return self._current_vector_store

    def get_available_embedding_models(self) -> list[str]:
        return embedding_model_registry.list_available()

    def get_available_reranker_models(self) -> list[str]:
        return reranker_registry.list_available()

    def get_available_vector_stores(self) -> list[str]:
        return vector_store_registry.list_available()
