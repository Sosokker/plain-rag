import logging

from app.core.config import settings
from app.core.interfaces import EmbeddingModel, Reranker
from app.core.registry import embedding_model_registry, reranker_registry
from app.services.embedding_providers import MiniLMEmbeddingModel
from app.services.rerankers import MiniLMReranker

logger = logging.getLogger(__name__)


class ConfigService:
    def __init__(self):
        self._current_embedding_model: EmbeddingModel | None = None
        self._current_reranker_model: Reranker | None = None
        self._loading_status: dict[str, bool] = {
            "embedding_model": False,
            "reranker_model": False,
        }

        # Register available models
        self._register_models()

    def _register_models(self):
        # Register embedding models
        embedding_model_registry.register("MiniLMEmbeddingModel", MiniLMEmbeddingModel)
        # Register reranker models
        reranker_registry.register("MiniLMReranker", MiniLMReranker)

    async def initialize_models(self):
        logger.info("Initializing default models...")
        # Initialize default embedding model
        default_embedding_model_name = settings.EMBEDDING_MODEL
        await self.set_embedding_model(default_embedding_model_name)
        logger.info(
            f"Default embedding model initialized: {default_embedding_model_name}"
        )

        # Initialize default reranker model (if any)
        # Assuming a default reranker can be set in settings if needed
        # For now, let's assume MiniLMReranker is the default if not specified
        default_reranker_model_name = "MiniLMReranker"  # Or from settings
        await self.set_reranker_model(default_reranker_model_name)
        logger.info(
            f"Default reranker model initialized: {default_reranker_model_name}"
        )

    async def set_embedding_model(self, model_name: str) -> str:
        if (
            self._current_embedding_model
            and self._current_embedding_model.__class__.__name__ == model_name
        ):
            return f"Embedding model '{model_name}' is already in use."

        if self._loading_status["embedding_model"]:
            return "Another embedding model is currently being loaded. Please wait."

        try:
            self._loading_status["embedding_model"] = True
            logger.info(f"Attempting to load embedding model: {model_name}")
            model_constructor = embedding_model_registry.get(model_name)
            self._current_embedding_model = model_constructor()
            settings.EMBEDDING_MODEL = model_name  # Update settings
            logger.info(f"Successfully loaded embedding model: {model_name}")
            return f"Embedding model set to '{model_name}' successfully."
        except KeyError:
            logger.warning(f"Embedding model '{model_name}' not found in registry.")
            return (
                f"Embedding model '{model_name}' not available. "
                f"Current model remains '{self._current_embedding_model.__class__.__name__ if self._current_embedding_model else 'None'}'."
            )
        except Exception as e:
            logger.exception(f"Error loading embedding model {model_name}: {e}")
            return f"Failed to load embedding model '{model_name}': {e}"
        finally:
            self._loading_status["embedding_model"] = False

    async def set_reranker_model(self, model_name: str) -> str:
        if (
            self._current_reranker_model
            and self._current_reranker_model.__class__.__name__ == model_name
        ):
            return f"Reranker model '{model_name}' is already in use."

        if self._loading_status["reranker_model"]:
            return "Another reranker model is currently being loaded. Please wait."

        try:
            self._loading_status["reranker_model"] = True
            logger.info(f"Attempting to load reranker model: {model_name}")
            model_constructor = reranker_registry.get(model_name)
            self._current_reranker_model = model_constructor()
            # settings.RERANKER_MODEL = model_name # Add this to settings if you want to persist
            logger.info(f"Successfully loaded reranker model: {model_name}")
            return f"Reranker model set to '{model_name}' successfully."
        except KeyError:
            logger.warning(f"Reranker model '{model_name}' not found in registry.")
            return (
                f"Reranker model '{model_name}' not available. "
                f"Current model remains '{self._current_reranker_model.__class__.__name__ if self._current_reranker_model else 'None'}'."
            )
        except Exception as e:
            logger.exception(f"Error loading reranker model {model_name}: {e}")
            return f"Failed to load reranker model '{model_name}': {e}"
        finally:
            self._loading_status["reranker_model"] = False

    def get_current_embedding_model(self) -> EmbeddingModel | None:
        return self._current_embedding_model

    def get_current_reranker_model(self) -> Reranker | None:
        return self._current_reranker_model

    def get_available_embedding_models(self) -> list[str]:
        return embedding_model_registry.list_available()

    def get_available_reranker_models(self) -> list[str]:
        return reranker_registry.list_available()
