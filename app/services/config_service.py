import logging

from app.core.config import settings
from app.core.interfaces import EmbeddingModel, Reranker
from app.core.registry import embedding_model_registry, reranker_registry
from app.schemas.enums import EmbeddingModelName, RerankerModelName
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

        self._register_models()

    def _register_models(self):
        """Register all default models"""
        embedding_model_registry.register("MiniLMEmbeddingModel", MiniLMEmbeddingModel)
        reranker_registry.register("MiniLMReranker", MiniLMReranker)

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

    def get_current_embedding_model(self) -> EmbeddingModel | None:
        return self._current_embedding_model

    def get_current_reranker_model(self) -> Reranker | None:
        return self._current_reranker_model

    def get_available_embedding_models(self) -> list[str]:
        return embedding_model_registry.list_available()

    def get_available_reranker_models(self) -> list[str]:
        return reranker_registry.list_available()
