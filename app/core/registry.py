from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A generic registry to store and retrieve objects by name."""

    def __init__(self):
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T):
        """Registers an item with a given name."""
        if not isinstance(name, str) or not name:
            raise ValueError("Name must be a non-empty string.")
        if name in self._items:
            raise ValueError(f"Item with name '{name}' already registered.")
        self._items[name] = item

    def get(self, name: str) -> T:
        """Retrieves an item by its name."""
        if name not in self._items:
            raise KeyError(f"Item with name '{name}' not found in registry.")
        return self._items[name]

    def unregister(self, name: str):
        """Unregisters an item by its name."""
        if name not in self._items:
            raise KeyError(f"Item with name '{name}' not found in registry.")
        del self._items[name]

    def list_available(self) -> list[str]:
        """Lists all available item names in the registry."""
        return list(self._items.keys())


class EmbeddingModelRegistry(Registry[Callable[..., Any]]):
    """Registry specifically for embedding model constructors."""


class RerankerRegistry(Registry[Callable[..., Any]]):
    """Registry specifically for reranker constructors."""


# Global instances of the registries
embedding_model_registry = EmbeddingModelRegistry()
reranker_registry = RerankerRegistry()
