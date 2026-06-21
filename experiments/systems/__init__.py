"""System registry — maps short names to SystemAdapter classes.

Adapters are imported lazily so that touching one system (or the
render-only path) does not drag in every system's heavy dependencies
(e.g. A-Mem's ChromaDB / sentence-transformers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SystemAdapter

# short name -> "module:classname" under experiments.systems
_REGISTRY: dict[str, str] = {
    "amem": "amem:AMemAdapter",
    "hipporag": "hipporag:HippoRAGAdapter",
    "hipporag2": "hipporag2:HippoRAG2Adapter",
}


def get_system(name: str) -> "type[SystemAdapter]":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown system {name!r}. Available: {sorted(_REGISTRY)}"
        )
    import importlib

    module_name, cls_name = _REGISTRY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, cls_name)


class _LazySystems:
    """Mapping-like view that imports each adapter on first access."""

    def __contains__(self, name: object) -> bool:
        return name in _REGISTRY

    def __getitem__(self, name: str) -> "type[SystemAdapter]":
        return get_system(name)

    def __iter__(self):
        return iter(_REGISTRY)

    def keys(self):
        return _REGISTRY.keys()

    def __len__(self) -> int:
        return len(_REGISTRY)


SYSTEMS = _LazySystems()
