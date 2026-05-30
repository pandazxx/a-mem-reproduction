"""System registry — maps short names to SystemAdapter classes."""

from __future__ import annotations

from .amem import AMemAdapter
from .base import SystemAdapter
from .hipporag import HippoRAGAdapter
from .hipporag2 import HippoRAG2Adapter

SYSTEMS: dict[str, type[SystemAdapter]] = {
    AMemAdapter.name: AMemAdapter,
    HippoRAGAdapter.name: HippoRAGAdapter,
    HippoRAG2Adapter.name: HippoRAG2Adapter,
}


def get_system(name: str) -> type[SystemAdapter]:
    if name not in SYSTEMS:
        raise KeyError(
            f"Unknown system {name!r}. Available: {sorted(SYSTEMS)}"
        )
    return SYSTEMS[name]
