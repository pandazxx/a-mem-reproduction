"""
Base interface for retrieval systems benchmarked in the comparison harness.

Concrete adapters (A-Mem, HippoRAG, HippoRAG 2) live alongside this file.
Each implements:

  - ``ingest(items)``  — index a list of ``{"id", "content", "timestamp"}``
  - ``query(question)`` — return ``{"answer", "retrieved_ids", **traces}``
  - ``dump_state(...)`` — write the run artifact + system-specific HTML

The harness (``experiments/compare.py``) is system-agnostic; it just iterates
``SYSTEMS`` from ``experiments/systems/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class SystemAdapter(ABC):
    """One concrete subclass per retrieval system."""

    #: Short identifier used in CLI flags and as a results subdirectory name.
    name: str = "system"

    #: Human-readable label shown in result.html.
    label: str = "System"

    @abstractmethod
    def ingest(self, items: list[dict[str, Any]]) -> None:
        """Index the dataset memories. ``items`` is the dataset's ``memories`` list."""

    @abstractmethod
    def query(self, question: str) -> dict[str, Any]:
        """
        Answer one question. Return at minimum:
            ``answer``        : free-text answer from the system
            ``retrieved_ids`` : IDs of items the system surfaced

        Systems may include extra fields (links_followed, seed_entities,
        filtered_triples, …) which are persisted and used by the renderer.
        """

    @abstractmethod
    def dump_state(
        self,
        queries: list[dict[str, Any]],
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        """
        Persist all state needed to reconstruct + render this system's run
        to ``output_dir/run.json``. Also emit any system-specific HTML
        under ``output_dir/html/``.
        """

    @classmethod
    @abstractmethod
    def render_from_run(cls, run_path: Path, output_dir: Path) -> None:
        """
        Re-emit this system's HTML from an already-frozen ``run.json``.
        Does NOT call the LLM — used by ``just render`` to iterate on
        visualisations without re-running the evaluation.
        """
