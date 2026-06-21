"""
Base interface for memory systems plugged into the experiment framework.

Concrete adapters (A-Mem, HippoRAG, HippoRAG 2) live alongside this file.
Each system implements the two core operations from issue #4 — *memory
construction* (``ingest``) and *memory retrieval* (``query``) — plus the
artifact/visualisation hooks the runner needs.

Each adapter:

  - ``__init__(params)`` — accept a parameter set (tuning knobs). Unknown /
    omitted keys fall back to the system's own defaults, which are pinned to
    reproduce prior behaviour.
  - ``ingest(items)``  — build memory from a list of ``{"id", "content",
    "timestamp"}`` (the framework adapts ``Dataset.statements`` into this shape).
  - ``query(question)`` — return ``{"answer", "retrieved_ids", **trace}``.
  - ``dump_state(...)`` — freeze the run + render visualisations, split into
    two sub-directories of ``output_dir``:
        ``memory/index.html``          memory-structure visualisation
        ``result/result.json``         frozen run artifact
        ``result/index.html`` + ``traces/``   query-result visualisation
  - ``render_from_run(...)`` — re-emit the HTML from a frozen
    ``result/result.json`` without calling the LLM (used by ``just render``).

The runner (``experiments/compare.py``) is system-agnostic; it iterates the
``SYSTEMS`` registry from ``experiments/systems/__init__.py`` and scores each
system's answers + retrieved IDs uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class SystemAdapter(ABC):
    """One concrete subclass per memory system."""

    #: Short identifier used in CLI flags and as a results subdirectory name.
    name: str = "system"

    #: Human-readable label shown in comparison pages.
    label: str = "System"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        #: Tuning knobs for this run. Subclasses read keys with their own
        #: defaults so an empty dict reproduces baseline behaviour.
        self.params: dict[str, Any] = dict(params or {})

    @abstractmethod
    def ingest(self, items: list[dict[str, Any]]) -> None:
        """Build memory from ``items`` — each ``{"id", "content", "timestamp"}``."""

    @abstractmethod
    def query(self, question: str) -> dict[str, Any]:
        """
        Answer one query. Return at minimum:
            ``answer``        : free-text answer from the system
            ``retrieved_ids`` : IDs of statements the system surfaced
                                (scored against the test-set's
                                ``required_retrieval``)

        Systems may include extra trace fields (links_followed, seed_entities,
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
        Persist + render this experiment under ``output_dir``:
            ``result/result.json``  — frozen run (memory + per-query results)
            ``result/index.html``   — query-result visualisation
            ``memory/index.html``   — memory-structure visualisation
        """

    @classmethod
    @abstractmethod
    def render_from_run(cls, run_path: Path, output_dir: Path) -> None:
        """
        Re-emit ``memory/`` + ``result/`` HTML from an already-frozen
        ``result/result.json``. Does NOT call the LLM — used by
        ``just render`` to iterate on visualisations.
        """
