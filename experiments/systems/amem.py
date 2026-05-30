"""
A-Mem adapter for the comparison harness.

Wraps the existing ``AgenticMemorySystem`` (defined in ``experiments/amem.py``)
and re-uses ``experiments/render.py`` for the mermaid + pyvis output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import render
from ..amem import AgenticMemorySystem
from .base import SystemAdapter


class AMemAdapter(SystemAdapter):
    name = "amem"
    label = "A-Mem"

    def __init__(self) -> None:
        self.mem = AgenticMemorySystem()

    def ingest(self, items: list[dict[str, Any]]) -> None:
        for i, m in enumerate(items, 1):
            self.mem.add_note(
                content=m["content"],
                timestamp=m.get("timestamp"),
                id=m["id"],
            )
            note = self.mem.memories[m["id"]]
            print(
                f"  [{i:2d}/{len(items)}] {m['id']}  "
                f"tags={note.tags}  links={len(note.links)}  "
                f"neighbors_evolved={len(note.evolution_history)}"
            )

    def query(self, question: str) -> dict[str, Any]:
        return self.mem.read(question, k=5)

    def dump_state(
        self,
        queries: list[dict[str, Any]],
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        run_path = output_dir / "run.json"
        meta = {**metadata, "system": self.name, "system_label": self.label}
        render.write_run(run_path, self.mem.memories, queries, meta)
        run = render.load_run(run_path)
        render.render_all(run, output_dir)
