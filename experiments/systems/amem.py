"""
A-Mem adapter for the comparison harness.

This is a thin SystemAdapter wrapper around ``AgenticMemorySystem``
(defined in ``experiments/amem.py``). The interesting algorithmic work
lives there; this file just translates between the harness's
``(items, question) → run.json`` contract and A-Mem's API.

Algorithm recap
---------------
A-Mem (Xu et al. 2025, §3) is a Zettelkasten-inspired memory system.
For every new interaction it runs three LLM calls:

  1. Note construction (P_s1)
       LLM extracts keywords, tags, contextual description from the raw
       content. The text gets enriched into a structured note.

  2. Link generation + memory evolution (P_s2 / P_s3)
       Top-k cosine-similar existing notes are surfaced; an LLM decides
       which ones to link to AND whether to rewrite the linked notes'
       attributes (write-time reconsolidation).

  3. Retrieval (§3.4)
       Query is embedded; top-k notes by cosine + their 1-hop link
       neighbours are returned; a reader LLM synthesises the final
       answer from that bundle.

Inputs and outputs
------------------
``ingest(items)``
    items : list[dict] — each dict has at minimum
        - ``id``        : str   (e.g. "m01")
        - ``content``   : str   (raw text)
        - ``timestamp`` : str   (optional ISO-8601)
    Side effect : populates ``self.mem.memories`` with one MemoryNote
                  per item. Note IDs are preserved from the input.

``query(question)``
    question : str
    returns : dict with
        - ``answer``         : str
        - ``retrieved_ids``  : list[str]              (memory IDs)
        - ``links_followed`` : list[(str, str)]       (from_id, to_id)

``dump_state(queries, output_dir, metadata)``
    queries : list[dict] — the harness's per-question records
                           (question_id, score, predicted_answer, …)
    output_dir : Path     — written to ``output_dir/run.json`` plus
                           mermaid + HTML under ``output_dir/html/``.

``render_from_run(run_path, output_dir)`` [classmethod]
    Re-emits the HTML from an existing ``run.json``. No LLM calls;
    used by ``just render``.
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
        # AgenticMemorySystem owns the ChromaDB collection (for embeddings)
        # and the in-memory ``memories: dict[str, MemoryNote]`` map.
        self.mem = AgenticMemorySystem()

    def ingest(self, items: list[dict[str, Any]]) -> None:
        """
        Index ``items`` chronologically. Each call to ``add_note`` runs
        the full A-Mem pipeline for one note:

            (a) Construct note — LLM call P_s1 fills keywords / tags /
                                 context, then the concatenated text is
                                 embedded via the local sentence-
                                 transformer (no API call).
            (b) Link + evolve  — LLM call P_s2/P_s3 reads the top-k
                                 nearest neighbours and decides links +
                                 whether to rewrite neighbour attributes.

        Cost  : 2 LLM calls per item (+ up to k extra if many neighbours
                are rewritten in one shot — see paper §3.3).
        Order : matters. Later items can evolve earlier ones, so we
                preserve the dataset's chronological order.
        """
        for i, m in enumerate(items, 1):
            self.mem.add_note(
                content=m["content"],
                timestamp=m.get("timestamp"),
                id=m["id"],            # preserve dataset IDs (e.g. "m01")
            )
            note = self.mem.memories[m["id"]]
            print(
                f"  [{i:2d}/{len(items)}] {m['id']}  "
                f"tags={note.tags}  links={len(note.links)}  "
                f"neighbors_evolved={len(note.evolution_history)}"
            )

    def query(self, question: str) -> dict[str, Any]:
        """
        Retrieve-then-answer.

        Steps inside ``AgenticMemorySystem.read``:
            1. Embed query and pull top-k=5 notes by cosine similarity.
            2. For each direct hit, follow its ``links`` (1-hop) and add
               those notes to the bundle (deduplicated).
            3. Reader LLM sees the bundle (id + timestamp + content) and
               answers in one sentence. The prompt prefers later
               timestamps on contradictions — important for the
               ``information_update`` category.

        Returns the harness contract:
            { "answer": str,
              "retrieved_ids": list[str],
              "links_followed": list[(str, str)] }
        """
        return self.mem.read(question, k=5)

    def dump_state(
        self,
        queries: list[dict[str, Any]],
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        """
        Persist + render.

        ``run.json`` shape (written via render.write_run):
            {
              "metadata":  {... + system, system_label},
              "memories":  [MemoryNote.to_dict(), …],   # full state
              "queries":   queries                       # input as-is
            }
        Then re-render every artifact under ``output_dir/`` by replaying
        ``render_from_run`` against the file we just wrote.
        """
        run_path = output_dir / "run.json"
        meta = {**metadata, "system": self.name, "system_label": self.label}
        render.write_run(run_path, self.mem.memories, queries, meta)
        self.render_from_run(run_path, output_dir)

    @classmethod
    def render_from_run(cls, run_path: Path, output_dir: Path) -> None:
        """
        Re-render mermaid + interactive HTML from a frozen ``run.json``.
        No LLM, no ChromaDB; loads memories purely from the JSON.
        """
        run = render.load_run(run_path)
        render.render_all(run, output_dir)
