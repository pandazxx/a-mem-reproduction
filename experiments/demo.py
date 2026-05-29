#!/usr/bin/env python3
"""
A-Mem demo — quick walkthrough of the three core operations.

Ingests the first 10 memories from the comparison dataset (so the demo
finishes in roughly a minute) and answers 3 representative questions:

  1. Note construction  (keywords / tags / context extracted by the LLM)
  2. Memory evolution    (existing notes evolving when new info arrives)
  3. Agentic retrieval   (cosine similarity + link traversal + reader LLM)

For the full 40-memory / 22-question evaluation, run ``just eval`` instead.

Requires:
  export NVIDIA_API_KEY=nvapi-…
  just sync
  just demo
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

from . import _nim
from .amem import AgenticMemorySystem


def _require_api_key() -> None:
    if "NVIDIA_API_KEY" not in os.environ:
        print(
            "ERROR: set NVIDIA_API_KEY first.\n"
            "  export NVIDIA_API_KEY=nvapi-…\n"
        )
        sys.exit(1)

DATASET_PATH = Path(__file__).parent / "comparison-dataset" / "dataset.json"
DEMO_MEMORY_COUNT = 10
DEMO_QUESTIONS = ["q01", "q02", "q07"]  # single-hop + a two-hop


def hr(title: str = "") -> None:
    if title:
        print(f"\n{'─' * 4} {title} {'─' * (60 - len(title))}")
    else:
        print("─" * 66)


def main() -> None:
    _require_api_key()
    print("A-Mem Reproduction Demo")
    print(f"LLM:        NVIDIA NIM ({_nim.LLM_MODEL})")
    print(f"Embeddings: local sentence-transformers (all-MiniLM-L6-v2)")
    print(f"Dataset:    comparison-dataset/ (first {DEMO_MEMORY_COUNT} memories)")
    print(f"            See README.md in that folder for design rationale.")
    hr()

    with DATASET_PATH.open() as f:
        dataset = json.load(f)
    memories = dataset["memories"][:DEMO_MEMORY_COUNT]
    questions = [q for q in dataset["questions"] if q["id"] in DEMO_QUESTIONS]

    mem = AgenticMemorySystem()

    # ── Phase 1: Ingest memories ─────────────────────────────────────────
    hr(f"Phase 1 — Ingesting {len(memories)} memories")
    for i, m in enumerate(memories, 1):
        print(f"\n  [{i}/{len(memories)}] {m['id']}  {textwrap.shorten(m['content'], 65)}")
        mem.add_note(content=m["content"], timestamp=m["timestamp"], id=m["id"])
        note = mem.memories[m["id"]]
        print(f"         keywords={note.keywords}")
        print(f"         tags={note.tags}")
        print(f"         context={textwrap.shorten(note.context, 70)}")
        print(f"         links={[lid for lid in note.links]}  "
              f"evolved_neighbors={len(note.evolution_history)}")

    # ── Phase 2: Memory graph summary ────────────────────────────────────
    hr("Phase 2 — Memory graph after ingestion")
    total_links = 0
    total_evolutions = 0
    for mid, note in mem.memories.items():
        marker = "*" if note.evolution_history else " "
        print(f"  {marker} {mid}  links={len(note.links):2d}  "
              f"evolutions={len(note.evolution_history)}  "
              f"{textwrap.shorten(note.content, 55)}")
        for lid in note.links:
            target = mem.memories.get(lid)
            if target:
                print(f"        → {lid}: {textwrap.shorten(target.content, 50)}")
        total_links += len(note.links)
        total_evolutions += len(note.evolution_history)
    print(f"\n  Total links: {total_links}   "
          f"Total evolution events: {total_evolutions}")

    # ── Phase 3: Agentic retrieval + answering ───────────────────────────
    hr("Phase 3 — Agentic retrieval (retrieve + read)")
    traces: list[tuple[dict, dict]] = []
    for q in questions:
        print(f"\n  Q ({q['id']}, {q['category']}): {q['question']}")
        print(f"  Expected: {q['expected_answer']}")
        response = mem.read(q["question"], k=5)
        print(f"  Got:      {response['answer']}")
        print(f"  Retrieved: {response['retrieved_ids']}")
        if response["links_followed"]:
            print(f"  Links followed: {response['links_followed']}")
        traces.append((q, response))

    # ── Phase 4: Mermaid visualisation ───────────────────────────────────
    hr("Phase 4 — Mermaid diagrams (paste into a Markdown viewer)")
    mem_order = [m["id"] for m in memories]
    print("\n### Memory graph\n")
    print("```mermaid")
    print(mem.to_mermaid_graph(order=mem_order))
    print("```")
    for q, response in traces:
        print(f"\n### Retrieval trace — {q['id']}\n")
        print("```mermaid")
        print(mem.to_mermaid_trace(q["question"], response))
        print("```")

    hr()
    print("Demo complete.  For full 22-question evaluation: just eval")


if __name__ == "__main__":
    main()
