#!/usr/bin/env python3
"""
A-Mem demo — quick walkthrough of the three core operations.

Ingests the first 10 statements from the comparison dataset (so the demo
finishes in roughly a minute) and answers 3 representative queries:

  1. Note construction  (keywords / tags / context extracted by the LLM)
  2. Memory evolution    (existing notes evolving when new info arrives)
  3. Agentic retrieval   (cosine similarity + link traversal + reader LLM)

For the full 40-statement / 22-query evaluation, run ``just eval`` instead.

Requires:
  export NVIDIA_API_KEY=nvapi-…
  just sync
  just demo
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

from . import _nim, compare, render, schema
from .systems.amem import AgenticMemorySystem


def _require_api_key() -> None:
    if "NVIDIA_API_KEY" not in os.environ:
        print(
            "ERROR: set NVIDIA_API_KEY first.\n"
            "  export NVIDIA_API_KEY=nvapi-…\n"
        )
        sys.exit(1)

DEMO_STATEMENT_COUNT = 10
DEMO_QUESTIONS = ["Q00001", "Q00002", "Q00007"]  # single-hop + a two-hop


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
    print(f"Dataset:    datasets/comparison/ (first {DEMO_STATEMENT_COUNT} statements)")
    print(f"            See README.md in that folder for design rationale.")
    hr()

    dataset = schema.load_dataset("comparison")
    testset = schema.load_testset("comparison")
    statements = dataset.statements[:DEMO_STATEMENT_COUNT]
    queries = [q for q in testset.queries if q["id"] in DEMO_QUESTIONS]

    mem = AgenticMemorySystem()

    # ── Phase 1: Ingest statements ───────────────────────────────────────
    hr(f"Phase 1 — Ingesting {len(statements)} statements")
    for i, s in enumerate(statements, 1):
        print(f"\n  [{i}/{len(statements)}] {s['id']}  {textwrap.shorten(s['statement'], 65)}")
        mem.add_note(content=s["statement"], timestamp=s["timestamp"], id=s["id"])
        note = mem.memories[s["id"]]
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
    for q in queries:
        print(f"\n  Q ({q['id']}, {q['category']}): {q['query']}")
        print(f"  Expected: {q['expected_answer']}")
        response = mem.read(q["query"], k=5)
        print(f"  Got:      {response['answer']}")
        print(f"  Retrieved: {response['retrieved_ids']}")
        if response["links_followed"]:
            print(f"  Links followed: {response['links_followed']}")
        traces.append((q, response))

    # ── Phase 4: Persist run artifact + render ───────────────────────────
    hr("Phase 4 — Persist + render")
    out_dir = Path("results/demo")
    (out_dir / "result").mkdir(parents=True, exist_ok=True)

    queries_payload = []
    for q, response in traces:
        required = q.get("required_retrieval", [])
        queries_payload.append({
            "question_id": q["id"],
            "category": q["category"],
            "question": q["query"],
            "expected_answer": q["expected_answer"],
            "expected_winner": q.get("expected_winner", ""),
            "required_retrieval": required,
            "predicted_answer": response["answer"],
            "score": compare._score_answer_for(q, response["answer"]),
            "retrieval_score": compare._score_retrieval(response["retrieved_ids"], required),
            "retrieved_ids": response["retrieved_ids"],
            "links_followed": response["links_followed"],
        })

    metadata = {
        "dataset": dataset.name,
        "testset": testset.name,
        "system": "amem",
        "system_label": "A-Mem",
        "paramset": "demo",
        "llm_model": _nim.LLM_MODEL,
        "embedding_model": mem.model_name,
        "mode": "demo",
    }
    memories_dict = {mid: note.to_dict() for mid, note in mem.memories.items()}
    run_path = out_dir / "result" / "result.json"
    render.write_run(run_path, memories_dict, queries_payload, metadata)
    run = render.load_run(run_path)
    render.render_all(run, out_dir)
    print(f"  Wrote {run_path}")
    print(f"  Rendered to {out_dir}/  "
          f"(memory/index.html + result/index.html)")

    hr()
    print("Demo complete.  For full 22-query evaluation: just eval")
    print("To tweak visualisations and re-render: edit experiments/render.py, "
          "then `just render`")


if __name__ == "__main__":
    main()
