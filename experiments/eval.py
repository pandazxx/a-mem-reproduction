#!/usr/bin/env python3
"""
A-Mem evaluation harness for the HippoRAG-vs-A-Mem comparison dataset.

Filled-in A-Mem adapter for the system-agnostic skeleton at
``comparison-dataset/eval_template.py`` (HippoRAG adapter lives in the other
reproduction repo). Loads ``comparison-dataset/dataset.json``, ingests all 40
memories in chronological order, runs all 22 questions through A-Mem's
read pipeline, scores the answers, and emits results.json + summary.md.

Usage:
    just sync                       # one-time
    export NVIDIA_API_KEY=nvapi-…
    just eval                       # full run
    just eval -- --limit 5          # smoke test (first 5 questions)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .amem import AgenticMemorySystem


def _require_api_key() -> None:
    if "NVIDIA_API_KEY" not in os.environ:
        print("ERROR: set NVIDIA_API_KEY first.\n  export NVIDIA_API_KEY=nvapi-…")
        sys.exit(1)

DATASET_PATH = Path(__file__).parent / "comparison-dataset" / "dataset.json"


# -----------------------------------------------------------------------------
# A-Mem adapter
# -----------------------------------------------------------------------------

class AMemSystem:
    """Adapter so the eval harness can drive AgenticMemorySystem."""

    def __init__(self) -> None:
        self.mem = AgenticMemorySystem()

    def ingest(self, memories: list[dict[str, Any]]) -> None:
        for i, m in enumerate(memories, 1):
            self.mem.add_note(
                content=m["content"],
                timestamp=m.get("timestamp"),
                id=m["id"],
            )
            note = self.mem.memories[m["id"]]
            print(
                f"  [{i:2d}/{len(memories)}] {m['id']}  "
                f"tags={note.tags}  links={len(note.links)}  "
                f"neighbors_evolved={len(note.evolution_history)}"
            )

    def query(self, question: str) -> dict[str, Any]:
        return self.mem.read(question, k=5)


# -----------------------------------------------------------------------------
# Scoring (verbatim from comparison-dataset/eval_template.py)
# -----------------------------------------------------------------------------

def normalise(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def score_answer(predicted: str, expected: str) -> dict[str, Any]:
    pred_norm = normalise(predicted)
    exp_norm = normalise(expected)
    pred_tokens = set(pred_norm.split())
    exp_tokens = set(exp_norm.split())
    exact = pred_norm == exp_norm
    substring = exp_norm in pred_norm or pred_norm in exp_norm
    common = pred_tokens & exp_tokens
    if not pred_tokens or not exp_tokens:
        f1 = 0.0
    else:
        p = len(common) / len(pred_tokens) if pred_tokens else 0
        r = len(common) / len(exp_tokens) if exp_tokens else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "exact_match": exact,
        "substring_match": substring,
        "token_overlap_f1": round(f1, 3),
        "correct": substring or f1 >= 0.5,
    }


def is_abstention(predicted: str) -> bool:
    pred_norm = normalise(predicted)
    abstention_phrases = [
        "i don't know", "do not know", "not mentioned", "no information",
        "cannot determine", "unknown", "unclear", "not specified",
        "not enough information", "no record", "not provided",
    ]
    return any(phrase in pred_norm for phrase in abstention_phrases)


def score_absence_question(predicted: str) -> dict[str, Any]:
    abstained = is_abstention(predicted)
    return {"abstained": abstained, "correct": abstained}


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

def evaluate(
    system: AMemSystem,
    dataset: dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    print(f"\n── Ingesting {len(dataset['memories'])} memories into A-Mem ──")
    system.ingest(dataset["memories"])

    questions = dataset["questions"]
    if limit:
        questions = questions[:limit]

    print(f"\n── Answering {len(questions)} questions ──")
    results = []
    for q in questions:
        print(f"\n[{q['id']}] ({q['category']}) {q['question']}")
        response = system.query(q["question"])
        answer = response.get("answer", "")
        print(f"  expected: {q['expected_answer']}")
        print(f"  got:      {answer}")

        if q["category"] == "absence_abstention":
            score = score_absence_question(answer)
        else:
            score = score_answer(answer, q["expected_answer"])

        marker = "✓" if score.get("correct") else "✗"
        print(f"  {marker} {score}")

        results.append({
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "expected_winner": q["expected_winner"],
            "predicted_answer": answer,
            "retrieved_ids": response.get("retrieved_ids", []),
            "links_followed": response.get("links_followed", []),
            "score": score,
        })
    return results


def summarise_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["score"].get("correct"):
            by_cat[r["category"]]["correct"] += 1
    return dict(by_cat)


def emit_summary_md(results: list[dict[str, Any]]) -> str:
    by_cat = summarise_by_category(results)
    categories = sorted(by_cat.keys())

    lines = [
        "# A-Mem — Comparison Dataset Results",
        "",
        "Run via `just eval`. Pair with the HippoRAG reproduction's results to fill",
        "in `comparison-dataset/analysis.md`.",
        "",
        "## Per-category accuracy",
        "",
        "| Category | Expected winner | A-Mem correct/total |",
        "|---|---|---|",
    ]
    expected_winner_by_cat = {r["category"]: r["expected_winner"] for r in results}
    total_correct = 0
    total = 0
    for cat in categories:
        stats = by_cat[cat]
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
        lines.append(
            f"| {cat} | {expected_winner_by_cat.get(cat, '?')} | "
            f"{stats['correct']}/{stats['total']} ({pct:.0f}%) |"
        )
        total_correct += stats["correct"]
        total += stats["total"]
    lines.append(f"| **Total** | — | **{total_correct}/{total}** |")
    lines += [
        "",
        "## Per-question results",
        "",
        "| Q | Cat | Expected winner | A-Mem answer | Correct? |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        ans = r["predicted_answer"].replace("|", "\\|").replace("\n", " ")
        if len(ans) > 80:
            ans = ans[:77] + "…"
        lines.append(
            f"| {r['question_id']} | {r['category']} | {r['expected_winner']} | "
            f"{ans} | {'✓' if r['score'].get('correct') else '✗'} |"
        )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    _require_api_key()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH), type=Path)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (smoke test).")
    parser.add_argument("--output-dir", default="results", type=Path)
    args = parser.parse_args()

    with args.dataset.open() as f:
        dataset = json.load(f)
    print(f"Loaded {args.dataset}: "
          f"{len(dataset['memories'])} memories, "
          f"{len(dataset['questions'])} questions")

    system = AMemSystem()
    results = evaluate(system, dataset, limit=args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps({"amem": results}, indent=2)
    )
    summary = emit_summary_md(results)
    (args.output_dir / "summary.md").write_text(summary)

    # Mermaid visualisations — rendered natively by GitHub Markdown.
    mem_order = [m["id"] for m in dataset["memories"] if m["id"] in system.mem.memories]
    graph_md = (
        "# A-Mem Memory Graph\n\n"
        "Nodes are memories (in ingestion order). Edges are A-Mem links generated\n"
        "during the link-generation step. Highlighted nodes had at least one\n"
        "memory-evolution event (their context or tags were rewritten when a\n"
        "later memory arrived).\n\n"
        "```mermaid\n"
        + system.mem.to_mermaid_graph(order=mem_order)
        + "\n```\n"
    )
    (args.output_dir / "memory_graph.md").write_text(graph_md)

    trace_lines = ["# A-Mem Retrieval Traces\n",
                   "Solid arrows = direct vector-search hits.  ",
                   "Dashed arrows = one-hop A-Mem link traversals.\n"]
    for r in results:
        trace = system.mem.to_mermaid_trace(
            r["question"],
            {"retrieved_ids": r["retrieved_ids"], "links_followed": r["links_followed"]},
        )
        trace_lines.append(f"## {r['question_id']} — {r['category']}\n")
        trace_lines.append(f"**Q:** {r['question']}  ")
        trace_lines.append(f"**Expected:** {r['expected_answer']}  ")
        trace_lines.append(f"**Got:** {r['predicted_answer']}  ")
        trace_lines.append(f"**Correct:** {'✓' if r['score'].get('correct') else '✗'}\n")
        trace_lines.append("```mermaid\n" + trace + "\n```\n")
    (args.output_dir / "traces.md").write_text("\n".join(trace_lines))

    print(f"\nWrote {args.output_dir}/{{results.json,summary.md,memory_graph.md,traces.md}}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
