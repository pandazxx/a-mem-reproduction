"""
HippoRAG vs A-Mem Comparison — Evaluation Harness Skeleton

This is a SKELETON. You need to fill in the system-specific loaders for
your HippoRAG and A-Mem reproductions. The dataset loading, scoring, and
reporting logic is system-agnostic and ready to use.

Usage:
    python eval_template.py --hipporag-dir /path/to/hipporag --amem-dir /path/to/amem

Outputs:
    results.json        — raw per-question answers from each system
    summary.md          — pretty-printed per-category comparison table
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> dict[str, Any]:
    """Load the comparison dataset JSON."""
    with path.open() as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# System adapters — YOU NEED TO IMPLEMENT THESE
# -----------------------------------------------------------------------------

class HippoRAGSystem:
    """Adapter for your HippoRAG reproduction."""

    def __init__(self, repo_dir: Path, llm_model: str, embedding_model: str):
        """Initialise the HippoRAG system.

        TODO: import from your hipporag-reproduction codebase and construct
        the HippoRAG instance with the same config you used for the demo.
        """
        self.repo_dir = repo_dir
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        # TODO: instantiate the HippoRAG object here
        # e.g., self.hipporag = HippoRAG(corpus_name="comparison", ...)
        raise NotImplementedError("Wire up your HippoRAG demo here")

    def ingest(self, memories: list[dict[str, Any]]) -> None:
        """Feed memories to the HippoRAG indexing pipeline.

        Memories are pre-sorted chronologically. For each memory:
        - content is the passage text
        - timestamp may be passed if your reproduction tracks it
        """
        # TODO: call OpenIE pipeline, build the graph, persist the index
        raise NotImplementedError

    def query(self, question: str) -> dict[str, Any]:
        """Answer one question. Return dict with at least:
        - answer: the system's text answer
        - retrieved_ids: list of memory IDs the system surfaced (if traceable)
        """
        # TODO: run NER → PPR → top-K passage retrieval → reader LLM
        raise NotImplementedError


class AMemSystem:
    """Adapter for your A-Mem reproduction."""

    def __init__(self, repo_dir: Path, llm_model: str, embedding_model: str):
        """Initialise the A-Mem system.

        TODO: import from your a-mem-reproduction codebase and construct
        the A-Mem instance with the same config you used for the demo.
        """
        self.repo_dir = repo_dir
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        # TODO: instantiate the A-Mem object here
        raise NotImplementedError("Wire up your A-Mem demo here")

    def ingest(self, memories: list[dict[str, Any]]) -> None:
        """Feed memories to A-Mem's note construction pipeline.

        Memories are pre-sorted chronologically — important for memory
        evolution to trigger naturally on the later-arriving updates.
        """
        # TODO: for each memory, call note construction → link generation
        # → memory evolution
        raise NotImplementedError

    def query(self, question: str) -> dict[str, Any]:
        """Answer one question. Return dict with at least:
        - answer: the system's text answer
        - retrieved_ids: list of memory IDs (if traceable)
        - links_followed: list of (from_id, to_id) link traversals (optional)
        """
        # TODO: embed query → top-k retrieval → link traversal → reader LLM
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Scoring — system-agnostic
# -----------------------------------------------------------------------------

def normalise(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def score_answer(predicted: str, expected: str) -> dict[str, Any]:
    """Compute several scoring signals. Use whichever matches your needs.

    Returns:
        exact_match: bool
        substring_match: bool  (predicted contains expected, or vice versa)
        token_overlap_f1: float
    """
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
        # Heuristic: a question is "correct" if substring match or F1 >= 0.5.
        # Override per-category if you want stricter or more lenient grading.
        "correct": substring or f1 >= 0.5,
    }


def is_abstention(predicted: str) -> bool:
    """Heuristic: does the predicted answer indicate the system abstained?"""
    pred_norm = normalise(predicted)
    abstention_phrases = [
        "i don't know", "do not know", "not mentioned", "no information",
        "cannot determine", "unknown", "unclear", "not specified",
        "not enough information", "no record", "not provided",
    ]
    return any(phrase in pred_norm for phrase in abstention_phrases)


def score_absence_question(predicted: str) -> dict[str, Any]:
    """Special scoring for absence/abstention questions."""
    abstained = is_abstention(predicted)
    return {
        "abstained": abstained,
        "correct": abstained,  # correct = abstained on absence questions
    }


# -----------------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------------

def evaluate(system, system_name: str, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Ingest memories then answer each question. Return per-question results."""
    print(f"\n=== Ingesting memories into {system_name} ===")
    system.ingest(dataset["memories"])
    print(f"Ingested {len(dataset['memories'])} memories.")

    results = []
    for q in dataset["questions"]:
        print(f"\n[{system_name}] {q['id']} ({q['category']}): {q['question']}")
        response = system.query(q["question"])
        answer = response.get("answer", "")
        print(f"  → {answer[:200]}")

        if q["category"] == "absence_abstention":
            score = score_absence_question(answer)
        else:
            score = score_answer(answer, q["expected_answer"])

        results.append({
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "expected_winner": q["expected_winner"],
            "predicted_answer": answer,
            "retrieved_ids": response.get("retrieved_ids", []),
            "score": score,
        })
    return results


def summarise_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Aggregate correct/total per category."""
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["score"].get("correct"):
            by_cat[r["category"]]["correct"] += 1
    return dict(by_cat)


def emit_summary_md(
    hipporag_results: list[dict[str, Any]],
    amem_results: list[dict[str, Any]],
    dataset: dict[str, Any],
) -> str:
    """Generate a Markdown summary comparing the two systems."""
    hr_by_cat = summarise_by_category(hipporag_results)
    am_by_cat = summarise_by_category(amem_results)
    categories = sorted({r["category"] for r in hipporag_results})

    lines = [
        "# HippoRAG vs A-Mem — Evaluation Summary",
        "",
        "## Per-category accuracy",
        "",
        "| Category | Expected winner | HippoRAG | A-Mem | Actual winner |",
        "|---|---|---|---|---|",
    ]
    expected_winner_by_cat = {q["category"]: q["expected_winner"] for q in dataset["questions"]}
    for cat in categories:
        hr = hr_by_cat.get(cat, {"correct": 0, "total": 0})
        am = am_by_cat.get(cat, {"correct": 0, "total": 0})
        hr_pct = (hr["correct"] / hr["total"] * 100) if hr["total"] else 0
        am_pct = (am["correct"] / am["total"] * 100) if am["total"] else 0
        if hr_pct > am_pct + 5:
            winner = "HippoRAG"
        elif am_pct > hr_pct + 5:
            winner = "A-Mem"
        else:
            winner = "tie"
        lines.append(
            f"| {cat} | {expected_winner_by_cat.get(cat, '?')} | "
            f"{hr['correct']}/{hr['total']} ({hr_pct:.0f}%) | "
            f"{am['correct']}/{am['total']} ({am_pct:.0f}%) | {winner} |"
        )

    hr_total = sum(c["correct"] for c in hr_by_cat.values())
    am_total = sum(c["correct"] for c in am_by_cat.values())
    total = sum(c["total"] for c in hr_by_cat.values())
    lines += [
        f"| **Total** | — | **{hr_total}/{total}** | **{am_total}/{total}** | — |",
        "",
        "## Per-question results",
        "",
        "| Q | Cat | Expected | HippoRAG correct | A-Mem correct |",
        "|---|---|---|---|---|",
    ]
    for hr, am in zip(hipporag_results, amem_results):
        lines.append(
            f"| {hr['question_id']} | {hr['category']} | {hr['expected_winner']} | "
            f"{'✓' if hr['score'].get('correct') else '✗'} | "
            f"{'✓' if am['score'].get('correct') else '✗'} |"
        )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.json", type=Path)
    parser.add_argument("--hipporag-dir", required=True, type=Path)
    parser.add_argument("--amem-dir", required=True, type=Path)
    parser.add_argument("--llm-model", default="meta/llama-3.3-70b-instruct")
    parser.add_argument("--embedding-model", default="nvidia/nv-embed-v2")
    parser.add_argument("--output-dir", default=".", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Loaded dataset: {len(dataset['memories'])} memories, "
          f"{len(dataset['questions'])} questions")

    hipporag = HippoRAGSystem(args.hipporag_dir, args.llm_model, args.embedding_model)
    amem = AMemSystem(args.amem_dir, args.llm_model, args.embedding_model)

    hipporag_results = evaluate(hipporag, "HippoRAG", dataset)
    amem_results = evaluate(amem, "A-Mem", dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.json").open("w") as f:
        json.dump({
            "hipporag": hipporag_results,
            "amem": amem_results,
        }, f, indent=2)

    summary = emit_summary_md(hipporag_results, amem_results, dataset)
    (args.output_dir / "summary.md").write_text(summary)
    print(f"\nWrote results.json and summary.md to {args.output_dir}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
