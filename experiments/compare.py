#!/usr/bin/env python3
"""
Comparison harness — runs A-Mem, HippoRAG, and HippoRAG 2 against a single
dataset, scores each, and emits a side-by-side result.html.

Layout produced under ``results/<dataset_name>/``:

    results/<dataset>/<system>/run.json    — frozen system state + per-Q results
    results/<dataset>/<system>/summary.md  — per-category scores
    results/<dataset>/<system>/html/       — per-question traces (interactive)
    results/<dataset>/result.html          — overall comparison

The dataset is selected from ``experiments/datasets/__init__.py``; systems
from ``experiments/systems/__init__.py``. Add a new dataset by dropping
``dataset.json`` next to those and registering it in ``DATASETS``.

Usage:
    just compare                       # default: comparison dataset, all systems
    just compare -- --dataset foo      # plug in a different dataset
    just compare -- --systems amem,hipporag2
    just compare -- --limit 5          # smoke test (first 5 questions)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import _nim, datasets, systems
from .systems.base import SystemAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Scoring (same heuristics as datasets/comparison/eval_template.py)
# -----------------------------------------------------------------------------

def _normalise(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _score_answer(predicted: str, expected: str) -> dict[str, Any]:
    pred_norm, exp_norm = _normalise(predicted), _normalise(expected)
    pred_tokens, exp_tokens = set(pred_norm.split()), set(exp_norm.split())
    common = pred_tokens & exp_tokens
    if pred_tokens and exp_tokens:
        p = len(common) / len(pred_tokens)
        r = len(common) / len(exp_tokens)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        f1 = 0.0
    substring = exp_norm in pred_norm or pred_norm in exp_norm
    return {
        "exact_match": pred_norm == exp_norm,
        "substring_match": substring,
        "token_overlap_f1": round(f1, 3),
        "correct": substring or f1 >= 0.5,
    }


_ABSTENTION_PHRASES = (
    "i don't know", "do not know", "not mentioned", "no information",
    "cannot determine", "unknown", "unclear", "not specified",
    "not enough information", "no record", "not provided",
)


def _is_abstention(predicted: str) -> bool:
    norm = _normalise(predicted)
    return any(p in norm for p in _ABSTENTION_PHRASES)


def _score_absence(predicted: str) -> dict[str, Any]:
    abstained = _is_abstention(predicted)
    return {"abstained": abstained, "correct": abstained}


def _score(question: dict[str, Any], predicted: str) -> dict[str, Any]:
    if question.get("category") == "absence_abstention":
        return _score_absence(predicted)
    return _score_answer(predicted, question["expected_answer"])


# -----------------------------------------------------------------------------
# Per-system evaluation
# -----------------------------------------------------------------------------

def run_system(
    system: SystemAdapter,
    dataset: datasets.Dataset,
    limit: int | None,
    output_dir: Path,
) -> list[dict[str, Any]]:
    print(f"\n{'=' * 68}\n  System: {system.label} ({system.name})\n{'=' * 68}")

    print(f"\n── Ingesting {len(dataset.memories)} memories ──")
    system.ingest(dataset.memories)

    questions = dataset.questions[:limit] if limit else dataset.questions
    print(f"\n── Answering {len(questions)} questions ──")

    queries: list[dict[str, Any]] = []
    for q in questions:
        print(f"\n[{q['id']}] ({q['category']}) {q['question']}")
        response = system.query(q["question"])
        answer = response.get("answer", "")
        score = _score(q, answer)
        marker = "✓" if score.get("correct") else "✗"
        print(f"  expected: {q['expected_answer']}")
        print(f"  got:      {answer}")
        print(f"  {marker} {score}")

        queries.append({
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "expected_winner": q["expected_winner"],
            "predicted_answer": answer,
            "score": score,
            **{k: v for k, v in response.items() if k != "answer"},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset.name,
        "llm_model": _nim.LLM_MODEL,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "limit": limit,
    }
    system.dump_state(queries, output_dir, metadata)
    (output_dir / "summary.md").write_text(_summary_md(system, queries))
    return queries


def _summary_md(system: SystemAdapter, queries: list[dict[str, Any]]) -> str:
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for q in queries:
        by_cat[q["category"]]["total"] += 1
        if q["score"].get("correct"):
            by_cat[q["category"]]["correct"] += 1
    cats = sorted(by_cat)
    expected = {q["category"]: q["expected_winner"] for q in queries}
    total_c = sum(s["correct"] for s in by_cat.values())
    total = sum(s["total"] for s in by_cat.values())
    lines = [
        f"# {system.label} — Results",
        "",
        "## Per-category accuracy",
        "",
        "| Category | Expected winner | correct/total |",
        "|---|---|---|",
    ]
    for cat in cats:
        s = by_cat[cat]
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        lines.append(f"| {cat} | {expected.get(cat, '?')} | "
                     f"{s['correct']}/{s['total']} ({pct:.0f}%) |")
    lines.append(f"| **Total** | — | **{total_c}/{total}** |")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Comparison page (result.html)
# -----------------------------------------------------------------------------

def emit_result_html(
    dataset: datasets.Dataset,
    runs: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Side-by-side per-question + per-category accuracy across all systems."""
    system_names = list(runs.keys())

    # Per-category
    by_cat: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"c": 0, "t": 0}))
    expected: dict[str, str] = {}
    for sys_name, queries in runs.items():
        for q in queries:
            cat = q["category"]
            by_cat[cat][sys_name]["t"] += 1
            if q["score"].get("correct"):
                by_cat[cat][sys_name]["c"] += 1
            expected[cat] = q["expected_winner"]
    cats = sorted(by_cat)

    cat_rows = []
    for cat in cats:
        cells = [f"<td><b>{cat}</b></td>", f"<td>{expected.get(cat, '?')}</td>"]
        winning_sys = None
        best_pct = -1.0
        for sys_name in system_names:
            s = by_cat[cat][sys_name]
            pct = s["c"] / s["t"] * 100 if s["t"] else 0
            if pct > best_pct + 0.0001:
                best_pct = pct
                winning_sys = sys_name
        for sys_name in system_names:
            s = by_cat[cat][sys_name]
            pct = s["c"] / s["t"] * 100 if s["t"] else 0
            cls = ' class="winner"' if sys_name == winning_sys and best_pct > 0 else ""
            cells.append(f'<td{cls}>{s["c"]}/{s["t"]} ({pct:.0f}%)</td>')
        cat_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Per-question
    questions_by_id: dict[str, dict[str, Any]] = {}
    for queries in runs.values():
        for q in queries:
            questions_by_id.setdefault(q["question_id"], q)
    q_rows = []
    for qid in sorted(questions_by_id):
        q = questions_by_id[qid]
        cells = [
            f"<td><code>{qid}</code></td>",
            f"<td>{q['category']}</td>",
            f"<td>{_escape(q['question'])}</td>",
            f"<td>{q['expected_winner']}</td>",
        ]
        for sys_name in system_names:
            qmap = {qq["question_id"]: qq for qq in runs[sys_name]}
            qq = qmap.get(qid)
            if qq is None:
                cells.append("<td>—</td>")
                continue
            ok = "✓" if qq["score"].get("correct") else "✗"
            cls = "ok" if qq["score"].get("correct") else "bad"
            href = f"{sys_name}/html/traces/{qid}.html"
            ans = _escape(_truncate(qq["predicted_answer"], 60))
            cells.append(
                f'<td class="{cls}"><a href="{href}" title="{ans}">{ok}</a></td>'
            )
        q_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Totals
    totals: list[str] = []
    for sys_name in system_names:
        total_c = sum(1 for q in runs[sys_name] if q["score"].get("correct"))
        total = len(runs[sys_name])
        pct = total_c / total * 100 if total else 0
        totals.append(f"<td><b>{total_c}/{total} ({pct:.0f}%)</b></td>")

    sys_th = "".join(f"<th>{systems.get_system(s).label}</th>" for s in system_names)

    html_str = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Comparison — {dataset.name}</title>
<style>
body {{ font: 14px system-ui, sans-serif; max-width: 1300px; margin: 2em auto; padding: 0 1em; color: #1f2937; }}
h1 {{ font-size: 1.5em; }}
h2 {{ font-size: 1.15em; margin-top: 1.5em; }}
table {{ border-collapse: collapse; margin: 1em 0; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
th {{ background: #f9fafb; }}
td.ok a, td.bad a {{ font-size: 1.2em; text-decoration: none; }}
td.ok a {{ color: #16a34a; }}
td.bad a {{ color: #dc2626; }}
td.winner {{ background: #ecfccb; font-weight: bold; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
.meta {{ color: #6b7280; font-size: 0.92em; }}
.links {{ margin: 1em 0; }}
.links a {{ display: inline-block; padding: 6px 12px; background: #2563eb; color: white;
            border-radius: 4px; margin-right: 8px; text-decoration: none; }}
</style></head><body>
<h1>Comparison — <code>{dataset.name}</code></h1>
<p class="meta">
  Dataset: {len(dataset.memories)} memories, {len(dataset.questions)} questions ·
  LLM: <code>{_nim.LLM_MODEL}</code> ·
  Generated: {datetime.now().isoformat(timespec="seconds")}
</p>

<div class="links">
{"".join(f'<a href="{s}/html/index.html">{systems.get_system(s).label} traces →</a>' for s in system_names)}
</div>

<h2>Per-category accuracy</h2>
<p class="meta">Best cell in each row is highlighted green.</p>
<table>
<thead><tr><th>Category</th><th>Expected winner</th>{sys_th}</tr></thead>
<tbody>
{"".join(cat_rows)}
<tr><th colspan="2" style="text-align:right">Total</th>{"".join(totals)}</tr>
</tbody></table>

<h2>Per-question results</h2>
<p class="meta">Click a cell to open that system's trace for the question.</p>
<table>
<thead><tr><th>Q</th><th>Category</th><th>Question</th><th>Expected winner</th>{sys_th}</tr></thead>
<tbody>
{"".join(q_rows)}
</tbody></table>
</body></html>"""
    (output_dir / "result.html").write_text(html_str)


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _require_api_key() -> None:
    if "NVIDIA_API_KEY" not in os.environ:
        print("ERROR: set NVIDIA_API_KEY first.\n  export NVIDIA_API_KEY=nvapi-…")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="comparison",
                        help=f"Dataset to use. Registered: {sorted(datasets.DATASETS)}")
    parser.add_argument("--systems", default="amem,hipporag,hipporag2",
                        help=f"Comma-separated. Registered: {sorted(systems.SYSTEMS)}")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions (smoke test).")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"), type=Path)
    args = parser.parse_args()

    _require_api_key()

    dataset = datasets.load(args.dataset)
    selected = [s.strip() for s in args.systems.split(",") if s.strip()]
    for name in selected:
        if name not in systems.SYSTEMS:
            raise SystemExit(f"Unknown system {name!r}. "
                             f"Available: {sorted(systems.SYSTEMS)}")

    out_root = args.output_dir / dataset.name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}/")

    all_runs: dict[str, list[dict[str, Any]]] = {}
    for name in selected:
        system_cls = systems.get_system(name)
        system = system_cls()
        system_dir = out_root / name
        all_runs[name] = run_system(system, dataset, args.limit, system_dir)

    print(f"\n{'=' * 68}\n  Emitting comparison page\n{'=' * 68}")
    emit_result_html(dataset, all_runs, out_root)
    print(f"  → {out_root / 'result.html'}")


if __name__ == "__main__":
    main()
