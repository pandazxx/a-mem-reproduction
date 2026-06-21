#!/usr/bin/env python3
"""
Experiment runner — the framework's main entry point (issue #4).

Runs one or more *experiments* — each a ``(system, parameter set)`` pair —
against a single ``(dataset, test-set)`` and emits a side-by-side comparison.

An experiment is identified on disk by ``<system>_<paramset>`` and produces:

    results/<dataset>_<testset>/<system>_<paramset>/
        memory/index.html      memory-structure visualisation
        result/result.json     frozen run (memory + per-query results)
        result/index.html      query-result visualisation (+ traces/)
        result/summary.md      per-category answer + retrieval scores
    results/<dataset>_<testset>/index.html   comparison across experiments

Datasets come from ``experiments/datasets/`` (statements), test-sets from
``experiments/testsets/`` (queries with ``required_retrieval``), parameter
sets from ``experiments/params/<system>/``. See ``experiments/schema.py``.

Each query is scored two ways:
  - answer quality   : predicted vs ``expected_answer`` (F1 / substring /
                       abstention)
  - retrieval quality: ``retrieved_ids`` vs the test-set's
                       ``required_retrieval`` (precision / recall / F1 / hit)

Usage:
    just compare                                  # comparison⨯comparison, default params
    just compare -- --experiments amem:wide,hipporag:default
    just compare -- --dataset comparison --testset comparison
    just compare -- --limit 5                     # smoke test (first 5 queries)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import _nim, schema
from .schema import Experiment
from .systems import SYSTEMS, get_system
from .systems.base import SystemAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Answer scoring
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


def _score_answer_for(query: dict[str, Any], predicted: str) -> dict[str, Any]:
    if query.get("category") == "absence_abstention":
        return _score_absence(predicted)
    return _score_answer(predicted, query.get("expected_answer", ""))


# -----------------------------------------------------------------------------
# Retrieval scoring (retrieved_ids vs required_retrieval)
# -----------------------------------------------------------------------------

def _score_retrieval(
    retrieved_ids: list[str],
    required: list[str],
) -> dict[str, Any]:
    """
    Compare surfaced statement IDs against the test-set's required set.

    Queries with no ``required_retrieval`` (abstention / trick questions)
    have nothing to retrieve, so recall/precision are ``None`` and ``hit``
    is True (vacuously satisfied) — these don't penalise the retrieval score.
    """
    req = set(required)
    ret = set(retrieved_ids)
    if not req:
        return {
            "required": [],
            "hits": [],
            "recall": None,
            "precision": None,
            "f1": None,
            "hit": True,
        }
    hits = req & ret
    recall = len(hits) / len(req)
    precision = len(hits) / len(ret) if ret else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "required": sorted(req),
        "hits": sorted(hits),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "hit": req <= ret,
    }


# -----------------------------------------------------------------------------
# Per-experiment run
# -----------------------------------------------------------------------------

def run_experiment(
    experiment: Experiment,
    dataset: schema.Dataset,
    testset: schema.TestSet,
    limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    paramset = schema.load_params(experiment.system, experiment.paramset)
    system_cls = get_system(experiment.system)
    system = system_cls(paramset.values)

    print(f"\n{'=' * 68}\n  Experiment: {experiment.slug}  "
          f"({system.label}, params={paramset.values})\n{'=' * 68}")

    items = dataset.ingest_items()
    print(f"\n── Ingesting {len(items)} statements ──")
    system.ingest(items)

    queries = testset.queries[:limit] if limit else testset.queries
    print(f"\n── Answering {len(queries)} queries ──")

    results: list[dict[str, Any]] = []
    for q in queries:
        print(f"\n[{q['id']}] ({q.get('category', '?')}) {q['query']}")
        response = system.query(q["query"])
        answer = response.get("answer", "")
        retrieved_ids = response.get("retrieved_ids", [])
        required = q.get("required_retrieval", [])
        ascore = _score_answer_for(q, answer)
        rscore = _score_retrieval(retrieved_ids, required)
        marker = "✓" if ascore.get("correct") else "✗"
        rec = "n/a" if rscore["recall"] is None else f"{rscore['recall']:.0%}"
        print(f"  expected: {q.get('expected_answer', '')}")
        print(f"  got:      {answer}")
        print(f"  {marker} answer={ascore}  retrieval_recall={rec}")

        results.append({
            "question_id": q["id"],
            "category": q.get("category", ""),
            "question": q["query"],
            "expected_answer": q.get("expected_answer", ""),
            "expected_winner": q.get("expected_winner", ""),
            "required_retrieval": required,
            "predicted_answer": answer,
            "score": ascore,
            "retrieval_score": rscore,
            **{k: v for k, v in response.items() if k != "answer"},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset.name,
        "testset": testset.name,
        "system": experiment.system,
        "paramset": experiment.paramset,
        "params": paramset.values,
        "llm_model": _nim.LLM_MODEL,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "limit": limit,
    }
    system.dump_state(results, output_dir, metadata)
    (output_dir / "result" / "summary.md").write_text(_summary_md(experiment, system, results))
    return {
        "slug": experiment.slug,
        "system": experiment.system,
        "paramset": experiment.paramset,
        "label": system.label,
        "queries": results,
    }


def _mean_recall(queries: list[dict[str, Any]]) -> float | None:
    vals = [q["retrieval_score"]["recall"] for q in queries
            if q.get("retrieval_score", {}).get("recall") is not None]
    return sum(vals) / len(vals) if vals else None


def _summary_md(
    experiment: Experiment,
    system: SystemAdapter,
    queries: list[dict[str, Any]],
) -> str:
    by_cat: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "recall_sum": 0.0, "recall_n": 0}
    )
    for q in queries:
        c = by_cat[q["category"]]
        c["total"] += 1
        if q["score"].get("correct"):
            c["correct"] += 1
        recall = q.get("retrieval_score", {}).get("recall")
        if recall is not None:
            c["recall_sum"] += recall
            c["recall_n"] += 1
    expected = {q["category"]: q["expected_winner"] for q in queries}
    total_c = sum(c["correct"] for c in by_cat.values())
    total = sum(c["total"] for c in by_cat.values())
    mean_recall = _mean_recall(queries)
    mr = "n/a" if mean_recall is None else f"{mean_recall:.0%}"
    lines = [
        f"# {system.label} — {experiment.slug}",
        "",
        f"Parameters: `{system.params}`",
        "",
        "## Per-category scores",
        "",
        "| Category | Expected winner | answer correct/total | mean retrieval recall |",
        "|---|---|---|---|",
    ]
    for cat in sorted(by_cat):
        c = by_cat[cat]
        pct = c["correct"] / c["total"] * 100 if c["total"] else 0
        crec = f"{c['recall_sum'] / c['recall_n']:.0%}" if c["recall_n"] else "n/a"
        lines.append(f"| {cat} | {expected.get(cat, '?')} | "
                     f"{c['correct']}/{c['total']} ({pct:.0f}%) | {crec} |")
    lines.append(f"| **Total** | — | **{total_c}/{total}** | **{mr}** |")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Comparison page (index.html)
# -----------------------------------------------------------------------------

def emit_comparison_html(
    name: str,
    runs: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Side-by-side per-question + per-category accuracy across experiments.

    ``runs`` is a list of ``{slug, system, paramset, label, queries}``.
    """
    slugs = [r["slug"] for r in runs]
    by_slug = {r["slug"]: r for r in runs}
    labels = {r["slug"]: f'{r["label"]} <span class="ps">{r["paramset"]}</span>'
              for r in runs}

    # ── Per-category answer accuracy ──
    by_cat: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"c": 0, "t": 0})
    )
    expected: dict[str, str] = {}
    for r in runs:
        for q in r["queries"]:
            cat = q["category"]
            by_cat[cat][r["slug"]]["t"] += 1
            if q["score"].get("correct"):
                by_cat[cat][r["slug"]]["c"] += 1
            expected[cat] = q.get("expected_winner", "")
    cats = sorted(by_cat)

    cat_rows = []
    for cat in cats:
        cells = [f"<td><b>{cat}</b></td>", f"<td>{expected.get(cat, '?')}</td>"]
        best_pct, winner = -1.0, None
        for slug in slugs:
            s = by_cat[cat][slug]
            pct = s["c"] / s["t"] * 100 if s["t"] else 0
            if pct > best_pct + 0.0001:
                best_pct, winner = pct, slug
        for slug in slugs:
            s = by_cat[cat][slug]
            pct = s["c"] / s["t"] * 100 if s["t"] else 0
            cls = ' class="winner"' if slug == winner and best_pct > 0 else ""
            cells.append(f'<td{cls}>{s["c"]}/{s["t"]} ({pct:.0f}%)</td>')
        cat_rows.append("<tr>" + "".join(cells) + "</tr>")

    # ── Per-question grid (answer ✓/✗ + retrieval recall) ──
    questions_by_id: dict[str, dict[str, Any]] = {}
    for r in runs:
        for q in r["queries"]:
            questions_by_id.setdefault(q["question_id"], q)
    q_rows = []
    for qid in sorted(questions_by_id):
        q = questions_by_id[qid]
        cells = [
            f"<td><code>{qid}</code></td>",
            f"<td>{q['category']}</td>",
            f"<td>{_escape(q['question'])}</td>",
            f"<td>{q.get('expected_winner', '')}</td>",
        ]
        for slug in slugs:
            qmap = {qq["question_id"]: qq for qq in by_slug[slug]["queries"]}
            qq = qmap.get(qid)
            if qq is None:
                cells.append("<td>—</td>")
                continue
            ok = "✓" if qq["score"].get("correct") else "✗"
            cls = "ok" if qq["score"].get("correct") else "bad"
            href = f"{slug}/result/traces/{qid}.html"
            ans = _escape(_truncate(qq["predicted_answer"], 60))
            rs = qq.get("retrieval_score", {})
            rec = "" if rs.get("recall") is None else f' <span class="rec">{rs["recall"]:.0%}</span>'
            cells.append(
                f'<td class="{cls}"><a href="{href}" title="{ans}">{ok}</a>{rec}</td>'
            )
        q_rows.append("<tr>" + "".join(cells) + "</tr>")

    # ── Totals (answer accuracy + mean retrieval recall) ──
    totals, rec_totals = [], []
    for slug in slugs:
        qs = by_slug[slug]["queries"]
        total_c = sum(1 for q in qs if q["score"].get("correct"))
        total = len(qs)
        pct = total_c / total * 100 if total else 0
        totals.append(f"<td><b>{total_c}/{total} ({pct:.0f}%)</b></td>")
        mr = _mean_recall(qs)
        rec_totals.append(f"<td>{'n/a' if mr is None else f'{mr:.0%}'}</td>")

    sys_th = "".join(f"<th>{labels[s]}</th>" for s in slugs)
    n_q = len(questions_by_id)

    html_str = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Comparison — {name}</title>
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
.ps {{ color: #6b7280; font-weight: normal; font-size: 0.85em; }}
.rec {{ color: #6b7280; font-size: 0.82em; }}
.meta {{ color: #6b7280; font-size: 0.92em; }}
.links {{ margin: 1em 0; }}
.links a {{ display: inline-block; padding: 6px 12px; background: #2563eb; color: white;
            border-radius: 4px; margin-right: 8px; text-decoration: none; }}
</style></head><body>
<h1>Comparison — <code>{name}</code></h1>
<p class="meta">
  {n_q} queries · {len(runs)} experiments ·
  LLM: <code>{_nim.LLM_MODEL}</code> ·
  Generated: {datetime.now().isoformat(timespec="seconds")}
</p>

<div class="links">
{"".join(f'<a href="{s}/result/index.html">{by_slug[s]["label"]} {by_slug[s]["paramset"]} results →</a>' for s in slugs)}
</div>

<h2>Per-category answer accuracy</h2>
<p class="meta">Best cell in each row is highlighted green.</p>
<table>
<thead><tr><th>Category</th><th>Expected winner</th>{sys_th}</tr></thead>
<tbody>
{"".join(cat_rows)}
<tr><th colspan="2" style="text-align:right">Answer total</th>{"".join(totals)}</tr>
<tr><th colspan="2" style="text-align:right">Mean retrieval recall</th>{"".join(rec_totals)}</tr>
</tbody></table>

<h2>Per-question results</h2>
<p class="meta">✓/✗ = answer correctness (click to open the trace); small % = retrieval recall.</p>
<table>
<thead><tr><th>Q</th><th>Category</th><th>Question</th><th>Expected winner</th>{sys_th}</tr></thead>
<tbody>
{"".join(q_rows)}
</tbody></table>
</body></html>"""
    (output_dir / "index.html").write_text(html_str)


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
                        help=f"Dataset. Available: {schema.available_datasets()}")
    parser.add_argument("--testset", default=None,
                        help="Test-set. Defaults to the dataset name. "
                             f"Available: {schema.available_testsets()}")
    parser.add_argument("--experiments", default="amem:default,hipporag:default,hipporag2:default",
                        help="Comma-separated system[:paramset] tokens. "
                             f"Systems: {sorted(SYSTEMS)}")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N queries (smoke test).")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"), type=Path)
    args = parser.parse_args()

    _require_api_key()

    testset_name = args.testset or args.dataset
    dataset = schema.load_dataset(args.dataset)
    testset = schema.load_testset(testset_name)

    experiments = [Experiment.parse(tok) for tok in args.experiments.split(",") if tok.strip()]
    for e in experiments:
        if e.system not in SYSTEMS:
            raise SystemExit(f"Unknown system {e.system!r}. Available: {sorted(SYSTEMS)}")
        schema.load_params(e.system, e.paramset)  # validate paramset exists

    cmp_name = schema.comparison_dirname(dataset.name, testset.name)
    out_root = args.output_dir / cmp_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}/")

    runs: list[dict[str, Any]] = []
    for e in experiments:
        runs.append(run_experiment(e, dataset, testset, args.limit, out_root / e.slug))

    print(f"\n{'=' * 68}\n  Emitting comparison page\n{'=' * 68}")
    emit_comparison_html(cmp_name, runs, out_root)
    print(f"  → {out_root / 'index.html'}")


if __name__ == "__main__":
    main()
