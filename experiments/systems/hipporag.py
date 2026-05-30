"""
HippoRAG v1 adapter for the comparison harness.

Ports the indexing + PPR retrieval from
https://github.com/pandazxx/hipporag-reproduction/blob/main/experiments/demo.py
into a class that fits the SystemAdapter interface. A reader LLM call
synthesises a free-text answer from the top-K retrieved passages — the
original demo only displayed retrieved passages without generating answers.

Deviations from the paper preserved from the reference implementation:
  [DEVIATION-4] ANN: brute-force numpy dot instead of FAISS IndexFlat.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .. import _nim
from .base import SystemAdapter


# ── Reader prompt (shared style with A-Mem) ────────────────────────────────

_READER_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Use only the provided passages."
)


def _read(question: str, passages: list[tuple[str, str]]) -> str:
    """passages is list of (id, content) tuples."""
    if not passages:
        return "unknown / not mentioned"
    ctx = "\n".join(f"- [{pid}] {content}" for pid, content in passages)
    prompt = (
        f"Answer the question using only the passages below.\n"
        f"If the answer is not contained in them, reply exactly: "
        f"\"unknown / not mentioned\".\n\n"
        f"Passages:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely in one short sentence."
    )
    return _nim.chat_text(prompt, system=_READER_SYSTEM)


# ── PPR (power iteration) ──────────────────────────────────────────────────

def _personalized_pagerank(
    adj: sp.csr_matrix,
    seed_indices: list[int],
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    N = adj.shape[0]
    row_sums = np.array(adj.sum(axis=1), dtype=np.float64).flatten()
    row_sums[row_sums == 0] = 1.0
    T = sp.diags(1.0 / row_sums) @ adj

    s = np.zeros(N, dtype=np.float64)
    if seed_indices:
        s[seed_indices] = 1.0 / len(seed_indices)

    r = s.copy()
    for _ in range(max_iter):
        r_new = (1.0 - alpha) * T.T.dot(r) + alpha * s
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    return r


# ── Adapter ────────────────────────────────────────────────────────────────

class HippoRAGAdapter(SystemAdapter):
    name = "hipporag"
    label = "HippoRAG v1"

    def __init__(self, sim_threshold: float = 0.8) -> None:
        self.sim_threshold = sim_threshold
        self.item_ids: list[str] = []
        self.passages: list[str] = []
        self.index: dict[str, Any] = {}

    # -- indexing --

    def ingest(self, items: list[dict[str, Any]]) -> None:
        self.item_ids = [m["id"] for m in items]
        self.passages = [m["content"] for m in items]
        P = len(self.passages)

        print(f"  HippoRAG ingest: {P} passages")
        print(f"  Phase 1 — OpenIE  (NIM LLM)")
        triples_per_passage: dict[int, list[tuple]] = {}
        for i, passage in enumerate(self.passages):
            triples = _nim.extract_triples(passage)
            triples_per_passage[i] = triples
            print(f"    [{i+1:2d}/{P}] {self.item_ids[i]}: {len(triples)} triple(s)")

        print(f"  Phase 2 — Build entity index")
        entity_to_passages: dict[str, set[int]] = defaultdict(set)
        for pidx, triples in triples_per_passage.items():
            for subj, _pred, obj in triples:
                entity_to_passages[subj].add(pidx)
                entity_to_passages[obj].add(pidx)

        entities = sorted(entity_to_passages.keys())
        entity_idx = {e: i for i, e in enumerate(entities)}
        N = len(entities)
        print(f"    Unique entities : {N}")

        embeddings = _nim.embed_batch(entities) if entities else np.zeros((0, 1024))

        graph: dict[tuple, float] = defaultdict(float)
        for triples in triples_per_passage.values():
            for subj, _pred, obj in triples:
                si, oi = entity_idx[subj], entity_idx[obj]
                graph[(si, oi)] += 1.0
                graph[(oi, si)] += 1.0
        triple_edges = len(graph) // 2

        sims = embeddings @ embeddings.T if N > 0 else np.zeros((0, 0))
        syn_count = 0
        for i in range(N):
            for j in range(i + 1, N):
                if sims[i, j] >= self.sim_threshold:
                    graph[(i, j)] = sims[i, j]
                    graph[(j, i)] = sims[i, j]
                    syn_count += 1
        print(f"    Triple edges    : {triple_edges}")
        print(f"    Synonymy edges  : {syn_count}  (cosine ≥ {self.sim_threshold})")

        rows, cols, data = [], [], []
        for (i, j), w in graph.items():
            rows.append(i); cols.append(j); data.append(w)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)

        specificity = np.array(
            [1.0 / len(entity_to_passages[e]) for e in entities], dtype=np.float64
        ) if N > 0 else np.zeros(0)

        pr, pc = [], []
        for e, pidxs in entity_to_passages.items():
            ei = entity_idx[e]
            for pidx in pidxs:
                pr.append(ei); pc.append(pidx)
        P_matrix = sp.csr_matrix(
            (np.ones(len(pr), dtype=np.float64), (pr, pc)), shape=(max(N, 1), P)
        )

        self.index = dict(
            entities=entities,
            entity_idx=entity_idx,
            embeddings=embeddings,
            adj=adj,
            specificity=specificity,
            P_matrix=P_matrix,
            triples_per_passage=triples_per_passage,
        )

    # -- query --

    def query(self, question: str) -> dict[str, Any]:
        q_ents = _nim.extract_query_entities(question)
        entities = self.index.get("entities", [])
        embeddings = self.index.get("embeddings")
        adj = self.index.get("adj")

        seed_indices: list[int] = []
        seed_trace: list[dict[str, Any]] = []
        if entities and embeddings is not None and len(entities) > 0:
            for qe in q_ents:
                qe_emb = _nim.embed(qe)
                sims = embeddings @ qe_emb
                best_idx = int(np.argmax(sims))
                seed_trace.append({
                    "query_entity": qe,
                    "matched": entities[best_idx],
                    "similarity": round(float(sims[best_idx]), 3),
                })
                if best_idx not in seed_indices:
                    seed_indices.append(best_idx)

        if adj is None or adj.shape[0] == 0 or not seed_indices:
            top: list[tuple[int, float]] = []
        else:
            ppr = _personalized_pagerank(adj, seed_indices)
            weighted = ppr * self.index["specificity"]
            scores = np.array(
                self.index["P_matrix"].T.dot(weighted), dtype=np.float64,
            ).flatten()
            order = np.argsort(-scores)[:5]
            top = [(int(i), float(scores[i])) for i in order if scores[i] > 0]

        retrieved_ids = [self.item_ids[pidx] for pidx, _ in top]
        passages = [(self.item_ids[pidx], self.passages[pidx]) for pidx, _ in top]
        answer = _read(question, passages)

        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieval_scores": [round(s, 5) for _, s in top],
            "query_entities": q_ents,
            "seed_trace": seed_trace,
        }

    # -- persist + render --

    def dump_state(
        self,
        queries: list[dict[str, Any]],
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        meta = {
            **metadata,
            "system": self.name,
            "system_label": self.label,
            "sim_threshold": self.sim_threshold,
            "n_entities": len(self.index.get("entities", [])),
        }
        items = [
            {"id": pid, "content": txt, "triples": list(self.index["triples_per_passage"].get(i, []))}
            for i, (pid, txt) in enumerate(zip(self.item_ids, self.passages))
        ]
        payload = {"metadata": meta, "items": items, "queries": queries}
        (output_dir / "run.json").write_text(json.dumps(payload, indent=2))

        _write_retrieval_html(output_dir, queries, items, self.label)


# ── Shared trace renderer (used by both HippoRAG adapters) ────────────────

def _write_retrieval_html(
    output_dir: Path,
    queries: list[dict[str, Any]],
    items: list[dict[str, Any]],
    label: str,
) -> None:
    html_dir = output_dir / "html"
    traces_dir = html_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    items_by_id = {it["id"]: it for it in items}

    for q in queries:
        retrieved = [items_by_id.get(rid, {"id": rid, "content": "?"})
                     for rid in q.get("retrieved_ids", [])]
        rows = "".join(
            f'<tr><td>{i+1}</td><td><code>{r["id"]}</code></td><td>{_escape(r["content"])}</td></tr>'
            for i, r in enumerate(retrieved)
        )
        extras = []
        if q.get("query_entities"):
            extras.append(f"<p><b>Query entities:</b> <code>{_escape(', '.join(q['query_entities']))}</code></p>")
        if q.get("seed_trace"):
            seeds_html = "".join(
                f"<li><code>{_escape(s['query_entity'])}</code> → "
                f"<code>{_escape(s['matched'])}</code> "
                f"(sim={s['similarity']})</li>"
                for s in q["seed_trace"]
            )
            extras.append(f"<p><b>Seed entities:</b></p><ul>{seeds_html}</ul>")
        if q.get("filtered_triples"):
            triples_html = "".join(
                f"<li><code>({_escape(t[0])}, {_escape(t[1])}, {_escape(t[2])})</code></li>"
                for t in q["filtered_triples"]
            )
            extras.append(f"<p><b>Recognition memory kept:</b></p><ul>{triples_html}</ul>")
        ok = "✓" if q.get("score", {}).get("correct") else "✗"
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{label} — {q['question_id']}</title>
<style>
body {{ font: 14px system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
h1 {{ font-size: 1.3em; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
th {{ background: #f9fafb; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
.ok {{ color: #16a34a; }} .bad {{ color: #dc2626; }}
</style></head><body>
<h1>{label} — {q['question_id']} ({q.get('category', '?')}) {ok}</h1>
<p><b>Q:</b> {_escape(q['question'])}</p>
<p><b>Expected:</b> {_escape(q.get('expected_answer', ''))}</p>
<p><b>Got:</b> {_escape(q.get('predicted_answer', ''))}</p>
{"".join(extras)}
<h2>Top retrieved items</h2>
<table><thead><tr><th>#</th><th>id</th><th>content</th></tr></thead><tbody>
{rows}
</tbody></table>
<p><a href="../index.html">← back to index</a></p>
</body></html>"""
        (traces_dir / f"{q['question_id']}.html").write_text(page)

    (html_dir / "index.html").write_text(_index_html(label, queries))


def _index_html(label: str, queries: list[dict[str, Any]]) -> str:
    rows = []
    for q in queries:
        ok = "✓" if q.get("score", {}).get("correct") else "✗"
        rows.append(
            f'<tr><td><a href="traces/{q["question_id"]}.html">{q["question_id"]}</a></td>'
            f'<td>{q.get("category", "")}</td>'
            f'<td>{_escape(q.get("question", ""))}</td>'
            f'<td style="text-align:center">{ok}</td></tr>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{label} — traces</title>
<style>
body {{ font: 14px system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
h1 {{ font-size: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
th {{ background: #f9fafb; }}
a {{ color: #2563eb; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head><body>
<h1>{label} — per-question traces</h1>
<table>
<thead><tr><th>Q</th><th>Category</th><th>Question</th><th>Correct?</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
</body></html>"""


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
