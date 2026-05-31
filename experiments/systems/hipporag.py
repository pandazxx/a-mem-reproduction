"""
HippoRAG v1 adapter for the comparison harness.

Ports the indexing + PPR retrieval from
https://github.com/pandazxx/hipporag-reproduction/blob/main/experiments/demo.py
into a class that fits the SystemAdapter interface. A reader LLM call
synthesises a free-text answer from the top-K retrieved passages — the
original demo only displayed retrieved passages without generating answers.

Algorithm summary (Jiménez et al. 2024, NeurIPS — "HippoRAG")
-------------------------------------------------------------
Inspired by hippocampal indexing. Offline indexing:

    passages → OpenIE → (entity, relation, entity) triples → KG
    + synonymy edges where entity-embedding cosine ≥ threshold

Online retrieval:

    query → NER → entity seeds
          → Personalized PageRank over the KG
          → node-specificity weighting (down-weight popular entities)
          → project PPR scores onto passages via entity-to-passage matrix
          → top-K passages → reader LLM → answer

Why PPR? It propagates retrieval probability mass along KG edges, so a
2-or-3-hop chain (e.g. "researcher" → "Bob Martinez" → "Alice Chen")
can surface a passage whose text doesn't overlap the query — something
naive vector search can't do.

Deviations from the paper preserved from the reference implementation:
  [DEVIATION-4] ANN: brute-force numpy dot instead of FAISS IndexFlat.

Inputs and outputs (SystemAdapter contract)
-------------------------------------------
``ingest(items)``
    items : list[dict] with keys ``id``, ``content``, ``timestamp?``.
    Side effect: populates ``self.index`` with the KG, entity
    embeddings, sparse adjacency, specificity, and the
    entity-to-passage presence matrix.

``query(question)`` returns:
    {
      "answer"            : str,
      "retrieved_ids"     : list[str],            # passage IDs, top-5
      "retrieval_scores"  : list[float],
      "query_entities"    : list[str],            # raw NER output
      "seed_trace"        : list[dict],           # query entity → KG entity match
    }
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
    """
    Reader LLM. Synthesises a free-text answer from the retrieved passages.

    Inputs
        question : str
        passages : list[(id, content)] — top-K passages from retrieval,
                   in score-descending order.

    Output
        str — one-sentence answer, or ``"unknown / not mentioned"`` if
        the model decides no passage contains the answer (lets the
        scorer's abstention heuristic catch it).

    Shared with HippoRAG2Adapter via ``from .hipporag import _read``.
    """
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
    """
    Personalized PageRank via power iteration.

    Solves     r = (1 - α) Tᵀ r + α s
    where      T = row-stochastic transition matrix (each row sums to 1)
               s = uniform distribution over seed nodes (and 0 elsewhere)
               α = teleport probability back to s on each step

    α=0.15 (paper uses 0.10) is the standard choice. Higher α → faster
    convergence + less far-hop propagation. We keep it loose enough to
    let probability flow 2-3 hops out, which is what multi-hop QA needs.

    Inputs
        adj          : sparse (N, N) symmetric weighted adjacency
        seed_indices : list[int]  — indices of nodes that get the
                                    initial probability mass (uniform
                                    over the list)

    Output
        ndarray, shape (N,), dtype float64
            r[i] = stationary probability of node i — interpret as
            "retrieval relevance" of that entity given the seeds.

    Edge cases
        - Empty seed_indices → r stays at 0 everywhere. Caller checks
          this before invoking us.
        - Isolated nodes (row sum 0) get their row sum clamped to 1.0
          so we don't divide by zero; in practice they contribute
          nothing because their outbound transition is the zero row.
    """
    N = adj.shape[0]
    # Row-stochastic transition matrix T = D⁻¹ · adj.
    # We use a diagonal sparse matrix for D⁻¹ so the multiply stays sparse.
    row_sums = np.array(adj.sum(axis=1), dtype=np.float64).flatten()
    row_sums[row_sums == 0] = 1.0
    T = sp.diags(1.0 / row_sums) @ adj

    # Seed distribution s. Uniform over the seed indices; zero elsewhere.
    s = np.zeros(N, dtype=np.float64)
    if seed_indices:
        s[seed_indices] = 1.0 / len(seed_indices)

    # Power iteration. Converges geometrically with rate (1-α).
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
        """
        Build the HippoRAG index in 7 steps. Mutates ``self.index``.

        Steps
        -----
        1. OpenIE per passage
             For each passage, two LLM calls (NER then OpenIE
             conditioned on those entities) produce a list of
             ``(subject, predicate, object)`` string triples.

             Cost  : 2 LLM calls × P passages.
             Shape : triples_per_passage[pidx] = list[(s, p, o)]

        2. Entity index
             Collapse all subjects + objects across all triples into a
             unique sorted entity list. Build entity → integer index map.

             Shape : entities: list[str] of length N
                     entity_idx: dict[str → int]
                     entity_to_passages[e]: set[int]   (which passages
                                                        mention entity e)

        3. Entity embeddings
             One batched NIM embedding call over all N entities.

             Shape : embeddings: (N, D) float64, L2-normalised
                     (D = 1024 for nv-embedqa-e5-v5)

        4. Triple edges
             Every (s, o) pair from every triple adds 1.0 to both
             directions of the symmetric adjacency. Duplicates
             accumulate (heavier edges for repeated relations).

             Shape : graph[(si, oi)] = weight

        5. Synonymy edges
             Brute-force compute the full (N, N) cosine-similarity
             matrix and add an edge anywhere it exceeds
             ``sim_threshold``. The paper uses FAISS IndexFlat — we use
             ``embeddings @ embeddings.T`` because N is small here.
             [DEVIATION-4]

        6. Sparse adjacency
             Convert the ``graph`` dict into a CSR sparse matrix so PPR
             can do efficient matrix-vector multiplies.

             Shape : adj: csr_matrix (N, N) float64

        7. Specificity + entity-to-passage matrix
             - ``specificity[e] = 1 / |passages containing e|``
               Down-weights popular entities at query time so common
               glue words don't dominate PPR rankings.
             - ``P_matrix[e, p] = 1`` iff entity e appears in passage p.
               Used at query time to convert per-entity PPR scores into
               per-passage scores.

             Shape : specificity: ndarray (N,) float64
                     P_matrix:    csr_matrix (N, P) float64
        """
        self.item_ids = [m["id"] for m in items]
        self.passages = [m["content"] for m in items]
        P = len(self.passages)

        # ─── Step 1: OpenIE ──────────────────────────────────────────
        print(f"  HippoRAG ingest: {P} passages")
        print(f"  Phase 1 — OpenIE  (NIM LLM)")
        triples_per_passage: dict[int, list[tuple]] = {}
        for i, passage in enumerate(self.passages):
            triples = _nim.extract_triples(passage)
            triples_per_passage[i] = triples
            print(f"    [{i+1:2d}/{P}] {self.item_ids[i]}: {len(triples)} triple(s)")

        # ─── Step 2: Entity index ────────────────────────────────────
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

        # ─── Step 3: Entity embeddings (one batched NIM call) ────────
        embeddings = _nim.embed_batch(entities) if entities else np.zeros((0, 1024))

        # ─── Step 4: Triple edges ────────────────────────────────────
        # Symmetric weighted edges; heavier for repeated relations.
        graph: dict[tuple, float] = defaultdict(float)
        for triples in triples_per_passage.values():
            for subj, _pred, obj in triples:
                si, oi = entity_idx[subj], entity_idx[obj]
                graph[(si, oi)] += 1.0
                graph[(oi, si)] += 1.0
        triple_edges = len(graph) // 2

        # ─── Step 5: Synonymy edges (cosine ≥ threshold) ─────────────
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

        # ─── Step 6: Sparse CSR adjacency ────────────────────────────
        rows, cols, data = [], [], []
        for (i, j), w in graph.items():
            rows.append(i); cols.append(j); data.append(w)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)

        # ─── Step 7: Specificity + entity-to-passage projection ──────
        specificity = np.array(
            [1.0 / len(entity_to_passages[e]) for e in entities], dtype=np.float64
        ) if N > 0 else np.zeros(0)

        # P_matrix is a sparse (N, P) incidence matrix; multiplying its
        # transpose against an (N,) score vector yields per-passage scores.
        pr, pc = [], []
        for e, pidxs in entity_to_passages.items():
            ei = entity_idx[e]
            for pidx in pidxs:
                pr.append(ei); pc.append(pidx)
        P_matrix = sp.csr_matrix(
            (np.ones(len(pr), dtype=np.float64), (pr, pc)), shape=(max(N, 1), P)
        )

        self.index = dict(
            entities=entities,                 # list[str], length N
            entity_idx=entity_idx,             # dict[str → int]
            embeddings=embeddings,             # (N, D) float64
            adj=adj,                           # csr (N, N) float64
            specificity=specificity,           # (N,) float64
            P_matrix=P_matrix,                 # csr (N, P) float64
            triples_per_passage=triples_per_passage,
        )

    # -- query --

    def query(self, question: str) -> dict[str, Any]:
        """
        Retrieve top-5 passages and synthesise an answer.

        Steps
        -----
        1. Query NER
             LLM extracts entities + key concepts from the question.

             Cost  : 1 LLM call.
             Shape : q_ents: list[str]

        2. Seed matching (query entity → KG entity)
             For each query entity, embed it via NIM and find the
             nearest KG entity by cosine similarity. That index becomes
             a PPR seed. We log the (query entity, matched KG entity,
             similarity) tuple for diagnostic display.

             Cost  : len(q_ents) NIM embedding calls.
             Shape : seed_indices: list[int]  (deduped)
                     seed_trace:   list[dict]

        3. Personalized PageRank
             Power iteration over the KG seeded by the matched entities.

             Shape : ppr: (N,) float64  — per-entity relevance

        4. Specificity weighting
             ``weighted = ppr * specificity`` — popular entities (high
             passage count → low specificity) get suppressed.

             Shape : weighted: (N,) float64

        5. Project onto passages
             ``scores = P_matrixᵀ · weighted`` — sum the weighted PPR
             score over every entity that appears in each passage.

             Shape : scores: (P,) float64

        6. Top-K + reader LLM
             Top-5 passage indices by ``scores`` → fetch the original
             text → reader LLM synthesises the final answer.

             Cost  : 1 LLM call.

        Output (SystemAdapter contract)
            {
              "answer"           : str,
              "retrieved_ids"    : list[str],         # top-5 passage IDs
              "retrieval_scores" : list[float],
              "query_entities"   : list[str],         # raw NER output
              "seed_trace"       : list[dict],        # per-NER-token match
            }
        """
        # ─── Step 1: query NER ──────────────────────────────────────
        q_ents = _nim.extract_query_entities(question)
        entities = self.index.get("entities", [])
        embeddings = self.index.get("embeddings")
        adj = self.index.get("adj")

        # ─── Step 2: match query entities to KG nodes ───────────────
        seed_indices: list[int] = []
        seed_trace: list[dict[str, Any]] = []
        if entities and embeddings is not None and len(entities) > 0:
            for qe in q_ents:
                qe_emb = _nim.embed(qe)                  # (D,)
                sims = embeddings @ qe_emb               # (N,)
                best_idx = int(np.argmax(sims))
                seed_trace.append({
                    "query_entity": qe,
                    "matched": entities[best_idx],
                    "similarity": round(float(sims[best_idx]), 3),
                })
                if best_idx not in seed_indices:
                    seed_indices.append(best_idx)

        # ─── Steps 3-5: PPR → specificity → passage projection ──────
        if adj is None or adj.shape[0] == 0 or not seed_indices:
            top: list[tuple[int, float]] = []
        else:
            ppr = _personalized_pagerank(adj, seed_indices)         # (N,)
            weighted = ppr * self.index["specificity"]              # (N,)
            scores = np.array(
                self.index["P_matrix"].T.dot(weighted), dtype=np.float64,
            ).flatten()                                             # (P,)
            order = np.argsort(-scores)[:5]
            top = [(int(i), float(scores[i])) for i in order if scores[i] > 0]

        # ─── Step 6: top-K passages → reader LLM ────────────────────
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

    @classmethod
    def render_from_run(cls, run_path: Path, output_dir: Path) -> None:
        data = json.loads(run_path.read_text())
        _write_retrieval_html(
            output_dir, data.get("queries", []),
            data.get("items", []), cls.label,
        )


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
