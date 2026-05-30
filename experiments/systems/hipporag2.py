"""
HippoRAG v2 adapter for the comparison harness.

Ports https://github.com/pandazxx/hipporag-reproduction/blob/main/experiments/demo_v2.py.

v2 vs v1:
  - Passage nodes are first-class graph citizens (alongside phrase nodes).
  - Seeds come from query→triple matching + LLM filter, not query NER.
  - PPR scores are read directly off passage nodes (no specificity weighting).
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
from .hipporag import _read, _write_retrieval_html


# ── PPR with arbitrary seed weights ────────────────────────────────────────

def _personalized_pagerank(
    adj: sp.csr_matrix,
    seeds: np.ndarray,
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    N = adj.shape[0]
    row_sums = np.array(adj.sum(axis=1), dtype=np.float64).flatten()
    row_sums[row_sums == 0] = 1.0
    T = sp.diags(1.0 / row_sums) @ adj

    s = seeds / seeds.sum() if seeds.sum() > 0 else np.ones(N) / N
    r = s.copy()
    for _ in range(max_iter):
        r_new = (1.0 - alpha) * T.T.dot(r) + alpha * s
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    return r


# ── Adapter ────────────────────────────────────────────────────────────────

class HippoRAG2Adapter(SystemAdapter):
    name = "hipporag2"
    label = "HippoRAG v2"

    def __init__(
        self,
        sim_threshold: float = 0.75,
        top_k_triples: int = 10,
        passage_seed_weight: float = 0.05,
    ) -> None:
        self.sim_threshold = sim_threshold
        self.top_k_triples = top_k_triples
        self.passage_seed_weight = passage_seed_weight
        self.item_ids: list[str] = []
        self.passages: list[str] = []
        self.index: dict[str, Any] = {}

    def ingest(self, items: list[dict[str, Any]]) -> None:
        self.item_ids = [m["id"] for m in items]
        self.passages = [m["content"] for m in items]
        P = len(self.passages)

        print(f"  HippoRAG2 ingest: {P} passages")
        print(f"  Phase 1 — OpenIE")
        triples_per_passage: dict[int, list[tuple]] = {}
        for i, passage in enumerate(self.passages):
            triples_per_passage[i] = _nim.extract_triples(passage)
            print(f"    [{i+1:2d}/{P}] {self.item_ids[i]}: "
                  f"{len(triples_per_passage[i])} triple(s)")

        # collect phrases + flat triple list
        phrase_set: set[str] = set()
        triples: list[tuple] = []  # (subj, pred, obj, pidx)
        for pidx, tps in triples_per_passage.items():
            for s, p, o in tps:
                phrase_set.add(s); phrase_set.add(o)
                triples.append((s, p, o, pidx))

        phrases = sorted(phrase_set)
        phrase_idx = {ph: i for i, ph in enumerate(phrases)}
        N_phrase = len(phrases)
        N = N_phrase + P
        print(f"    phrase nodes={N_phrase}  passage nodes={P}  triples={len(triples)}")

        # embeddings (one batch call per group)
        phrase_embs = _nim.embed_batch(phrases) if phrases else np.zeros((0, 1024))
        passage_embs = _nim.embed_batch(self.passages) if self.passages else np.zeros((0, 1024))
        triple_texts = [f"{s} {p} {o}" for s, p, o, _ in triples]
        triple_embs = (
            _nim.embed_batch(triple_texts) if triples
            else np.zeros((0, phrase_embs.shape[1] if N_phrase else 1024))
        )

        # edges
        graph: dict[tuple[int, int], float] = defaultdict(float)
        relation_pairs: set[tuple[int, int]] = set()
        for s, _p, o, _pidx in triples:
            si, oi = phrase_idx[s], phrase_idx[o]
            if si == oi:
                continue
            graph[(si, oi)] = max(graph[(si, oi)], 1.0)
            graph[(oi, si)] = max(graph[(oi, si)], 1.0)
            relation_pairs.add(tuple(sorted([si, oi])))

        syn_count = 0
        if N_phrase > 1:
            sims = phrase_embs @ phrase_embs.T
            for i in range(N_phrase):
                for j in range(i + 1, N_phrase):
                    if (sims[i, j] >= self.sim_threshold
                            and tuple(sorted([i, j])) not in relation_pairs):
                        graph[(i, j)] = max(graph[(i, j)], float(sims[i, j]))
                        graph[(j, i)] = max(graph[(j, i)], float(sims[i, j]))
                        syn_count += 1

        context_count = 0
        seen_ctx: set[tuple[int, int]] = set()
        for s, _p, o, pidx in triples:
            pni = N_phrase + pidx
            for ei in (phrase_idx[s], phrase_idx[o]):
                if (ei, pni) in seen_ctx:
                    continue
                graph[(ei, pni)] = 1.0
                graph[(pni, ei)] = 1.0
                seen_ctx.add((ei, pni))
                context_count += 1

        print(f"    relation_edges={len(relation_pairs)}  synonymy_edges={syn_count}  "
              f"context_edges={context_count}")

        rows, cols, data = [], [], []
        for (i, j), w in graph.items():
            rows.append(i); cols.append(j); data.append(w)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(max(N, 1), max(N, 1)),
                            dtype=np.float64)

        self.index = dict(
            phrases=phrases, phrase_idx=phrase_idx,
            N_phrase=N_phrase, P=P, N=N,
            phrase_embs=phrase_embs, passage_embs=passage_embs,
            triples=triples, triple_embs=triple_embs,
            adj=adj, triples_per_passage=triples_per_passage,
        )

    def query(self, question: str) -> dict[str, Any]:
        idx = self.index
        q_emb = _nim.embed(question, input_type="query")

        top_triples: list[tuple] = []
        top_triple_sims: list[float] = []
        if idx.get("triples"):
            triple_sims = idx["triple_embs"] @ q_emb
            order = np.argsort(-triple_sims)[: self.top_k_triples]
            top_triples = [
                (idx["triples"][i][0], idx["triples"][i][1], idx["triples"][i][2])
                for i in order
            ]
            top_triple_sims = [float(triple_sims[i]) for i in order]

        filtered = _nim.filter_triples(question, top_triples, top_k=4)

        seeds = np.zeros(idx["N"], dtype=np.float64) if idx["N"] else np.zeros(1)
        for s, _p, o in filtered:
            if s in idx["phrase_idx"]:
                seeds[idx["phrase_idx"][s]] += 1.0
            if o in idx["phrase_idx"]:
                seeds[idx["phrase_idx"][o]] += 1.0

        if idx["P"] and idx["passage_embs"].size:
            passage_sims = idx["passage_embs"] @ q_emb
            for pidx in range(idx["P"]):
                seeds[idx["N_phrase"] + pidx] += (
                    self.passage_seed_weight * max(0.0, float(passage_sims[pidx]))
                )

        if seeds.sum() == 0 and idx["P"]:
            for pidx in range(idx["P"]):
                seeds[idx["N_phrase"] + pidx] = 1.0 / idx["P"]

        if idx.get("adj") is not None and idx["adj"].shape[0] > 0:
            ppr = _personalized_pagerank(idx["adj"], seeds)
            passage_scores = ppr[idx["N_phrase"]:]
            order = np.argsort(-passage_scores)[:5]
            top = [(int(i), float(passage_scores[i])) for i in order if passage_scores[i] > 0]
        else:
            top = []

        retrieved_ids = [self.item_ids[pidx] for pidx, _ in top]
        passages = [(self.item_ids[pidx], self.passages[pidx]) for pidx, _ in top]
        answer = _read(question, passages)

        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieval_scores": [round(s, 5) for _, s in top],
            "top_triples": [list(t) for t in top_triples],
            "top_triple_sims": [round(s, 3) for s in top_triple_sims],
            "filtered_triples": [list(t) for t in filtered],
        }

    def dump_state(
        self,
        queries: list[dict[str, Any]],
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        idx = self.index
        meta = {
            **metadata,
            "system": self.name,
            "system_label": self.label,
            "sim_threshold": self.sim_threshold,
            "top_k_triples": self.top_k_triples,
            "passage_seed_weight": self.passage_seed_weight,
            "n_phrases": len(idx.get("phrases", [])),
            "n_triples": len(idx.get("triples", [])),
        }
        items = [
            {
                "id": pid,
                "content": txt,
                "triples": list(self.index["triples_per_passage"].get(i, [])),
            }
            for i, (pid, txt) in enumerate(zip(self.item_ids, self.passages))
        ]
        payload = {"metadata": meta, "items": items, "queries": queries}
        (output_dir / "run.json").write_text(json.dumps(payload, indent=2))

        _write_retrieval_html(output_dir, queries, items, self.label)
