"""
HippoRAG v2 adapter for the comparison harness.

Ports https://github.com/pandazxx/hipporag-reproduction/blob/main/experiments/demo_v2.py.

What v2 changes vs v1
---------------------
1. Passage nodes in the graph
     v1's KG had only phrase nodes. v2 adds one node per passage and
     connects every phrase node to the passages where it appears
     (``context edges``). PPR can now flow directly into passages.

2. Query → triple linking (instead of query NER)
     v1 ran NER on the question and matched entity strings.
     v2 embeds the whole question and retrieves the top-K most similar
     ``"subject predicate object"`` triple texts.

3. Recognition memory (online LLM filter)
     An LLM call keeps only the triples actually relevant to the query
     (up to 4). The phrases inside those kept triples become high-weight
     PPR seeds. All passages also seed at a low weight scaled by
     query↔passage embedding similarity.

4. Ranking
     PPR scores are read directly off the passage portion of the
     stationary distribution — no specificity weighting needed because
     passages are first-class graph citizens.

Algorithm diagram
-----------------
Indexing (one-time):
    passages → OpenIE → triples (s, p, o, pidx)
    embed phrases / passages / "s p o" triple texts
    build adj on N = N_phrase + P nodes with:
        relation edges  (phrase ↔ phrase, from triples)
        synonymy edges  (phrase ↔ phrase, cosine ≥ threshold)
        context edges   (phrase ↔ passage, from triple→source mapping)

Retrieval (per question):
    query → query embedding
          → top-K triples by cosine
          → LLM filter keeps the relevant ones
          → seed vector:
                phrase nodes inside kept triples: weight 1.0 each
                every passage node: passage_seed_weight × query_sim
          → PPR
          → read PPR[N_phrase:] → passage scores
          → top-K passages → reader LLM → answer

Inputs and outputs (SystemAdapter contract)
-------------------------------------------
``ingest(items)``
    items : list[dict] with keys ``id``, ``content``, ``timestamp?``.
    Side effect: populates ``self.index`` with phrase + passage + triple
    embeddings and the combined sparse adjacency.

``query(question)`` returns:
    {
      "answer"           : str,
      "retrieved_ids"    : list[str],            # top-5 passage IDs
      "retrieval_scores" : list[float],
      "top_triples"      : list[[s, p, o]],      # top-K by query→triple sim
      "top_triple_sims"  : list[float],
      "filtered_triples" : list[[s, p, o]],      # what recognition memory kept
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
from .hipporag import _read, _write_retrieval_html


# ── PPR with arbitrary seed weights ────────────────────────────────────────

def _personalized_pagerank(
    adj: sp.csr_matrix,
    seeds: np.ndarray,
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    PPR with arbitrary (non-uniform) seed weights.

    Differs from v1's PPR in one place: ``seeds`` is a full (N,) weight
    vector instead of a list of node indices. v2 needs this because
    different nodes get different initial probabilities (kept-triple
    phrases get weight 1.0; every passage gets a fractional weight
    proportional to its embedding similarity to the query).

    Inputs
        adj   : sparse (N, N) symmetric weighted adjacency
        seeds : ndarray (N,) — arbitrary non-negative weights;
                                renormalised to sum to 1 internally.
    Output
        ndarray (N,) float64 — stationary distribution
    """
    N = adj.shape[0]
    # Row-stochastic transition matrix T = D⁻¹ · adj.
    row_sums = np.array(adj.sum(axis=1), dtype=np.float64).flatten()
    row_sums[row_sums == 0] = 1.0
    T = sp.diags(1.0 / row_sums) @ adj

    # Normalise seeds to a probability distribution. If all-zero
    # (shouldn't happen here but a defensive fallback), spread uniformly.
    s = seeds / seeds.sum() if seeds.sum() > 0 else np.ones(N) / N

    # Same power iteration as v1's PPR; converges at rate (1-α).
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
        """
        Build the v2 index. Mutates ``self.index``.

        Node layout
        -----------
        We pack two node types into one (N_phrase + P)-dim index space:
            indices [0 .. N_phrase-1]   → phrase nodes
            indices [N_phrase .. N-1]   → passage nodes
                                          (passage pidx → node N_phrase + pidx)
        This single space is what PPR runs on and what ``adj`` indexes.

        Steps
        -----
        1. OpenIE
             Two NIM calls (NER → OpenIE) per passage. Same as v1.

             Shape : triples_per_passage[pidx] = list[(s, p, o)]

        2. Phrase + flat triple list
             Collect unique phrases across all triples; build a flat list
             of ``(s, p, o, pidx)`` so each triple remembers its source
             passage (needed for context edges).

             Shape : phrases:    list[str], length N_phrase
                     phrase_idx: dict[str → int]
                     triples:    list[(s, p, o, pidx)]

        3. Embeddings (three batched NIM calls)
             - phrase_embs  : (N_phrase, D) — one node per phrase
             - passage_embs : (P, D)        — one node per passage
             - triple_embs  : (T, D)        — over "s p o" concatenations,
                                              used at query time for the
                                              query→triple match

        4a. Relation edges (phrase ↔ phrase)
             For each triple add a symmetric weight-1 edge between subject
             and object phrase nodes. Self-loops are skipped. Multiple
             relations between the same phrases collapse via ``max``.

        4b. Synonymy edges (phrase ↔ phrase)
             Phrase-embedding cosine ≥ ``sim_threshold`` (default 0.75 —
             lower than v1's 0.80 because the recognition memory step
             filters noise downstream). Only added when the pair isn't
             already a relation edge.

        4c. Context edges (phrase ↔ passage)
             For every triple, connect both phrase nodes to that triple's
             source passage node. This is the new v2 wiring that lets
             PPR flow phrase → passage in one step.

        5. Sparse CSR adjacency
             Build the (N, N) sparse symmetric matrix from the merged
             edge dict.

             Shape : adj: csr_matrix (N, N) float64
        """
        self.item_ids = [m["id"] for m in items]
        self.passages = [m["content"] for m in items]
        P = len(self.passages)

        # ─── Step 1: OpenIE ──────────────────────────────────────────
        print(f"  HippoRAG2 ingest: {P} passages")
        print(f"  Phase 1 — OpenIE")
        triples_per_passage: dict[int, list[tuple]] = {}
        for i, passage in enumerate(self.passages):
            triples_per_passage[i] = _nim.extract_triples(passage)
            print(f"    [{i+1:2d}/{P}] {self.item_ids[i]}: "
                  f"{len(triples_per_passage[i])} triple(s)")

        # ─── Step 2: phrases + flat triple list ──────────────────────
        # Each triple remembers its source passage (pidx) so step 4c
        # can wire up context edges from phrase nodes to passage nodes.
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

        # ─── Step 3: three batched NIM embedding calls ───────────────
        phrase_embs = _nim.embed_batch(phrases) if phrases else np.zeros((0, 1024))
        passage_embs = _nim.embed_batch(self.passages) if self.passages else np.zeros((0, 1024))
        triple_texts = [f"{s} {p} {o}" for s, p, o, _ in triples]
        triple_embs = (
            _nim.embed_batch(triple_texts) if triples
            else np.zeros((0, phrase_embs.shape[1] if N_phrase else 1024))
        )

        # ─── Step 4: edges ───────────────────────────────────────────
        graph: dict[tuple[int, int], float] = defaultdict(float)

        # 4a. relation edges (phrase ↔ phrase, from triples)
        relation_pairs: set[tuple[int, int]] = set()
        for s, _p, o, _pidx in triples:
            si, oi = phrase_idx[s], phrase_idx[o]
            if si == oi:               # skip self-loops (e.g. "X is X")
                continue
            graph[(si, oi)] = max(graph[(si, oi)], 1.0)
            graph[(oi, si)] = max(graph[(oi, si)], 1.0)
            relation_pairs.add(tuple(sorted([si, oi])))

        # 4b. synonymy edges (phrase ↔ phrase, cosine ≥ threshold)
        # Only add where no relation edge already exists; otherwise we'd
        # be overwriting a stronger semantic edge with a weaker one.
        syn_count = 0
        if N_phrase > 1:
            sims = phrase_embs @ phrase_embs.T              # (N_phrase, N_phrase)
            for i in range(N_phrase):
                for j in range(i + 1, N_phrase):
                    if (sims[i, j] >= self.sim_threshold
                            and tuple(sorted([i, j])) not in relation_pairs):
                        graph[(i, j)] = max(graph[(i, j)], float(sims[i, j]))
                        graph[(j, i)] = max(graph[(j, i)], float(sims[i, j]))
                        syn_count += 1

        # 4c. context edges (phrase ↔ passage)
        # The v2-specific wiring: every phrase in a triple gets a
        # weight-1 undirected edge to that triple's source passage node.
        context_count = 0
        seen_ctx: set[tuple[int, int]] = set()
        for s, _p, o, pidx in triples:
            pni = N_phrase + pidx                            # passage node index
            for ei in (phrase_idx[s], phrase_idx[o]):
                if (ei, pni) in seen_ctx:
                    continue
                graph[(ei, pni)] = 1.0
                graph[(pni, ei)] = 1.0
                seen_ctx.add((ei, pni))
                context_count += 1

        print(f"    relation_edges={len(relation_pairs)}  synonymy_edges={syn_count}  "
              f"context_edges={context_count}")

        # ─── Step 5: build the sparse CSR adjacency ─────────────────
        rows, cols, data = [], [], []
        for (i, j), w in graph.items():
            rows.append(i); cols.append(j); data.append(w)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(max(N, 1), max(N, 1)),
                            dtype=np.float64)

        self.index = dict(
            phrases=phrases,                # list[str] length N_phrase
            phrase_idx=phrase_idx,          # dict[str → int]
            N_phrase=N_phrase, P=P, N=N,    # node-space sizes
            phrase_embs=phrase_embs,        # (N_phrase, D) float64
            passage_embs=passage_embs,      # (P, D) float64
            triples=triples,                # list[(s, p, o, pidx)]
            triple_embs=triple_embs,        # (T, D) float64
            adj=adj,                        # csr (N, N) float64
            triples_per_passage=triples_per_passage,
        )

    def query(self, question: str) -> dict[str, Any]:
        """
        v2 retrieval — query → triples → LLM filter → PPR → passages.

        Steps
        -----
        1. Embed the question
             One NIM embedding call with ``input_type="query"`` (the
             query-side of NIM's asymmetric embedding model).

             Shape : q_emb: (D,) float64

        2. Top-K triples by cosine similarity
             Score each indexed triple text against the query.

             Shape : triple_sims: (T,) → order: top-``top_k_triples``
                     top_triples: list[(s, p, o)]

        3. Recognition memory (LLM filter)
             One LLM call narrows the top-K triples down to up to 4
             that are actually relevant. Skips noise that surface-level
             cosine surfaced.

             Output : filtered: list[(s, p, o)]

        4. Build the seed vector (the v2-specific part)
             Two seed types blended:
               4a. Phrase seeds — every phrase in the filtered triples
                   gets weight 1.0 added to its node.
               4b. Passage seeds — every passage node gets
                   ``passage_seed_weight × cos(query, passage)`` added.
                   This is a soft fallback: even if PPR doesn't reach a
                   passage via the KG, query-similar passages still
                   surface.

             Shape : seeds: (N,) float64

        5. PPR (arbitrary seed weights)
             Same power iteration as v1; the ``seeds`` vector is
             non-uniform, which is why we use the v2 PPR variant.

             Shape : ppr: (N,) float64

        6. Read passage scores directly
             v1 had to project entity scores through ``P_matrix``.
             v2 reads ``ppr[N_phrase:]`` — passages are graph nodes.

             Shape : passage_scores: (P,) float64

        7. Top-K passages → reader LLM
             One LLM call synthesises the final answer.

        Output (SystemAdapter contract + diagnostic extras)
            {
              "answer"           : str,
              "retrieved_ids"    : list[str],        # top-5 passage IDs
              "retrieval_scores" : list[float],
              "top_triples"      : list[[s, p, o]],  # before LLM filter
              "top_triple_sims"  : list[float],
              "filtered_triples" : list[[s, p, o]],  # after LLM filter
            }
        """
        idx = self.index

        # ─── Step 1: embed the query ────────────────────────────────
        q_emb = _nim.embed(question, input_type="query")        # (D,)

        # ─── Step 2: top-K triples by cosine ────────────────────────
        top_triples: list[tuple] = []
        top_triple_sims: list[float] = []
        if idx.get("triples"):
            triple_sims = idx["triple_embs"] @ q_emb            # (T,)
            order = np.argsort(-triple_sims)[: self.top_k_triples]
            top_triples = [
                (idx["triples"][i][0], idx["triples"][i][1], idx["triples"][i][2])
                for i in order
            ]
            top_triple_sims = [float(triple_sims[i]) for i in order]

        # ─── Step 3: recognition memory LLM filter ──────────────────
        filtered = _nim.filter_triples(question, top_triples, top_k=4)

        # ─── Step 4: seed vector (phrase + passage seeds) ───────────
        seeds = np.zeros(idx["N"], dtype=np.float64) if idx["N"] else np.zeros(1)

        # 4a. Phrase seeds — kept triples' subject and object phrases
        for s, _p, o in filtered:
            if s in idx["phrase_idx"]:
                seeds[idx["phrase_idx"][s]] += 1.0
            if o in idx["phrase_idx"]:
                seeds[idx["phrase_idx"][o]] += 1.0

        # 4b. Passage seeds — small weight, scaled by query↔passage cosine.
        # Acts as a soft fallback for passages PPR wouldn't otherwise reach.
        if idx["P"] and idx["passage_embs"].size:
            passage_sims = idx["passage_embs"] @ q_emb          # (P,)
            for pidx in range(idx["P"]):
                seeds[idx["N_phrase"] + pidx] += (
                    self.passage_seed_weight * max(0.0, float(passage_sims[pidx]))
                )

        # Defensive fallback: if the LLM filter dropped everything AND
        # there are passages, seed every passage uniformly so PPR has
        # something to propagate.
        if seeds.sum() == 0 and idx["P"]:
            for pidx in range(idx["P"]):
                seeds[idx["N_phrase"] + pidx] = 1.0 / idx["P"]

        # ─── Steps 5-6: PPR → read passage portion ──────────────────
        if idx.get("adj") is not None and idx["adj"].shape[0] > 0:
            ppr = _personalized_pagerank(idx["adj"], seeds)     # (N,)
            passage_scores = ppr[idx["N_phrase"]:]              # (P,)
            order = np.argsort(-passage_scores)[:5]
            top = [(int(i), float(passage_scores[i])) for i in order if passage_scores[i] > 0]
        else:
            top = []

        # ─── Step 7: reader LLM ─────────────────────────────────────
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

    @classmethod
    def render_from_run(cls, run_path: Path, output_dir: Path) -> None:
        data = json.loads(run_path.read_text())
        _write_retrieval_html(
            output_dir, data.get("queries", []),
            data.get("items", []), cls.label,
        )
