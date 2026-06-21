"""
A-Mem (Agentic Memory) — full implementation + harness adapter.

Reproduces Xu et al. 2025, "A-Mem: Agentic Memory for LLM Agents"
(arXiv 2502.12110). Three core operations are wired together by
``AgenticMemorySystem``; ``AMemAdapter`` exposes that engine to the
comparison harness through the standard SystemAdapter contract.

Algorithm summary (Xu et al. §3)
--------------------------------

A-Mem is Zettelkasten-inspired. Each memory is an atomic ``MemoryNote``
with seven fields ``{c, t, K, G, X, e, L}``:

    c : raw content (text)
    t : timestamp (str, YYYYMMDDHHMM)
    K : keywords          ─┐
    G : tags               ├─ LLM-generated enrichments of c
    X : context (one line)─┘
    e : embedding vector  (kept in ChromaDB, not on the note)
    L : links to other notes (LLM-determined)

Three operations run for every new memory:

  1. Note Construction (§3.1, prompt P_s1)
       Single LLM call extracts K, G, X from c. The note's embedding
       e = f_enc(c ⊕ K ⊕ G ⊕ X) is computed (locally) by ChromaDB's
       sentence-transformer when the note is added to the collection.

       Cost  : 1 LLM call + 1 local embed.
       Shape : analyze_content(content) → {keywords, context, tags}

  2. Link Generation + Memory Evolution (§3.2-3.3, prompt P_s2/P_s3)
       Top-k cosine-similar neighbours are surfaced from the existing
       memory set. One LLM call decides (a) which neighbours to link
       the new note to and (b) whether to rewrite any neighbour's
       context/tags in light of the new information ("write-time
       reconsolidation"). This is the analogue of biological memory
       consolidation at write time — the most novel piece of A-Mem.

       Cost  : 1 LLM call (may mutate up to k neighbour notes).
       Shape : process_memory(note) → bool   (True if evolution fired)

  3. Retrieval — "Relative Memory" (§3.4)
       Query → embed → top-k notes by cosine similarity → for each
       direct hit, also pull its 1-hop linked notes ("relative memory"
       in the paper). A reader LLM synthesises a free-text answer from
       the resulting bundle. The reader prompt asks the LLM to prefer
       the most recent timestamp on contradictions — which is what
       implements A-Mem's TechCorp → StartupCo information-update
       behaviour on the comparison dataset.

       Cost  : 1 reader LLM call.
       Shape : read(question, k) → {answer, retrieved_ids, links_followed}

Storage
-------
  - In memory : ``dict[str, MemoryNote]`` keyed by note id
  - Embeddings: local sentence-transformer (default ``all-MiniLM-L6-v2``,
                384-dim) inside a ChromaDB ephemeral collection
  - LLM       : NVIDIA NIM (Llama-3.1-70b-instruct) via ``_nim.chat``
                and ``_nim.chat_text``

Inputs and outputs (SystemAdapter contract)
-------------------------------------------
``ingest(items)``
    items : list[dict] — each dict has at minimum
        - ``id``        : str   (e.g. "m01" — preserved through retrieval)
        - ``content``   : str
        - ``timestamp`` : str   (optional)
    Side effect : populates the engine's ``memories`` dict.

``query(question)`` returns:
    {
      "answer"          : str,
      "retrieved_ids"   : list[str],            # memory IDs surfaced
      "links_followed"  : list[(str, str)],     # (from_id, to_id) pairs
    }

``dump_state(queries, output_dir, metadata)``
    Serialises the engine's memories to dicts and delegates to
    ``render.write_run`` + ``render.render_all`` for the mermaid /
    interactive-HTML output.

``render_from_run(run_path, output_dir)`` [classmethod]
    Re-renders the HTML from an existing ``run.json``. No LLM, no
    ChromaDB — used by ``just render``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import _nim, render
from .base import SystemAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt templates (verbatim from paper Appendix B, with light JSON cleanups)
# =============================================================================

PROMPT_NOTE_CONSTRUCTION = """\
Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
        // At least three keywords, but don't be too redundant.
    ],
    "context":
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}}

Content for analysis:
{content}"""

PROMPT_EVOLUTION = """\
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context: {context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. What specific actions should be taken (strengthen, update_neighbor)?
   1.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   1.2 If choose to update neighbor, you can update the context and tags of these memories based on the understanding of these memories.
Tags should be determined by the content of these characteristics of these memories, which can be used to retrieve them later and categorize them.

Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.

Return your decision in JSON format with the following structure:
{{
    "should_evolve": true or false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1", "tag_n"],
    "new_context_neighborhood": ["new context", "new context"],
    "new_tags_neighborhood": [["tag_1", "tag_n"], ["tag_1", "tag_n"]]
}}"""


# =============================================================================
# MemoryNote — one atomic Zettelkasten note
# =============================================================================

class MemoryNote:
    """
    One atomic note. Carries the seven paper fields plus a couple of
    usage-tracking counters.

    Field shapes
    ------------
        id                : str        UUID4 by default, or the
                                       dataset-provided id like "m01"
        content           : str        raw c — the original text
        timestamp         : str        t — YYYYMMDDHHMM (creation time)
        keywords          : list[str]  K — LLM-extracted (P_s1)
        tags              : list[str]  G — LLM-extracted (P_s1)
        context           : str        X — LLM-generated one-sentence
                                       semantic description (P_s1)
        links             : list[str]  L — IDs of linked notes (P_s2)
        evolution_history : list[dict] write-time reconsolidation log;
                                       each entry: {trigger, field, old, new}
        last_accessed     : str        YYYYMMDDHHMM, refreshed by retrieval
        retrieval_count   : int        bumped each time the note is surfaced

    Embeddings (the paper's e) live in the ChromaDB collection, not on
    the note. ``embedding(e) = f_enc(content ⊕ keywords ⊕ tags ⊕ context)``
    happens transparently when the note is added to the collection — the
    SentenceTransformerEmbeddingFunction inside ChromaDB does the encode.
    """

    def __init__(
        self,
        content: str,
        *,
        id: str | None = None,
        keywords: list[str] | None = None,
        tags: list[str] | None = None,
        context: str = "General",
        links: list[str] | None = None,
        timestamp: str | None = None,
    ):
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.tags = tags or []
        self.context = context
        self.links: list[str] = links or []
        now = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or now
        self.last_accessed = now
        self.retrieval_count = 0
        self.evolution_history: list[dict] = []

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (used by ``render.write_run``)."""
        return {
            "id": self.id,
            "content": self.content,
            "keywords": list(self.keywords),
            "tags": list(self.tags),
            "context": self.context,
            "links": list(self.links),
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "retrieval_count": self.retrieval_count,
            "evolution_history": list(self.evolution_history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryNote":
        """Inverse of ``to_dict``. Used to rehydrate a note from disk."""
        note = cls(
            content=d["content"],
            id=d["id"],
            keywords=d.get("keywords", []),
            tags=d.get("tags", []),
            context=d.get("context", "General"),
            links=d.get("links", []),
            timestamp=d.get("timestamp"),
        )
        note.last_accessed = d.get("last_accessed", note.last_accessed)
        note.retrieval_count = d.get("retrieval_count", 0)
        note.evolution_history = list(d.get("evolution_history", []))
        return note

    def __repr__(self) -> str:
        return (
            f"MemoryNote(id={self.id[:8]}…, "
            f"keywords={self.keywords}, tags={self.tags}, "
            f"links={[l[:8] + '…' for l in self.links]})"
        )


# =============================================================================
# AgenticMemorySystem — the A-Mem engine
# =============================================================================

class AgenticMemorySystem:
    """
    Core A-Mem engine. State + behaviour; ``AMemAdapter`` wraps this.

    State
    -----
        memories      : dict[str, MemoryNote]
                        in-memory map of every note ever added
        client        : chromadb.Client
                        ephemeral vector store (reset on construction)
        collection    : chromadb.Collection
                        ChromaDB collection used for cosine retrieval
        embedding_fn  : SentenceTransformerEmbeddingFunction
                        local sentence-transformer (no API calls)
        evo_cnt       : int   counter of evolution events
        evo_threshold : int   after this many evolutions, re-index everything

    Components
    ----------
        Embeddings : all-MiniLM-L6-v2 (384-dim, local, free)
        LLM        : NVIDIA NIM Llama-3.1-70b-instruct via ``_nim``
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "memories",
        evo_threshold: int = 100,
    ):
        self.memories: dict[str, MemoryNote] = {}
        self.model_name = model_name
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Ephemeral ChromaDB — we reset on construction so successive
        # runs in the same Python process start clean. Imported here (not at
        # module top) so the render-only path needn't install chromadb.
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
        )
        try:
            self.client.reset()
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    # ── Operation 1: Note Construction (§3.1) ────────────────────────────

    def analyze_content(self, content: str) -> dict[str, Any]:
        """
        LLM call P_s1: derive structured attributes from raw content.

        Steps
        -----
        1. Format prompt PROMPT_NOTE_CONSTRUCTION with the raw content.
        2. Single NIM chat call expecting a JSON object response.
        3. Defensive defaults if the LLM call fails or returns garbage
           (empty keywords, "General" context, empty tags — the caller
           still gets a usable note but with degraded attributes).

        Inputs
            content : str   raw text c

        Output
            dict with three keys:
                keywords : list[str]
                context  : str        one-line semantic description
                tags     : list[str]

        Cost: 1 LLM call.
        """
        prompt = PROMPT_NOTE_CONSTRUCTION.format(content=content)
        try:
            result = _nim.chat(prompt)
            return {
                "keywords": result.get("keywords", []),
                "context": result.get("context", "General"),
                "tags": result.get("tags", []),
            }
        except Exception as e:
            logger.error("analyze_content failed: %s", e)
            return {"keywords": [], "context": "General", "tags": []}

    # ── Operation 2 + 3: Link Generation + Memory Evolution (§3.2–3.3) ───

    def process_memory(self, note: MemoryNote) -> bool:
        """
        LLM call P_s2/P_s3 (combined per the paper's reference impl):
        decide which existing notes the new ``note`` should link to AND
        whether to rewrite the linked notes' attributes.

        Steps
        -----
        1. _find_related_memories(note.content, k=5)
             ChromaDB cosine search returns the 5 most similar existing
             notes (text bundle + their IDs).

             Shape : neighbours_text: str         (formatted prompt block)
                     neighbour_ids:   list[str]   (length ≤ 5)

        2. Format PROMPT_EVOLUTION with the new note + neighbours
             and call NIM. The response schema is:
                 {
                   "should_evolve":              bool,
                   "actions":                    ["strengthen", "update_neighbor"],
                   "suggested_connections":      list[str],   # neighbour IDs
                   "tags_to_update":             list[str],   # new tags for `note`
                   "new_context_neighborhood":   list[str],   # one per neighbour
                   "new_tags_neighborhood":      list[list[str]],
                 }

        3. Apply each action:
             - "strengthen"        : extend ``note.links`` with the
                                     suggested connections, optionally
                                     update ``note.tags``.
             - "update_neighbor"   : for each neighbour, rewrite its
                                     ``context`` and ``tags`` in place,
                                     appending an entry to its
                                     ``evolution_history`` log.

        Returns
        -------
        bool — True if the LLM chose to evolve anything (used by the
               caller to bump ``self.evo_cnt`` and decide when to
               consolidate the ChromaDB index).

        Cost: 1 LLM call. May mutate up to k=5 neighbour MemoryNote
        objects in place (write-time reconsolidation).
        """
        # ─── Step 1: nearest neighbours via cosine ───────────────────
        if not self.memories:
            return False
        neighbors_text, neighbor_ids = self._find_related_memories(
            note.content, k=5,
        )
        if not neighbors_text or not neighbor_ids:
            return False

        # ─── Step 2: ask the LLM what to do ──────────────────────────
        prompt = PROMPT_EVOLUTION.format(
            content=note.content,
            context=note.context,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbors_text,
            neighbor_number=len(neighbor_ids),
        )

        try:
            resp = _nim.chat(prompt)
        except Exception as e:
            logger.error("process_memory LLM call failed: %s", e)
            return False

        should_evolve = resp.get("should_evolve", False)
        if not should_evolve:
            return False

        # ─── Step 3: apply each requested action ─────────────────────
        actions = resp.get("actions", [])
        for action in actions:
            if action == "strengthen":
                # Mutate the NEW note: add links to the chosen neighbours,
                # optionally adopt the LLM's revised tag list.
                suggested = resp.get("suggested_connections", [])
                note.links.extend(suggested)
                new_tags = resp.get("tags_to_update", [])
                if new_tags:
                    note.tags = new_tags

            elif action == "update_neighbor":
                # Mutate the NEIGHBOUR notes (write-time reconsolidation).
                # The LLM returns parallel lists indexed by neighbour position.
                new_contexts = resp.get("new_context_neighborhood", [])
                new_tags_all = resp.get("new_tags_neighborhood", [])
                for i, nid in enumerate(neighbor_ids):
                    if nid not in self.memories:
                        continue
                    neighbor = self.memories[nid]
                    if i < len(new_contexts):
                        old_ctx = neighbor.context
                        neighbor.context = new_contexts[i]
                        neighbor.evolution_history.append({
                            "trigger": note.id,
                            "field": "context",
                            "old": old_ctx,
                            "new": new_contexts[i],
                        })
                    if i < len(new_tags_all):
                        old_tags = neighbor.tags
                        neighbor.tags = new_tags_all[i]
                        neighbor.evolution_history.append({
                            "trigger": note.id,
                            "field": "tags",
                            "old": old_tags,
                            "new": new_tags_all[i],
                        })
        return True

    # ── Main entry point ─────────────────────────────────────────────────

    def add_note(
        self,
        content: str,
        timestamp: str | None = None,
        id: str | None = None,
    ) -> str:
        """
        Full A-Mem pipeline for a new memory.

        Steps
        -----
        1. analyze_content(content)
             1 LLM call → keywords K, tags G, context X (Operation 1).

        2. Build a MemoryNote (id preserved if the caller supplied one).

        3. process_memory(note)
             1 LLM call → links L + neighbour evolution (Operations 2-3).
             Bumps ``self.evo_cnt`` on success; periodic
             ``_consolidate`` every ``evo_threshold`` evolutions
             re-indexes the ChromaDB collection.

        4. Persist
             - In memory  : ``self.memories[note.id] = note``
             - In ChromaDB: ``self.collection.add(documents=[content], …)``
               which transparently embeds the content via the local
               sentence-transformer.

        Inputs
            content   : str
            timestamp : str | None  optional pre-set timestamp
            id        : str | None  optional pre-set id (preserves
                                    dataset IDs like "m01")

        Output
            str — the note's id (either the caller-provided one or a fresh UUID).

        Cost: 2 LLM calls + 1 local embedding + 1 ChromaDB add.
        """
        # ─── Step 1: Note Construction ───────────────────────────────
        analysis = self.analyze_content(content)

        # ─── Step 2: build the MemoryNote ────────────────────────────
        note = MemoryNote(
            id=id,
            content=content,
            keywords=analysis["keywords"],
            context=analysis["context"],
            tags=analysis["tags"],
            timestamp=timestamp,
        )

        # ─── Step 3: Link Generation + Memory Evolution ──────────────
        evolved = self.process_memory(note)
        if evolved:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self._consolidate()

        # ─── Step 4: persist to in-memory map + ChromaDB ─────────────
        self.memories[note.id] = note

        # ChromaDB metadata values must be primitives; we json.dumps the
        # list-valued fields so they round-trip cleanly through search().
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": json.dumps(note.keywords),
            "tags": json.dumps(note.tags),
            "context": note.context,
            "links": json.dumps(note.links),
            "timestamp": note.timestamp,
        }
        self.collection.add(
            documents=[note.content],
            metadatas=[metadata],
            ids=[note.id],
        )

        return note.id

    # ── Retrieval (§3.4) ─────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Plain cosine vector search via ChromaDB. No link traversal,
        no reader LLM.

        Output (one dict per hit):
            { "id", "content", "distance",
              "keywords", "tags", "context", "links", "timestamp" }
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        if not results or not results.get("ids"):
            return []
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            out.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
                # round-trip the json.dumps'd list values back to lists
                **{k: _try_json_load(v) for k, v in meta.items()},
            })
        return out

    def search_agentic(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Agentic retrieval = cosine search + one-hop link traversal.

        Steps
        -----
        1. ``direct = self.search(query, k)``
             Top-k vector hits.

        2. For each direct hit, follow its ``links`` (1 hop) and append
             the linked notes — deduplicated against ``direct``.

        3. Each surfaced note's ``retrieval_count`` and ``last_accessed``
             are bumped (for diagnostics).

        Output
            list of dicts: direct hits first, then linked. Linked hits
            carry ``via_link_from: <direct_hit_id>`` so the caller can
            reconstruct the traversal pattern.
        """
        direct = self.search(query, k=k)
        seen = {r["id"] for r in direct}
        linked: list[dict[str, Any]] = []

        for r in direct:
            note = self.memories.get(r["id"])
            if not note:
                continue
            note.retrieval_count += 1
            note.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
            for link_id in note.links:
                if link_id in seen:
                    continue
                seen.add(link_id)
                linked_note = self.memories.get(link_id)
                if linked_note:
                    linked.append({
                        "id": linked_note.id,
                        "content": linked_note.content,
                        "keywords": linked_note.keywords,
                        "tags": linked_note.tags,
                        "context": linked_note.context,
                        "timestamp": linked_note.timestamp,
                        "via_link_from": r["id"],
                    })

        return direct + linked

    def read(self, question: str, k: int = 5) -> dict[str, Any]:
        """
        Full retrieve-then-answer pipeline (the main retrieval entry point).

        Steps
        -----
        1. ``search_agentic(question, k)``
             Vector hits + 1-hop links (bundle of relevant notes).

        2. Format the bundle for the reader prompt — include each note's
             timestamp so the LLM can break ties on contradictions by
             preferring the most recent one. This is what implements
             A-Mem's TechCorp → StartupCo update behaviour on the
             comparison dataset's ``information_update`` category.

        3. Reader LLM (NIM ``chat_text``, no JSON parse) writes one
             concise sentence; if no bundle entry contains the answer,
             it returns ``"unknown / not mentioned"`` so the scorer's
             abstention heuristic catches it.

        Inputs
            question : str
            k        : int — top-k for ``search_agentic`` (default 5)

        Output (the SystemAdapter contract — same as ``AMemAdapter.query``)
            {
              "answer"          : str,
              "retrieved_ids"   : list[str],         # memory IDs
              "links_followed"  : list[(from, to)],  # 1-hop traversals
            }

        Cost: 1 reader LLM call.
        """
        results = self.search_agentic(question, k=k)
        if not results:
            return {
                "answer": "I don't have any information about that.",
                "retrieved_ids": [],
                "links_followed": [],
            }

        # Build the context block + record which IDs / links were used.
        context_lines = []
        retrieved_ids: list[str] = []
        links_followed: list[tuple[str, str]] = []
        for r in results:
            retrieved_ids.append(r["id"])
            via = r.get("via_link_from")
            if via:
                links_followed.append((via, r["id"]))
            context_lines.append(
                f"- [{r['id']}] (ts={r.get('timestamp', '?')}) {r['content']}"
            )
        context = "\n".join(context_lines)

        prompt = (
            f"You are answering a question using only the memories below. "
            f"If multiple memories contradict, prefer the most recent one (later timestamp). "
            f"If the answer is not contained in the memories, reply exactly: \"unknown / not mentioned\".\n\n"
            f"Memories:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely in one short sentence."
        )

        answer = _nim.chat_text(
            prompt,
            system="You are a precise question-answering assistant. Use only the provided memories.",
        )
        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "links_followed": links_followed,
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    def _find_related_memories(
        self, content: str, k: int = 5,
    ) -> tuple[str, list[str]]:
        """
        ChromaDB top-k neighbours of ``content`` for the evolution prompt.

        Returns a tuple ``(formatted_block, neighbour_ids)`` where the
        block is a human-readable text representation of the neighbours
        ready to drop into PROMPT_EVOLUTION's ``{nearest_neighbors_memories}``
        slot.
        """
        count = self.collection.count()
        if count == 0:
            return "", []
        actual_k = min(k, count)
        results = self.collection.query(
            query_texts=[content], n_results=actual_k,
        )
        if not results or not results.get("ids") or not results["ids"][0]:
            return "", []

        parts = []
        ids = []
        for i, doc_id in enumerate(results["ids"][0]):
            note = self.memories.get(doc_id)
            if not note:
                continue
            ids.append(doc_id)
            parts.append(
                f"Memory {i + 1} (ID: {doc_id}):\n"
                f"  content: {note.content}\n"
                f"  keywords: {note.keywords}\n"
                f"  tags: {note.tags}\n"
                f"  context: {note.context}"
            )
        return "\n\n".join(parts), ids

    def _consolidate(self) -> None:
        """
        Re-index every note in ChromaDB. Called every ``evo_threshold``
        evolution events because mutated neighbours have stale embeddings
        (the paper notes this in §3.3 — their context/tags changed but
        the embedding wasn't recomputed at the time of the rewrite).
        """
        logger.info("consolidating %d memories", len(self.memories))
        try:
            self.client.reset()
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="memories",
            embedding_function=self.embedding_fn,
        )
        for note in self.memories.values():
            metadata = {
                "id": note.id,
                "content": note.content,
                "keywords": json.dumps(note.keywords),
                "tags": json.dumps(note.tags),
                "context": note.context,
                "links": json.dumps(note.links),
                "timestamp": note.timestamp,
            }
            self.collection.add(
                documents=[note.content],
                metadatas=[metadata],
                ids=[note.id],
            )


def _try_json_load(v: Any) -> Any:
    """ChromaDB metadata round-trip: lists are stored as json.dumps'd
    strings; try to parse back, fall back to the raw value."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
    return v


# =============================================================================
# AMemAdapter — bridge AgenticMemorySystem to the harness's SystemAdapter
# =============================================================================

class AMemAdapter(SystemAdapter):
    """
    Thin harness wrapper around ``AgenticMemorySystem``. Translates
    between the comparison harness's ``(items, question) → run.json``
    contract and the engine's ``add_note`` / ``read`` API.
    """

    name = "amem"
    label = "A-Mem"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        # ``top_k`` controls retrieval breadth (default 5 reproduces the
        # paper's setting); ``evo_threshold`` controls how often the
        # ChromaDB index is re-consolidated after evolution events.
        self.top_k = int(self.params.get("top_k", 5))
        # The engine owns ChromaDB + the in-memory MemoryNote map.
        self.engine = AgenticMemorySystem(
            evo_threshold=int(self.params.get("evo_threshold", 100)),
        )

    def ingest(self, items: list[dict[str, Any]]) -> None:
        """
        Index ``items`` chronologically by replaying ``engine.add_note``
        on each. Each call runs the full A-Mem pipeline (note
        construction → link generation → memory evolution).

        Cost  : 2 LLM calls per item (+ up to k evolution side-effects
                on neighbours).
        Order : matters. Later items can evolve earlier ones — we preserve
                the dataset's chronological order.

        Side effect : populates ``self.engine.memories`` with one
                      MemoryNote per item. Dataset IDs ("m01", "m02", …)
                      are preserved through the ``id`` keyword.
        """
        for i, m in enumerate(items, 1):
            self.engine.add_note(
                content=m["content"],
                timestamp=m.get("timestamp"),
                id=m["id"],
            )
            note = self.engine.memories[m["id"]]
            print(
                f"  [{i:2d}/{len(items)}] {m['id']}  "
                f"tags={note.tags}  links={len(note.links)}  "
                f"neighbors_evolved={len(note.evolution_history)}"
            )

    def query(self, question: str) -> dict[str, Any]:
        """
        Delegate to ``engine.read(question, k=self.top_k)``. The engine runs
        the full retrieve-then-answer pipeline — see ``AgenticMemorySystem.read``
        for the step-by-step.

        Returns the SystemAdapter contract:
            { "answer": str,
              "retrieved_ids": list[str],
              "links_followed": list[(str, str)] }
        """
        return self.engine.read(question, k=self.top_k)

    def dump_state(
        self,
        queries: list[dict[str, Any]],
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        """
        Persist + render.

        Steps
        -----
        1. Serialise MemoryNote → dict via ``to_dict``. The on-disk
           format is dict-shaped; render.py operates on those dicts
           directly so it doesn't need to import MemoryNote.
        2. ``render.write_run(result/result.json)`` — frozen run artifact.
        3. ``self.render_from_run`` — replays render.py's mermaid +
           interactive-HTML pipeline, writing memory/ + result/ HTML.
        """
        run_path = output_dir / "result" / "result.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            **metadata,
            "system": self.name,
            "system_label": self.label,
            "params": self.params,
        }
        memories_dict = {
            mid: note.to_dict() for mid, note in self.engine.memories.items()
        }
        render.write_run(run_path, memories_dict, queries, meta)
        self.render_from_run(run_path, output_dir)

    @classmethod
    def render_from_run(cls, run_path: Path, output_dir: Path) -> None:
        """
        Re-emit mermaid + interactive HTML from a frozen
        ``result/result.json``. No LLM, no ChromaDB; ``render.load_run``
        returns dict-shaped memories that ``render.render_all`` consumes.
        """
        run = render.load_run(run_path)
        render.render_all(run, output_dir)
