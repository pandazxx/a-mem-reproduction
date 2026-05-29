"""
A-Mem: Agentic Memory for LLM Agents — core implementation.

Reproduces the three core operations from §3 of Xu et al. (2025):
  1. Note Construction  — LLM extracts keywords, context, tags from raw content
  2. Link Generation + Memory Evolution — LLM decides links and evolves neighbors
  3. Retrieval — cosine similarity + one-hop link traversal

LLM calls are routed through ``_nim.py`` (NVIDIA NIM endpoint).
Embeddings use sentence-transformers locally via ChromaDB, same as the original.
"""

from __future__ import annotations

import html
import json
import logging
import textwrap
import uuid
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from . import _nim

if TYPE_CHECKING:
    from pyvis.network import Network

logger = logging.getLogger(__name__)

# ── Prompt templates (from paper Appendix B) ─────────────────────────────

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


# ── MemoryNote ────────────────────────────────────────────────────────────

class MemoryNote:
    """A single atomic note in the Zettelkasten-style memory."""

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

    def __repr__(self) -> str:
        return (
            f"MemoryNote(id={self.id[:8]}…, "
            f"keywords={self.keywords}, tags={self.tags}, "
            f"links={[l[:8] + '…' for l in self.links]})"
        )


# ── AgenticMemorySystem ───────────────────────────────────────────────────

class AgenticMemorySystem:
    """
    Core A-Mem system.

    - Embeddings: local sentence-transformers via ChromaDB (all-MiniLM-L6-v2)
    - LLM: NVIDIA NIM (Llama-3.1-70b-instruct) via ``_nim.chat()``
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
        """LLM call P_s1: extract keywords, context, tags from raw content."""
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
        LLM call P_s2/P_s3 (combined in the original implementation):
        find related memories, decide links, evolve neighbors.

        Returns True if evolution occurred.
        """
        if not self.memories:
            return False

        neighbors_text, neighbor_ids = self._find_related_memories(
            note.content, k=5,
        )
        if not neighbors_text or not neighbor_ids:
            return False

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

        actions = resp.get("actions", [])
        for action in actions:
            if action == "strengthen":
                suggested = resp.get("suggested_connections", [])
                note.links.extend(suggested)
                new_tags = resp.get("tags_to_update", [])
                if new_tags:
                    note.tags = new_tags

            elif action == "update_neighbor":
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
        Full A-Mem pipeline for a new memory:
          1. Construct note (LLM → keywords, context, tags)
          2. Link generation + memory evolution (LLM)
          3. Store in vector DB
        """
        analysis = self.analyze_content(content)

        note = MemoryNote(
            id=id,
            content=content,
            keywords=analysis["keywords"],
            context=analysis["context"],
            tags=analysis["tags"],
            timestamp=timestamp,
        )

        evolved = self.process_memory(note)
        if evolved:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self._consolidate()

        self.memories[note.id] = note

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
        Basic vector search — top-k by cosine similarity.
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
                **{k: _try_json_load(v) for k, v in meta.items()},
            })
        return out

    def read(self, question: str, k: int = 5) -> dict[str, Any]:
        """
        Full retrieve-then-answer pipeline.

        Returns:
            answer: free-text answer from the reader LLM
            retrieved_ids: IDs surfaced by ``search_agentic``
            links_followed: pairs (from_id, to_id) where retrieval crossed a link
        """
        results = self.search_agentic(question, k=k)
        if not results:
            return {"answer": "I don't have any information about that.", "retrieved_ids": [], "links_followed": []}

        context_lines = []
        retrieved_ids = []
        links_followed = []
        for r in results:
            retrieved_ids.append(r["id"])
            via = r.get("via_link_from")
            if via:
                links_followed.append((via, r["id"]))
            context_lines.append(f"- [{r['id']}] (ts={r.get('timestamp', '?')}) {r['content']}")
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

    def search_agentic(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Agentic retrieval: vector search + one-hop link traversal.
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

    # ── Mermaid renderers ────────────────────────────────────────────────

    def to_mermaid_graph(
        self,
        *,
        order: list[str] | None = None,
        max_label: int = 40,
    ) -> str:
        """
        Render the full memory graph as a mermaid ``graph LR`` block.

        Nodes are memories (labelled with id + truncated content). Edges are
        A-Mem links. Notes that were the target of a memory-evolution event
        get a highlighted style so the evolution write-back is visible.
        """
        ordered_ids = order if order is not None else sorted(self.memories.keys())
        ordered_ids = [mid for mid in ordered_ids if mid in self.memories]

        lines = ["graph LR"]
        for mid in ordered_ids:
            note = self.memories[mid]
            label = _mermaid_label(note.content, max_label)
            lines.append(f'    {_mermaid_id(mid)}["{mid}: {label}"]')

        seen: set[tuple[str, str]] = set()
        for mid in ordered_ids:
            note = self.memories[mid]
            for lid in note.links:
                if lid not in self.memories:
                    continue
                pair = tuple(sorted([mid, lid]))
                if pair in seen:
                    continue
                seen.add(pair)
                lines.append(f"    {_mermaid_id(mid)} --- {_mermaid_id(lid)}")

        evolved = [mid for mid in ordered_ids if self.memories[mid].evolution_history]
        if evolved:
            lines.append("    classDef evolved fill:#fef3c7,stroke:#d97706,stroke-width:2px")
            lines.append(
                "    class " + ",".join(_mermaid_id(mid) for mid in evolved) + " evolved"
            )
        return "\n".join(lines)

    def to_mermaid_trace(
        self,
        question: str,
        result: dict[str, Any],
        *,
        max_label: int = 40,
    ) -> str:
        """
        Render a single retrieval trace as a mermaid ``graph TD`` block.

        Solid arrows = direct top-k hits from vector search.
        Dashed arrows = one-hop link traversals.
        """
        retrieved_ids = result.get("retrieved_ids", [])
        links_followed = result.get("links_followed", [])
        via_targets = {to_id for _, to_id in links_followed}
        direct_ids = [mid for mid in retrieved_ids if mid not in via_targets]

        lines = ["graph TD"]
        q_label = _mermaid_label(question, 60)
        lines.append(f'    Q(["Q: {q_label}"])')

        nodes_rendered: set[str] = set()
        for mid in direct_ids:
            note = self.memories.get(mid)
            if not note:
                continue
            label = _mermaid_label(note.content, max_label)
            lines.append(f'    {_mermaid_id(mid)}["{mid}: {label}"]')
            lines.append(f"    Q ==> {_mermaid_id(mid)}")
            nodes_rendered.add(mid)

        for from_id, to_id in links_followed:
            target = self.memories.get(to_id)
            if not target:
                continue
            if to_id not in nodes_rendered:
                label = _mermaid_label(target.content, max_label)
                lines.append(f'    {_mermaid_id(to_id)}["{to_id}: {label}"]')
                nodes_rendered.add(to_id)
            lines.append(f"    {_mermaid_id(from_id)} -.->|link| {_mermaid_id(to_id)}")

        return "\n".join(lines)

    # ── Pyvis renderers (interactive HTML) ───────────────────────────────

    def to_pyvis_graph(
        self,
        *,
        height: str = "750px",
        width: str = "100%",
    ) -> Network:
        """
        Render the memory graph as an interactive pyvis ``Network``.

        Drag nodes to reposition, hover for a full tooltip (content +
        keywords + tags + context + evolution history), click to inspect.
        Yellow nodes are memories that had a memory-evolution event.

        Call ``.write_html(path, notebook=False, open_browser=False)`` on
        the returned network to save a self-contained HTML file.
        """
        from pyvis.network import Network

        net = Network(
            height=height, width=width, notebook=False,
            directed=False, bgcolor="#ffffff", font_color="#1f2937",
            cdn_resources="remote",
        )
        net.barnes_hut(
            gravity=-3000, central_gravity=0.3,
            spring_length=120, spring_strength=0.04,
        )

        for mid, note in self.memories.items():
            net.add_node(
                mid,
                label=mid,
                title=_pyvis_tooltip(note),
                color="#fbbf24" if note.evolution_history else "#60a5fa",
                shape="dot",
                size=15 + min(len(note.links) * 2, 20),
            )

        seen: set[tuple[str, str]] = set()
        for mid, note in self.memories.items():
            for lid in note.links:
                if lid not in self.memories:
                    continue
                pair = tuple(sorted([mid, lid]))
                if pair in seen:
                    continue
                seen.add(pair)
                net.add_edge(mid, lid, color="#9ca3af", width=1)
        return net

    def to_pyvis_trace(
        self,
        question: str,
        result: dict[str, Any],
        *,
        height: str = "600px",
        width: str = "100%",
    ) -> Network:
        """
        Render a single retrieval trace as an interactive pyvis ``Network``.

        Red diamond = the query.
        Blue dots   = direct top-k vector hits (solid blue edge from Q).
        Purple dots = one-hop link traversals (dashed purple edge from hit).
        """
        from pyvis.network import Network

        net = Network(
            height=height, width=width, notebook=False,
            directed=True, bgcolor="#ffffff", font_color="#1f2937",
            cdn_resources="remote",
        )
        net.barnes_hut(
            gravity=-2500, central_gravity=0.4,
            spring_length=110, spring_strength=0.06,
        )

        net.add_node(
            "_query",
            label="Q",
            title=f"<b>Question</b><br>{html.escape(question)}",
            color="#ef4444", shape="diamond", size=22,
        )

        retrieved_ids = result.get("retrieved_ids", [])
        links_followed = result.get("links_followed", [])
        via_targets = {to_id for _, to_id in links_followed}
        direct_ids = [mid for mid in retrieved_ids if mid not in via_targets]

        added: set[str] = set()
        for mid in direct_ids:
            note = self.memories.get(mid)
            if not note:
                continue
            net.add_node(
                mid, label=mid, title=_pyvis_tooltip(note),
                color="#60a5fa", shape="dot", size=16,
            )
            net.add_edge(
                "_query", mid,
                color="#3b82f6", width=3, title="direct retrieval",
            )
            added.add(mid)

        for from_id, to_id in links_followed:
            target = self.memories.get(to_id)
            if not target:
                continue
            if to_id not in added:
                net.add_node(
                    to_id, label=to_id, title=_pyvis_tooltip(target),
                    color="#a78bfa", shape="dot", size=13,
                )
                added.add(to_id)
            net.add_edge(
                from_id, to_id,
                color="#a78bfa", width=2, dashes=True,
                title="link traversal",
            )
        return net

    # ── Internal helpers ─────────────────────────────────────────────────

    def _find_related_memories(
        self, content: str, k: int = 5,
    ) -> tuple[str, list[str]]:
        """Retrieve top-k neighbors from ChromaDB. Returns formatted text + IDs."""
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
        Re-index all memories in ChromaDB.
        Called periodically after ``evo_threshold`` evolutions.
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
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
    return v


def _mermaid_label(text: str, width: int) -> str:
    """Shorten + sanitise text for use inside a mermaid `id["…"]` label."""
    short = textwrap.shorten(text, width=width, placeholder="…")
    return (
        short.replace('"', "'")
             .replace("\n", " ")
             .replace("|", "/")
    )


def _pyvis_tooltip(note: MemoryNote) -> str:
    """HTML tooltip shown on hover/click in pyvis."""
    evolution = ""
    if note.evolution_history:
        events = "<br>".join(
            f"  · trigger={html.escape(e.get('trigger', '?'))} "
            f"field={html.escape(e.get('field', '?'))}"
            for e in note.evolution_history[:5]
        )
        evolution = (
            f"<br><b>Evolved ({len(note.evolution_history)}×):</b><br>{events}"
        )
    return (
        f"<b>{html.escape(note.id)}</b> "
        f"<span style='color:#6b7280'>({html.escape(note.timestamp)})</span><br>"
        f"<b>Content:</b> {html.escape(note.content)}<br>"
        f"<b>Keywords:</b> {html.escape(', '.join(note.keywords) or '(none)')}<br>"
        f"<b>Tags:</b> {html.escape(', '.join(note.tags) or '(none)')}<br>"
        f"<b>Context:</b> {html.escape(note.context)}<br>"
        f"<b>Links:</b> {html.escape(', '.join(note.links) or '(none)')}"
        f"{evolution}"
    )


def _mermaid_id(raw: str) -> str:
    """Make an ID mermaid-safe (alphanumerics + underscore)."""
    out = []
    for ch in raw:
        out.append(ch if ch.isalnum() else "_")
    safe = "".join(out)
    if safe and safe[0].isdigit():
        safe = "n" + safe
    return safe or "n"
