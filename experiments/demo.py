#!/usr/bin/env python3
"""
A-Mem demo — end-to-end walkthrough of the three core operations.

Adds a handful of conversation snippets, shows:
  1. Note construction  (keywords / tags / context extracted by the LLM)
  2. Memory evolution    (existing notes evolving when new info arrives)
  3. Agentic retrieval   (cosine similarity + link traversal)

Requires:
  export NVIDIA_API_KEY=nvapi-…
  pip install openai sentence-transformers chromadb numpy nltk rank_bm25
  python -m experiments.demo
"""

from __future__ import annotations

import os
import sys
import textwrap

# ── Ensure NVIDIA_API_KEY is set ──────────────────────────────────────────

if "NVIDIA_API_KEY" not in os.environ:
    print(
        "ERROR: set NVIDIA_API_KEY first.\n"
        "  export NVIDIA_API_KEY=nvapi-…\n"
    )
    sys.exit(1)

from .amem import AgenticMemorySystem

# ── Demo corpus — conversation snippets from two speakers ─────────────────

CONVERSATIONS = [
    (
        "Speaker Dave says: Hey Calvin, long time no talk! A lot has happened. "
        "I've taken up photography and it's been great — been taking pics of "
        "the scenery around here which is really cool."
    ),
    (
        "Speaker Calvin says: Thanks, Dave! It feels great having my own space "
        "to work in. I've been experimenting with different genres lately, "
        "pushing myself out of my comfort zone. Adding electronic elements to "
        "my songs gives them a fresh vibe. It's been an exciting process of "
        "self-discovery and growth!"
    ),
    (
        "Speaker Dave says: That's awesome, Calvin! Speaking of trying new "
        "things, I've been getting into landscape photography specifically. "
        "The golden hour shots are incredible — the way the light hits the "
        "mountains is just breathtaking."
    ),
    (
        "Speaker Calvin says: I totally understand that feeling! When I compose "
        "music, I try to capture that same kind of natural beauty. Actually, "
        "I've been thinking about scoring a nature documentary — combining my "
        "music with visual storytelling."
    ),
    (
        "Speaker Dave says: That would be a perfect collaboration! I could "
        "provide the photography and visuals while you handle the soundtrack. "
        "I've also started editing my photos with some artistic filters — "
        "really trying to blur the line between photography and art."
    ),
]


def hr(title: str = "") -> None:
    if title:
        print(f"\n{'─' * 4} {title} {'─' * (60 - len(title))}")
    else:
        print("─" * 66)


def print_note(note, prefix: str = "") -> None:
    print(f"{prefix}ID:       {note.id[:12]}…")
    print(f"{prefix}Content:  {textwrap.shorten(note.content, 80)}")
    print(f"{prefix}Keywords: {note.keywords}")
    print(f"{prefix}Tags:     {note.tags}")
    print(f"{prefix}Context:  {note.context}")
    print(f"{prefix}Links:    {[l[:12] + '…' for l in note.links]}")
    if note.evolution_history:
        print(f"{prefix}Evolved:  {len(note.evolution_history)} time(s)")


def main() -> None:
    print("A-Mem Reproduction Demo")
    print(f"LLM:        NVIDIA NIM ({__import__('experiments._nim', fromlist=['LLM_MODEL']).LLM_MODEL})")
    print(f"Embeddings: local sentence-transformers (all-MiniLM-L6-v2)")
    hr()

    mem = AgenticMemorySystem()

    # ── Phase 1: Add memories ────────────────────────────────────────────
    hr("Phase 1 — Adding memories")
    ids = []
    for i, content in enumerate(CONVERSATIONS, 1):
        print(f"\n  [{i}/{len(CONVERSATIONS)}] Adding memory …")
        mid = mem.add_note(content)
        ids.append(mid)
        note = mem.memories[mid]
        print_note(note, prefix="    ")
        print()

    # ── Phase 2: Inspect the memory graph ────────────────────────────────
    hr("Phase 2 — Memory graph after ingestion")
    total_links = 0
    total_evolutions = 0
    for mid in ids:
        note = mem.memories[mid]
        links_display = []
        for lid in note.links:
            target = mem.memories.get(lid)
            if target:
                links_display.append(textwrap.shorten(target.content, 50))
            else:
                links_display.append(lid[:12] + "…")
        print(f"  {note.id[:12]}…  links={len(note.links)}  evolutions={len(note.evolution_history)}")
        for ld in links_display:
            print(f"      → {ld}")
        total_links += len(note.links)
        total_evolutions += len(note.evolution_history)
    print(f"\n  Total links: {total_links}   Total evolution events: {total_evolutions}")

    # ── Phase 3: Agentic retrieval ───────────────────────────────────────
    queries = [
        "What hobby did Dave pick up?",
        "music and nature collaboration",
        "photography techniques",
    ]
    hr("Phase 3 — Agentic retrieval")
    for query in queries:
        print(f"\n  Query: \"{query}\"")
        results = mem.search_agentic(query, k=3)
        for j, r in enumerate(results):
            source = "(link)" if "via_link_from" in r else "(direct)"
            print(f"    [{j + 1}] {source} {textwrap.shorten(r['content'], 70)}")
        if not results:
            print("    (no results)")

    hr()
    print("Demo complete.")


if __name__ == "__main__":
    main()
