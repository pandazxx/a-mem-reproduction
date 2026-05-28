#!/usr/bin/env python3
"""
A-Mem demo — end-to-end walkthrough of the three core operations.

Uses the same 15-passage corpus and 4 multi-hop questions as the HippoRAG
reproduction (https://github.com/pandazxx/hipporag-reproduction) so the
two systems can be compared side by side.

Shows:
  1. Note construction  (keywords / tags / context extracted by the LLM)
  2. Memory evolution    (existing notes evolving when new info arrives)
  3. Agentic retrieval   (cosine similarity + link traversal)

Requires:
  export NVIDIA_API_KEY=nvapi-…
  just sync
  just demo
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

# ── Corpus — same 15 passages used in the HippoRAG reproduction ──────────
# Allows direct comparison of A-Mem vs HippoRAG on identical data.

PASSAGES = [
    "The Quantum Computing Lab was founded by Professor Alice Chen in 2010.",
    "Professor Alice Chen has published over 200 papers on quantum error correction.",
    "The Quantum Computing Lab is located at Stanford University.",
    "Stanford University is one of the top research universities in California.",
    "Dr. Bob Martinez works at the Quantum Computing Lab on quantum algorithms.",
    "Dr. Bob Martinez is supervised by Professor Alice Chen.",
    "Quantum algorithms are a key research area at MIT and Stanford University.",
    "Professor Carol White is the director of the Physics department at Stanford University.",
    "The Physics department at Stanford University hosts the Quantum Computing Lab.",
    "Dr. Emily Davis recently joined the Quantum Computing Lab from Google Brain.",
    "Google Brain is a research division of Google focused on deep learning.",
    "Dr. Emily Davis works on quantum machine learning algorithms.",
    "Professor Alice Chen received the Turing Award in 2019.",
    "The Turing Award is given by the Association for Computing Machinery.",
    "The Quantum Computing Lab received a $10M NSF grant in 2023.",
]

# ── Multi-hop questions — same as the HippoRAG reproduction ──────────────

TEST_QUESTIONS = [
    {
        "q":            "Who supervises the researcher working on quantum algorithms?",
        "answer":       "Professor Alice Chen",
        "hops":         2,
        "chain":        "quantum algorithms → Dr. Bob Martinez → Professor Alice Chen",
        "key_passages": {4, 5},
    },
    {
        "q":            "What university hosts the lab that received the NSF grant?",
        "answer":       "Stanford University",
        "hops":         2,
        "chain":        "NSF grant → Quantum Computing Lab → Stanford University",
        "key_passages": {14, 2},
    },
    {
        "q":            "What award did the founder of the lab where Dr. Bob Martinez works receive?",
        "answer":       "Turing Award",
        "hops":         3,
        "chain":        "Dr. Bob Martinez → Quantum Computing Lab → Professor Alice Chen → Turing Award",
        "key_passages": {4, 0, 12},
    },
    {
        "q":            "What research field does the organization Dr. Emily Davis came from focus on?",
        "answer":       "deep learning",
        "hops":         2,
        "chain":        "Dr. Emily Davis → Google Brain → deep learning",
        "key_passages": {9, 10},
    },
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
    from . import _nim

    print("A-Mem Reproduction Demo")
    print(f"LLM:        NVIDIA NIM ({_nim.LLM_MODEL})")
    print(f"Embeddings: local sentence-transformers (all-MiniLM-L6-v2)")
    print(f"Corpus:     {len(PASSAGES)} passages (same as HippoRAG reproduction)")
    hr()

    mem = AgenticMemorySystem()

    # ── Phase 1: Ingest passages ─────────────────────────────────────────
    hr("Phase 1 — Ingesting passages")
    ids: list[str] = []
    for i, passage in enumerate(PASSAGES):
        print(f"\n  [{i + 1:2d}/{len(PASSAGES)}] {textwrap.shorten(passage, 65)}")
        mid = mem.add_note(passage)
        ids.append(mid)
        note = mem.memories[mid]
        print(f"         keywords={note.keywords}")
        print(f"         tags={note.tags}")
        print(f"         links={len(note.links)}  evolved_neighbors={len(note.evolution_history)}")

    # ── Phase 2: Memory graph summary ────────────────────────────────────
    hr("Phase 2 — Memory graph after ingestion")
    total_links = 0
    total_evolutions = 0
    for i, mid in enumerate(ids):
        note = mem.memories[mid]
        links_display = []
        for lid in note.links:
            target = mem.memories.get(lid)
            if target:
                links_display.append(textwrap.shorten(target.content, 50))
            else:
                links_display.append(lid[:12] + "…")
        marker = "*" if note.evolution_history else " "
        print(f"  {marker} [{i:2d}] links={len(note.links):2d}  evolutions={len(note.evolution_history)}"
              f"  {textwrap.shorten(note.content, 55)}")
        for ld in links_display:
            print(f"           → {ld}")
        total_links += len(note.links)
        total_evolutions += len(note.evolution_history)
    print(f"\n  Total links: {total_links}   Total evolution events: {total_evolutions}")

    # ── Phase 3: Multi-hop retrieval ─────────────────────────────────────
    hr("Phase 3 — Multi-hop retrieval (compare with HippoRAG)")
    for tq in TEST_QUESTIONS:
        query = tq["q"]
        expected = tq["answer"]
        chain = tq["chain"]
        hops = tq["hops"]
        key_idxs = tq["key_passages"]

        print(f"\n  Q ({hops}-hop): {query}")
        print(f"  Expected: {expected}")
        print(f"  Chain:    {chain}")

        results = mem.search_agentic(query, k=5)

        retrieved_passages = [r["content"] for r in results]
        key_found = sum(
            1 for idx in key_idxs
            if any(PASSAGES[idx] in rp for rp in retrieved_passages)
        )

        print(f"  Key passages found: {key_found}/{len(key_idxs)}")
        for j, r in enumerate(results):
            source = "(link)" if "via_link_from" in r else "(direct)"
            hit = "✓" if any(PASSAGES[idx] == r["content"] for idx in key_idxs) else " "
            print(f"    {hit} [{j + 1}] {source} {textwrap.shorten(r['content'], 65)}")

    hr()
    print("Demo complete.")


if __name__ == "__main__":
    main()
