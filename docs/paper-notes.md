# A-Mem — Paper Study Notes

Notes from reading *A-Mem: Agentic Memory for LLM Agents* (Xu et al., 2025).

- Paper: https://arxiv.org/abs/2502.12110
- Code (benchmark evaluation): https://github.com/WujiangXu/AgenticMemory
- Code (production-ready): https://github.com/WujiangXu/A-mem-sys
- Authors: Wujiang Xu et al., Rutgers University + AIOS Foundation

**Prerequisite reading:** HippoRAG 1 paper notes at `hipporag-reproduction/docs/paper-notes.md`. A-Mem is a fundamentally different paradigm from HippoRAG — compare them side by side.

---

## TL;DR

A-Mem is a **Zettelkasten-inspired** memory system for LLM agents. Each memory is an atomic "note" with LLM-generated structured attributes (keywords, tags, context). Notes are dynamically linked based on shared attributes and semantic similarity — the LLM decides the links, not just embedding distance. Crucially, when a new memory arrives, **existing linked memories are updated** (their attributes evolve based on the new information). This "memory evolution" mechanism is the closest published analogue to biological reconsolidation.

The paradigm difference from HippoRAG: HippoRAG builds a knowledge graph from explicit triples and retrieves via graph algorithms (PPR). A-Mem builds a note network from LLM-generated attributes and retrieves via embedding similarity + link traversal. HippoRAG is algorithm-driven; A-Mem is LLM-driven at every step.

---

## Core inspiration: the Zettelkasten method

The Zettelkasten ("slip box") is a knowledge management system invented by sociologist Niklas Luhmann. Its principles:

1. **Atomicity** — each note captures exactly one idea, self-contained.
2. **Linking** — notes are connected to related notes via explicit links, forming a web.
3. **Emergence** — over time, the link network reveals patterns and connections that no individual note contains.

A-Mem directly implements these three principles with LLMs:
- Each memory = one atomic note.
- Links = LLM-determined connections between notes.
- Evolution = the network refines itself as new notes arrive.

---

## The note structure

Each memory note `m_i` contains seven fields:

```
m_i = { c_i, t_i, K_i, G_i, X_i, e_i, L_i }
```

| Field | What it is | Who generates it |
|---|---|---|
| `c_i` | Original interaction content (raw text) | The agent's environment |
| `t_i` | Timestamp | System clock |
| `K_i` | Keywords (key concepts extracted from `c_i`) | **LLM** |
| `G_i` | Tags (categorisation labels) | **LLM** |
| `X_i` | Contextual description (rich semantic summary) | **LLM** |
| `e_i` | Embedding vector | **Text encoder** (all-minilm-l6-v2) |
| `L_i` | Set of linked memory IDs | **LLM** (during link generation) |

The embedding `e_i` is computed from the **concatenation** of all text fields: `e_i = f_enc(concat(c_i, K_i, G_i, X_i))`. So the vector captures not just the raw content but the LLM's interpretation of it.

**Key design choice:** three of seven fields (K, G, X) are LLM-generated enrichments of the raw content. This means A-Mem is using the LLM not just as a retriever but as a **comprehension engine** that adds structure to raw experience. The note is richer than the original interaction.

---

## Three core operations

### Operation 1 — Note Construction (§3.1)

**When:** every new interaction with the environment.

**Steps:**
1. Receive raw interaction content `c_i` and timestamp `t_i`.
2. **LLM call** (prompt `P_s1`): generate keywords `K_i`, tags `G_i`, and contextual description `X_i` from the raw content.
3. Compute embedding `e_i` from the concatenation of all text fields.
4. Initialise link set `L_i = {}` (empty; will be populated by link generation).

**Cost:** 1 LLM call per new memory. Input: raw content + prompt template. Output: structured attributes.

**Comparison to HippoRAG:** HippoRAG's indexing uses OpenIE to extract (subject, predicate, object) triples — structured but rigid. A-Mem uses free-form LLM generation — flexible but potentially noisy. HippoRAG discards the predicate; A-Mem preserves rich context.

### Operation 2 — Link Generation (§3.2)

**When:** immediately after a new note is constructed.

**Steps:**
1. Use the new note's embedding `e_n` to retrieve top-k most similar existing notes via cosine similarity.
2. **LLM call** (prompt `P_s2`): given the new note + the top-k candidates, determine which candidates should be linked.
3. Update `L_n` for the new note and `L_j` for each newly linked note (bidirectional).

**Why LLM-determined links, not just embedding threshold?**
- Embedding similarity catches surface-level relatedness ("same words").
- The LLM catches deeper connections: causal relationships, conceptual analogies, shared implications.
- Example: "I'm learning Python" and "I built a web scraper" have moderate embedding similarity, but the LLM recognises the causal connection (Python → web scraper).

**Two-stage process:** embedding retrieval first (cheap, filters to top-k), then LLM analysis (expensive, decides actual links). This is the same "recall then recognise" pattern as HippoRAG 2's triple filtering.

**Cost:** 1 LLM call per new memory (the link-decision call).

### Operation 3 — Memory Evolution (§3.3)

**When:** immediately after links are generated for a new note.

**This is the most important mechanism in the paper for your research.**

**Steps:**
1. For each memory `m_j` in the newly linked set `M_near`:
2. **LLM call** (prompt `P_s3`): given the new note `m_n`, the linked set `M_near`, and the existing note `m_j`, decide whether to update `m_j`'s context, keywords, and tags.
3. If the LLM decides to update: the evolved note `m_j*` replaces the original `m_j` in the memory set.

**Formal expression:**
```
m_j* ← LLM(m_n || M_near \ m_j || m_j || P_s3)
```

**What gets updated:** the textual attributes (K, G, X) of existing memories. The raw content `c_j` is preserved. The embedding `e_j` is **not explicitly re-computed** in the paper's description — this is a potential issue (the vector becomes stale relative to the updated text).

**Why this matters for reconsolidation research:**
- This IS a form of reconsolidation — existing memories are modified when new information arrives.
- But it's reconsolidation at **write time** (triggered by a new memory arriving), not at **read time** (triggered by retrieval).
- Biological reconsolidation happens on recall — every retrieval re-opens the memory for editing. A-Mem doesn't do that.
- The gap: what if retrieval also triggered evolution? That would be true read-time reconsolidation.

**Cost:** up to k LLM calls per new memory (one per linked note that gets evolved). In practice, not all linked notes need evolution.

### Total cost per memory operation

| Step | LLM calls | Other compute |
|---|---|---|
| Note construction | 1 | 1 embedding |
| Link generation | 1 (includes retrieval) | top-k cosine search |
| Memory evolution | up to k (one per evolved note) | — |
| **Total** | **2 + up to k** | 1 embedding + 1 cosine search |

The paper reports ~1,200 tokens per memory operation on average, which is **85–93% cheaper** than LoCoMo (~16,900 tokens) and MemGPT (~16,900 tokens). Cost: <$0.0003 per memory operation with commercial APIs.

---

## Retrieval (§3.4)

**When:** the agent needs to respond to a query using historical context.

**Steps:**
1. Embed the query: `e_q = f_enc(q)`.
2. Cosine similarity against all note embeddings: `s_{q,i} = cos(e_q, e_i)`.
3. Retrieve top-k notes by similarity.
4. **Link traversal ("relative memory"):** for each retrieved note, also pull its linked notes `L_i`.
5. Combine all retrieved + linked notes as context for the LLM to generate a response.

**Comparison to HippoRAG:** HippoRAG uses PPR for multi-hop retrieval through the graph. A-Mem uses one-hop link traversal from the directly-retrieved notes. A-Mem's "multi-hop" is limited to the depth of the link structure (usually 1 hop). For deeper chains, HippoRAG's PPR is more powerful.

---

## Implementation details from the paper

| Component | Choice | Notes |
|---|---|---|
| Text embedding model | **all-minilm-l6-v2** | Small (384 dims), fast, free. Much weaker than NV-Embed-v2 or Contriever used in HippoRAG. |
| LLM (primary) | **GPT-4o-mini** (API) | Also tested: Qwen-1.5B/3B, Llama 3.2 1B/3B, DeepSeek-R1-32B, Claude 3.0/3.5 Haiku |
| LLM (local) | **Ollama + LiteLLM** | For Qwen and Llama models |
| Top-k retrieval | k=10 (default) | Adjustable per task category |
| Benchmark | LoCoMo (7,512 QA, 5 categories) + DialSim (TV shows, 1,300 sessions) | |
| Baselines | LoCoMo, ReadAgent, MemoryBank, MemGPT | |

**Processing time:** 5.4s per operation with GPT-4o-mini, 1.1s with local Llama 3.2 1B.

**Scaling:** retrieval time increases from 0.31μs to 3.70μs going from 1K to 1M memories — near-linear scaling, which is just the cosine-similarity search growing with N.

---

## Results highlights

### LoCoMo benchmark (Table 1 in the paper)

A-Mem consistently ranks #1 across non-GPT models. Key findings:

- **Multi-hop** is where A-Mem shines most: roughly **doubles** performance vs baselines on multi-hop reasoning. This is the link-traversal mechanism earning its keep.
- **Single-hop** is good but not dominant — LoCoMo and MemGPT sometimes match on simple fact retrieval.
- **Adversarial** (unanswerable questions): A-Mem performs well, suggesting the structured notes help the LLM recognise when memory is insufficient.
- **Token efficiency:** A-Mem uses ~1,200 tokens per operation vs ~16,900 for baselines (85–93% reduction).

### Ablation study (Table 3)

| Variant | Multi-Hop F1 | Single-Hop F1 | Insight |
|---|---|---|---|
| w/o Link Generation AND Memory Evolution | 9.65 | 13.28 | Terrible — links are essential |
| w/o Memory Evolution (links only) | 21.35 | 39.17 | Links alone get you halfway |
| Full A-Mem | **27.02** | **44.65** | Memory evolution adds meaningful refinement |

**Takeaway:** link generation is the *critical* component (without it, the system collapses). Memory evolution is the *refinement* layer that pushes quality up, especially on multi-hop.

---

## Cost economics

| Metric | A-Mem | LoCoMo/MemGPT baselines |
|---|---|---|
| Tokens per memory operation | ~1,200 | ~16,900 |
| Cost per operation (API) | <$0.0003 | ~$0.004 |
| Processing time (GPT-4o-mini) | 5.4s | varies |
| Processing time (local Llama 3.2 1B) | 1.1s | varies |

A-Mem is dramatically cheaper per operation because it operates on compact notes rather than full conversation history.

---

## Comparison: A-Mem vs HippoRAG

| Aspect | HippoRAG 1/2 | A-Mem |
|---|---|---|
| **Inspiration** | Hippocampal indexing theory | Zettelkasten method |
| **Storage unit** | Entities (phrase nodes) + passages | Atomic notes with structured attributes |
| **Linking mechanism** | OpenIE triples + synonymy edges (embedding threshold) | **LLM-determined** links based on deep analysis |
| **Retrieval** | PPR over the graph (multi-hop via propagation) | Cosine similarity + 1-hop link traversal |
| **Memory update** | Static — graph never changes after indexing | **Memory evolution** — existing notes update on new information |
| **Cost structure** | Heavy offline indexing, cheap per-query | Moderate per-memory-operation, moderate per-query |
| **Multi-hop strength** | PPR naturally follows chains of any depth | Limited to link depth (usually 1 hop) |
| **Embedding model** | Contriever (768d) / NV-Embed-v2 (1024d) | all-minilm-l6-v2 (384d) — weaker |
| **Predicate/relation** | Extracted then discarded | Not extracted — tags and context capture semantic meaning instead |
| **Reconsolidation** | None | **Memory evolution** — closest to reconsolidation at write time |

**The fundamental trade-off:** HippoRAG is stronger at deep multi-hop retrieval (PPR propagates indefinitely). A-Mem is stronger at memory *management* (evolution, dynamic linking, structured attributes). They solve different halves of the memory problem.

---

## What A-Mem teaches about reconsolidation

### What it does (write-time reconsolidation)
When memory_new arrives → linked memories are re-evaluated → their attributes potentially updated. This is reconsolidation triggered by *new input*, not by *retrieval*.

### What it doesn't do (read-time reconsolidation)
When a memory is retrieved for a query, nothing changes. The retrieval is read-only. The memory's attributes, links, and embedding remain unchanged regardless of how many times it's accessed.

### What biological reconsolidation actually does
In the brain, *every retrieval* re-opens the memory for modification. The act of recalling a fact makes it temporarily editable. If the recall context adds new information (or contradicts the memory), the memory is updated before being re-stored.

### The gap your project could fill
Add read-time reconsolidation to A-Mem (or to a HippoRAG-style system):
- When a memory is retrieved, the system checks whether the query context provides information that should update the memory.
- If yes: update the note's K, G, X fields; re-compute embedding; potentially update links.
- This would make every retrieval a potential write operation — matching the biological model.

---

## Things to watch for during reproduction

1. **Two separate codebases.** The benchmark evaluation code (`AgenticMemory`) and the production-ready code (`A-mem-sys`) are different repos. For reproduction, use the benchmark code.

2. **Embedding model is weak.** all-minilm-l6-v2 is a 2022-era model, 384 dimensions. Much weaker than what HippoRAG uses. Swapping for a stronger model (e.g., all-MiniLM-L12-v2 or text-embedding-3-small) could change results — document the choice.

3. **LLM prompt sensitivity.** Three separate prompts (P_s1 for note construction, P_s2 for link generation, P_s3 for memory evolution) are critical. The prompts are in Appendix B of the paper. Read them carefully — the quality of generated keywords/tags/context depends entirely on these prompts.

4. **Memory evolution re-embedding question.** The paper describes updating K, G, X of existing notes but is ambiguous about whether the embedding `e_j` is recomputed after evolution. If not, the embedding becomes stale relative to the updated text — a real retrieval-quality issue.

5. **LoCoMo benchmark setup.** A-Mem evaluates on LoCoMo (same benchmark we studied in the benchmark deep-dive). 7,512 QA pairs across 5 categories. You'll need to set this up.

6. **LiteLLM + Ollama for local models.** If using local LLMs (which keeps costs zero), you'll need both installed.

7. **k=10 default but category-specific tuning.** The paper uses k=10 for most categories but adjusts for specific ones. Check Appendix A.5 for exact values.

---

## Study questions

### Section A — Mechanics

1. Walk through what happens when a single new interaction arrives. What LLM calls are made, in what order, and what data flows between them?
2. What are the seven fields of a memory note? Which are generated by the LLM vs computed algorithmically?
3. How does link generation work? Why is it a two-stage process (embedding retrieval first, then LLM decision)?
4. What exactly gets updated during memory evolution? What stays unchanged?
5. How does retrieval work? What is "relative memory" and how does it enable multi-hop?

### Section B — Reasoning

6. Why does A-Mem use all-minilm-l6-v2 (a weak model) for embeddings? What would change if you swapped for a stronger model?
7. The ablation shows that link generation matters more than memory evolution. Why?
8. A-Mem uses ~1,200 tokens per operation vs ~16,900 for baselines. Where does the 85-93% saving come from?
9. How does A-Mem's link traversal compare to HippoRAG's PPR for multi-hop retrieval? Which is more powerful, and in what cases?

### Section C — Critique

10. Memory evolution updates K, G, X but the paper is ambiguous about re-computing embeddings. What happens if the embedding is NOT recomputed? When would this matter?
11. A-Mem's links are LLM-determined. What failure modes does this introduce that HippoRAG's algorithmic edges don't have?
12. The paper evaluates on LoCoMo and DialSim. Are these representative of long-term agent memory? What would the results look like on MemoryAgentBench?
13. What happens to A-Mem's performance as the memory grows to 1M+ notes? The scaling analysis shows retrieval time is fine — but is retrieval *quality* fine?

### Section D — Application to your project

14. A-Mem's memory evolution is reconsolidation at *write time*. How would you add reconsolidation at *read time* (triggered by retrieval)?
15. Could you combine A-Mem's note structure with HippoRAG's PPR retrieval? What would that architecture look like?
16. A-Mem has no forgetting mechanism. Where would you add it? What should trigger forgetting?
17. Which parts of A-Mem's codebase could you reuse in your main project? Which would you replace?

---

## Reading order for the paper

1. **Abstract + §1 Introduction** — understand the Zettelkasten framing and why existing systems are insufficient.
2. **§3 Methodology** — the core: note construction, link generation, memory evolution, retrieval. Read slowly; this is the load-bearing section.
3. **Figure 2** — the architecture diagram. Make sure you can trace a complete memory lifecycle through it.
4. **§4.3 Results + §4.4 Ablation** — what works and what matters.
5. **§4.5 Hyperparameters + §4.6 Scaling** — practical deployment characteristics.
6. **Appendix B** — the actual prompts (P_s1, P_s2, P_s3). These are the system's "source code" in a real sense.
7. **Then re-read §3.3 Memory Evolution** with the reconsolidation lens from your broader research notes.
