# HippoRAG vs A-Mem — Comparison Dataset

A small, hand-designed dataset for testing the architectural differences between HippoRAG and A-Mem head-to-head. The design intentionally creates conditions where each system's strengths and weaknesses become visible.

This is **not** meant to be a published benchmark — it's a diagnostic tool for the project's reproduction phase. Roughly 40 memories + 22 questions across 7 categories, designed to be tractable in a few hours of evaluation.

---

## Why a custom dataset

Existing benchmarks (LoCoMo, MemoryAgentBench, Memora) test memory broadly. They don't isolate the specific architectural differences between HippoRAG (graph + PPR) and A-Mem (notes + LLM-determined links + memory evolution).

This dataset is **designed to make the differences visible** by including question categories where:

- **HippoRAG should win** (deep multi-hop chains, large-scale aggregation)
- **A-Mem should win** (implicit conceptual links, information updates)
- **Both should tie** (single-hop, absence/abstention)

If a system wins everywhere or loses everywhere on this dataset, that itself is a finding worth investigating (likely a reproduction bug).

---

## Scenario

A single user named **Sam** (a software engineer) interacts with an AI assistant over 8 weeks. Sam shares facts about their work, hobbies, relationships, projects, and family. Memories are presented in chronological order, simulating real-time accumulation.

The "world" is small and coherent — about 8 entities (people), 3 organizations, 3 locations, several projects and hobbies. This keeps the comparison interpretable.

Crucially, **Sam changes jobs mid-stream** (TechCorp → StartupCo, Week 7). This creates a deliberate contradiction in the memory store — the kind of update that A-Mem's memory evolution should handle better than HippoRAG's static graph.

---

## Categories and expected outcomes

| # | Category | What it stresses | Expected winner | Q count |
|---|---|---|---|---|
| 1 | Single-hop factual | Direct retrieval — control case | **Tie** | 4 |
| 2 | Two-hop chains | Light multi-hop | **Tie** or slight HippoRAG | 3 |
| 3 | Deep multi-hop (3+ hops) | PPR propagation through chains | **HippoRAG** | 3 |
| 4 | Implicit conceptual links | LLM-determined connections | **A-Mem** | 3 |
| 5 | Information update / contradiction | Memory evolution on stale facts | **A-Mem** | 3 |
| 6 | Compositional aggregation | "List all X" across many memories | **HippoRAG** | 3 |
| 7 | Absence / abstention | Recognizing missing information | **Tie** (both should abstain) | 3 |

Total: **22 questions** across **7 categories**.

---

## File structure

```
comparison-dataset/
├── README.md          ← you are here
├── dataset.json       ← memories + questions in machine-readable form
├── analysis.md        ← hypotheses, success criteria, how to interpret results
└── eval_template.py   ← skeleton evaluation script for both systems
```

---

## How to use this dataset

### With your HippoRAG reproduction

1. Load `dataset.json` and extract the `memories` list.
2. Feed each memory's `content` (with timestamp) to the HippoRAG indexing pipeline as if it were a passage.
3. For each question in `dataset.json`, run the query and capture the retrieved passages + final answer.
4. Compare against `expected_answer` and `requires_facts`.

### With your A-Mem reproduction

1. Same: load `dataset.json`.
2. Feed each memory to A-Mem's note construction pipeline (the memories are already in chronological order, so memory evolution will trigger naturally).
3. For each question, call A-Mem's retrieval and answer-generation pipeline.
4. Compare against `expected_answer`.

### Evaluation harness

`eval_template.py` provides a skeleton with:
- Loaders for both systems
- A unified question-running interface
- Score computation (exact match, F1, LLM-judge optional)
- Per-category breakdown
- Tabular comparison output

You'll need to fill in the system-specific glue code (the imports and constructor calls for each system). The harness itself is system-agnostic.

---

## What "winning" means here

For each question category, the dataset specifies an **expected winner**. After running both systems:

- If the expected winner wins by a clear margin in that category → the architectural hypothesis is confirmed.
- If the expected winner *loses* or they're tied → something interesting is happening. Investigate.
- If one system dominates *every* category → likely a reproduction bug or a benchmark contamination issue.

The point is **diagnostic insight**, not aggregate scores. Pay attention to *which* questions each system gets right or wrong, not just the totals.

---

## Known limitations

- **Tiny scale.** 40 memories vs the thousands in production benchmarks. Results don't extrapolate to large-scale behavior.
- **English-only, synthetic.** Real user conversations are messier.
- **Author bias.** I designed this dataset with hypotheses in mind, so the questions may inadvertently favor what I expected to find. Treat results with appropriate skepticism.
- **Single user, no multi-user privacy concerns.** Doesn't test cross-user contamination.
- **Single-turn questions.** No multi-turn dialogue dynamics.

For real evaluation, supplement this with a slice of MemoryAgentBench or LongMemEval. This dataset is for **diagnostic intuition**, not paper-grade evaluation.

---

## After running the dataset

Once you have results from both systems, fill in the `results` table in `analysis.md`. Then write up findings:

1. Did the expected winners actually win in each category?
2. What's the most interesting per-question disagreement (one system right, one wrong)?
3. What does this tell you about the architectural choices?
4. Which findings would inform a reconsolidation-focused project design?

Bring the filled-in `analysis.md` back and we can discuss what to write up.
