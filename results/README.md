# results/

Comparison runs are checked in here as study materials. Each subdirectory is one dataset.

## Layout

```
results/
└── <dataset_name>/                      e.g. comparison/
    ├── result.html                      side-by-side comparison page
    ├── amem/
    │   ├── run.json                     frozen state (memories + per-Q results)
    │   ├── summary.md                   per-category accuracy table
    │   ├── memory_graph.md              mermaid graph
    │   ├── traces.md                    mermaid per-query traces
    │   └── html/
    │       ├── index.html               trace index
    │       ├── memory_graph.html        interactive pyvis graph
    │       └── traces/<qid>.html        per-question retrieval trace
    ├── hipporag/                        (same shape; no memory_graph — KG only)
    │   ├── run.json
    │   ├── summary.md
    │   └── html/{index,traces/...}
    └── hipporag2/                       (same shape as hipporag/)
```

## How to (re)generate

```bash
export NVIDIA_API_KEY=nvapi-…
just sync
just compare                              # full A-Mem ⨯ HippoRAG ⨯ HippoRAG2 on the comparison dataset
just compare -- --systems amem            # one system only
just compare -- --dataset <name>          # different dataset (registered in experiments/datasets/__init__.py)
just compare -- --limit 5                 # smoke test
```

## What to look at

- **`<dataset>/result.html`** — start here. Side-by-side per-category and per-question grid. Click any ✓/✗ cell to jump into that system's trace for the question.
- **`<system>/summary.md`** — quick per-category number for one system.
- **`<system>/html/traces/<qid>.html`** — full diagnostic for one question on one system: retrieved items, seed entities (HippoRAG), filtered triples (HippoRAG 2), A-Mem links followed.
- **`amem/html/memory_graph.html`** — full A-Mem memory graph (click a node to dim non-neighbours; hover for full content + evolution history).
