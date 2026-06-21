# results/

Experiment runs are checked in here as study materials. Each top-level
subdirectory is one **dataset ⨯ test-set** comparison; inside it, each
subdirectory is one **experiment** (`<system>_<paramset>`).

## Layout

```
results/
└── <dataset>_<testset>/                 e.g. comparison_comparison/
    ├── index.html                       side-by-side comparison page
    ├── amem_default/                     one experiment = system + paramset
    │   ├── memory/
    │   │   ├── index.html               memory-structure visualisation
    │   │   └── memory_graph.md          mermaid graph
    │   └── result/
    │       ├── result.json              frozen state (memory + per-query results)
    │       ├── summary.md               per-category accuracy + retrieval recall
    │       ├── traces.md                mermaid per-query traces (A-Mem)
    │       ├── index.html               trace index
    │       └── traces/<QID>.html        per-query retrieval trace
    ├── hipporag_default/                 (same shape; memory/ shows KG triples)
    └── hipporag2_default/                (same shape as hipporag_default/)
```

Statement IDs are `S00001…`; query IDs are `Q00001…`.

## How to (re)generate

```bash
export NVIDIA_API_KEY=nvapi-…
just sync
just compare                                         # amem/hipporag/hipporag2 (default params)
just compare -- --experiments amem:default           # one experiment only
just compare -- --experiments amem:default,amem:wide # tune one system's params
just compare -- --dataset <name> --testset <name>    # different data
just compare -- --limit 5                             # smoke test
just render                                           # re-render HTML only (no LLM)
```

## What to look at

- **`<dataset>_<testset>/index.html`** — start here. Side-by-side per-category and per-question grid. Click any ✓/✗ cell to jump into that experiment's trace for the query.
- **`<experiment>/result/summary.md`** — per-category accuracy + mean retrieval recall for one experiment.
- **`<experiment>/result/traces/<QID>.html`** — full diagnostic for one query: retrieved items (★ = required), seed entities (HippoRAG), filtered triples (HippoRAG 2), A-Mem links followed, retrieval recall/precision.
- **`<experiment>/memory/index.html`** — the constructed memory: A-Mem's interactive note graph (click a node to dim non-neighbours) or HippoRAG's per-statement extracted triples.
