# Recipes for the A-Mem + HippoRAG comparison reproduction.
# Install just from https://github.com/casey/just  (brew install just / apt install just).

default: help

help:
    @just --list

# Sync Python dependencies via uv (creates .venv).
sync:
    uv sync

# Quick A-Mem walkthrough — 10 memories + 3 questions (~1 min).
demo:
    uv run python -m experiments.demo

# A-Mem only — full eval against the comparison dataset.
# Writes results/comparison/amem/{run.json, summary.md, html/}.
eval *ARGS:
    uv run python -m experiments.compare --systems amem {{ARGS}}

# Compare A-Mem ⨯ HippoRAG v1 ⨯ HippoRAG v2 on the same dataset.
# Writes results/<dataset>/{result.html, <system>/{run.json, summary.md, html/}}.
# Examples:
#   just compare
#   just compare -- --dataset comparison --systems amem,hipporag,hipporag2
#   just compare -- --limit 5
compare *ARGS:
    uv run python -m experiments.compare {{ARGS}}

# Re-render A-Mem mermaid + pyvis from an existing results/run.json (no LLM).
render *ARGS:
    uv run python -m experiments.render {{ARGS}}
