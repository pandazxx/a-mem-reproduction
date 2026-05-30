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

# Re-render every system's HTML + result.html from existing run.json files
# under results/. No LLM calls — use after editing experiments/render.py
# or the per-system renderers to iterate on visualisations.
# Examples:
#   just render                              # re-render everything under results/
#   just render -- --dataset comparison      # one dataset only
render *ARGS:
    uv run python -m experiments.render {{ARGS}}
