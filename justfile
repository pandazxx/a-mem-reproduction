# Recipes for the A-Mem reproduction demo.
# Install just from https://github.com/casey/just  (brew install just / apt install just).

default: help

help:
    @just --list

# Sync Python dependencies via uv (creates .venv).
sync:
    uv sync

# Quick walkthrough — 10 memories + 3 questions (~1 min). Writes results/demo/.
demo:
    uv run python -m experiments.demo

# Full evaluation — 40 memories + 22 questions. Writes results/run.json,
# results/summary.md, then renders mermaid + interactive HTML.
eval *ARGS:
    uv run python -m experiments.eval {{ARGS}}

# Re-render mermaid + pyvis from results/run.json (no LLM calls).
# Use after editing experiments/render.py to iterate on visualisations.
render *ARGS:
    uv run python -m experiments.render {{ARGS}}
