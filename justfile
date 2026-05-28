# Recipes for the A-Mem reproduction demo.
# Install just from https://github.com/casey/just  (brew install just / apt install just).

default: help

help:
    @just --list

# Sync Python dependencies via uv (creates .venv).
sync:
    uv sync

# Quick walkthrough — 10 memories + 3 questions (~1 min).
demo:
    uv run python -m experiments.demo

# Full evaluation — 40 memories + 22 questions against the comparison dataset.
# Writes results/results.json and results/summary.md.
eval *ARGS:
    uv run python -m experiments.eval {{ARGS}}
