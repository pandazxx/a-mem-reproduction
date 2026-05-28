# Recipes for the A-Mem reproduction demo.
# Install just from https://github.com/casey/just  (brew install just / apt install just).

default: help

help:
    @just --list

# Sync Python dependencies via uv (creates .venv).
sync:
    uv sync

# Run the A-Mem demo — note construction, memory evolution, agentic retrieval.
demo:
    uv run python -m experiments.demo
