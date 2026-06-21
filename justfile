# Recipes for the A-Mem + HippoRAG comparison reproduction.
# Install just from https://github.com/casey/just  (brew install just / apt install just).

default: help

help:
    @just --list

# Sync Python dependencies via uv (creates .venv).
sync:
    uv sync

# Quick A-Mem walkthrough — 10 statements + 3 queries (~1 min).
demo:
    uv run python -m experiments.demo

# A-Mem (default params) only — full eval against the comparison test-set.
# Writes results/comparison_comparison/amem_default/{memory,result}/.
eval *ARGS:
    uv run python -m experiments.compare --experiments amem:default {{ARGS}}

# Run + compare experiments (system:paramset) on one dataset ⨯ test-set.
# Writes results/<dataset>_<testset>/{index.html, <system>_<paramset>/{memory,result}/}.
# Examples:
#   just compare
#   just compare -- --dataset comparison --testset comparison \
#       --experiments amem:default,hipporag:default,hipporag2:default
#   just compare -- --experiments amem:default,amem:wide
#   just compare -- --limit 5
compare *ARGS:
    uv run python -m experiments.compare {{ARGS}}

# Re-render every experiment's memory/ + result/ HTML and the comparison
# index.html from existing result.json files under results/. No LLM calls —
# use after editing experiments/render.py or the per-system renderers.
# Examples:
#   just render                                       # all comparison dirs
#   just render -- --comparison comparison_comparison # one dir only
render *ARGS:
    uv run python -m experiments.render {{ARGS}}
