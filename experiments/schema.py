"""
Core data model for the memory-experiment framework.

The framework separates four first-class concepts (issue #4):

  Dataset      — a list of *statements* the system ingests to build memory.
                 ``datasets/<name>/dataset.json`` → ``{metadata, statements}``.
                 Each statement: ``{id: "S00001", statement, timestamp, comment}``.

  Test-set     — a list of *queries* run against a memory built from one
                 dataset. ``testsets/<name>/testset.json`` → ``{metadata, queries}``.
                 Each query: ``{id: "Q00001", query, required_retrieval: [S…],
                 expected_answer, comment}`` (+ optional ``category`` /
                 ``expected_winner`` used by the diagnostic comparison page).

  Parameter set — a named bag of tuning knobs fed to one system.
                 ``params/<system>/<paramset>.json`` → ``{...}``.

  Experiment   — the runnable unit: (system, parameter set) evaluated against
                 (dataset, test-set). Identified on disk by ``<system>_<paramset>``.

Loaders here read straight off disk (no import-time registry) so dropping a
new ``dataset.json`` / ``testset.json`` / ``params/*.json`` makes it available
immediately. ``available_*`` helpers enumerate what's on disk for the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
DATASETS_DIR = _HERE / "datasets"
TESTSETS_DIR = _HERE / "testsets"
PARAMS_DIR = _HERE / "params"


# ── Dataset ───────────────────────────────────────────────────────────────

@dataclass
class Dataset:
    """A list of statements, loaded from ``datasets/<name>/dataset.json``."""

    name: str
    path: Path
    statements: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def ingest_items(self) -> list[dict[str, Any]]:
        """
        Adapt statements to the ``SystemAdapter.ingest`` contract, which
        expects ``{id, content, timestamp}``. Bridges the framework's
        ``statement`` field name to the adapters' generic ``content``.
        """
        return [
            {
                "id": s["id"],
                "content": s["statement"],
                "timestamp": s.get("timestamp", ""),
            }
            for s in self.statements
        ]


def load_dataset(name: str) -> Dataset:
    path = DATASETS_DIR / name / "dataset.json"
    if not path.exists():
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {available_datasets()}"
        )
    data = json.loads(path.read_text())
    return Dataset(
        name=name,
        path=path,
        statements=data["statements"],
        metadata=data.get("metadata", {}),
    )


def available_datasets() -> list[str]:
    if not DATASETS_DIR.exists():
        return []
    return sorted(
        p.name for p in DATASETS_DIR.iterdir()
        if (p / "dataset.json").exists()
    )


# ── Test-set ──────────────────────────────────────────────────────────────

@dataclass
class TestSet:
    """A list of queries, loaded from ``testsets/<name>/testset.json``."""

    name: str
    path: Path
    dataset: str
    queries: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_testset(name: str) -> TestSet:
    path = TESTSETS_DIR / name / "testset.json"
    if not path.exists():
        raise KeyError(
            f"Unknown test-set {name!r}. Available: {available_testsets()}"
        )
    data = json.loads(path.read_text())
    meta = data.get("metadata", {})
    return TestSet(
        name=name,
        path=path,
        dataset=meta.get("dataset", name),
        queries=data["queries"],
        metadata=meta,
    )


def available_testsets() -> list[str]:
    if not TESTSETS_DIR.exists():
        return []
    return sorted(
        p.name for p in TESTSETS_DIR.iterdir()
        if (p / "testset.json").exists()
    )


# ── Parameter set ─────────────────────────────────────────────────────────

@dataclass
class ParameterSet:
    """Named tuning knobs for one system, from ``params/<system>/<name>.json``."""

    system: str
    name: str
    values: dict[str, Any] = field(default_factory=dict)


def load_params(system: str, paramset: str = "default") -> ParameterSet:
    path = PARAMS_DIR / system / f"{paramset}.json"
    if not path.exists():
        available = available_paramsets(system)
        raise KeyError(
            f"Unknown parameter set {system}:{paramset}. "
            f"Available for {system!r}: {available}"
        )
    return ParameterSet(
        system=system,
        name=paramset,
        values=json.loads(path.read_text()),
    )


def available_paramsets(system: str) -> list[str]:
    sys_dir = PARAMS_DIR / system
    if not sys_dir.exists():
        return []
    return sorted(p.stem for p in sys_dir.glob("*.json"))


# ── Experiment ────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    """One (system, parameter set) cell of a comparison."""

    system: str
    paramset: str = "default"

    @property
    def slug(self) -> str:
        """On-disk directory name, e.g. ``amem_default``."""
        return f"{self.system}_{self.paramset}"

    @classmethod
    def parse(cls, spec: str) -> "Experiment":
        """Parse a ``system[:paramset]`` CLI token (paramset defaults to ``default``)."""
        system, _, paramset = spec.partition(":")
        return cls(system=system.strip(), paramset=(paramset.strip() or "default"))


def comparison_dirname(dataset: str, testset: str) -> str:
    """Top-level results directory for one (dataset, test-set) pair."""
    return f"{dataset}_{testset}"
