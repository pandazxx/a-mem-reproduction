"""
Dataset registry. Add a new dataset by:
  1. Creating ``experiments/datasets/<name>/dataset.json`` (memories + questions)
  2. Adding an entry to ``DATASETS`` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent

DATASETS: dict[str, Path] = {
    "comparison": _HERE / "comparison" / "dataset.json",
}


@dataclass
class Dataset:
    name: str
    path: Path
    memories: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    metadata: dict[str, Any]


def load(name: str) -> Dataset:
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}"
        )
    path = DATASETS[name]
    data = json.loads(path.read_text())
    return Dataset(
        name=name,
        path=path,
        memories=data["memories"],
        questions=data["questions"],
        metadata=data.get("metadata", {}),
    )
