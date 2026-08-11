"""Load pinned policy-evaluation and benchmark-provenance manifests."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def _manifest_root() -> Path:
    return Path(str(files("sim_benchmarks").joinpath("manifests")))


def load_json(relative_path: str) -> dict[str, Any]:
    path = _manifest_root() / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def benchmark_manifests() -> list[dict[str, Any]]:
    root = _manifest_root() / "benchmarks"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def model_manifests() -> list[dict[str, Any]]:
    root = _manifest_root() / "models"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def dataset_catalog() -> dict[str, Any]:
    return load_json("datasets/catalog.json")


def reproduction_targets() -> dict[str, Any]:
    return load_json("reproduction/targets.json")
