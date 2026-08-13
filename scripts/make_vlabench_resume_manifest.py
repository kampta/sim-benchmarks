#!/usr/bin/env python3
"""Build an exact, validated VLABench resume manifest from recording databases."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from sim_benchmarks.benchmarks.vlabench_xr1 import (
    OFFICIAL_TRACKS,
    EpisodeIdentity,
    load_vlabench_episode_tasks,
    task_episode_identity,
)
from sim_benchmarks.reporting.vlabench import load_recording_databases


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_identities(results: list[dict[str, Any]]) -> set[EpisodeIdentity]:
    identities: set[EpisodeIdentity] = set()
    for result in results:
        suite = result.get("config", {}).get("params", {}).get("eval_track")
        for task in result.get("tasks", []):
            task_name = task.get("task")
            for episode in task.get("episodes", []):
                identity = task_episode_identity(
                    {
                        "suite": suite,
                        "name": task_name,
                        "episode_index": episode.get("episode_index"),
                        "episode_config_sha256": episode.get("episode_config_sha256"),
                    }
                )
                if identity in identities:
                    raise ValueError(f"duplicate completed episode identity across recordings: {identity}")
                identities.add(identity)
    return identities


def _official_identities(track_dir: Path) -> tuple[set[EpisodeIdentity], dict[str, str]]:
    identities: set[EpisodeIdentity] = set()
    track_hashes: dict[str, str] = {}
    for suite in OFFICIAL_TRACKS:
        path = track_dir / f"{suite}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        track_hashes[suite] = _file_sha256(path)
        for task in load_vlabench_episode_tasks(path, suite=suite, episode_limit=50):
            identity = task_episode_identity(task)
            if identity in identities:
                raise ValueError(f"duplicate identity in official track files: {identity}")
            identities.add(identity)
    return identities, track_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", nargs="+", required=True, type=Path)
    parser.add_argument("--track-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    databases = [path.resolve() for path in args.db]
    results, database_validation = load_recording_databases(databases)
    observed = _recorded_identities(results)
    expected, track_hashes = _official_identities(args.track_dir.resolve())
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(
            f"recordings contain {len(unexpected)} identities absent from the pinned tracks: {unexpected[:3]}"
        )

    entries = [
        {
            "suite": suite,
            "task_name": task_name,
            "episode_index": episode_index,
            "episode_config_sha256": config_sha,
        }
        for suite, task_name, episode_index, config_sha in sorted(observed)
    ]
    manifest = {
        "schema_version": 1,
        "completed_episode_count": len(entries),
        "completed_episodes": entries,
        "track_file_sha256": track_hashes,
        "recording_sources": [
            {
                **validation,
                "sha256": _file_sha256(Path(validation["path"])),
            }
            for validation in database_validation
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} validated episode identities to {args.output}")


if __name__ == "__main__":
    main()
