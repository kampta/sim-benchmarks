#!/usr/bin/env python3
"""Prepare clean scored databases and an exact retry manifest after a VLABench run."""

from __future__ import annotations

import argparse
import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import Any

from sim_benchmarks.benchmarks.vlabench_xr1 import (
    OFFICIAL_TRACKS,
    EpisodeIdentity,
    load_completed_episode_identities,
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


def _entry(identity: EpisodeIdentity) -> dict[str, Any]:
    suite, task_name, episode_index, config_sha = identity
    return {
        "suite": suite,
        "task_name": task_name,
        "episode_index": episode_index,
        "episode_config_sha256": config_sha,
    }


def _official_identities(track_dir: Path) -> set[EpisodeIdentity]:
    identities: set[EpisodeIdentity] = set()
    for suite in OFFICIAL_TRACKS:
        path = track_dir / f"{suite}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for task in load_vlabench_episode_tasks(path, suite=suite, episode_limit=50):
            identity = task_episode_identity(task)
            if identity in identities:
                raise ValueError(f"duplicate identity in pinned tracks: {identity}")
            identities.add(identity)
    return identities


def _read_attempts(databases: list[Path]) -> tuple[dict[EpisodeIdentity, dict[str, Any]], dict[Path, list[tuple[str, str]]]]:
    attempts: dict[EpisodeIdentity, dict[str, Any]] = {}
    excluded_storage: dict[Path, list[tuple[str, str]]] = {}
    for database in databases:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if integrity != "ok":
                raise ValueError(f"{database}: SQLite quick_check failed: {integrity}")
            metadata = {
                str(row["eval_id"]): json.loads(row["metadata"])
                for row in connection.execute("SELECT eval_id, metadata FROM eval_metadata")
            }
            excluded_storage[database] = []
            for row in connection.execute(
                """SELECT sid, eid, eval_id, task_name, status, metrics, steps, context,
                          failure_reason, failure_detail
                   FROM episode_results"""
            ):
                eval_id = str(row["eval_id"])
                if eval_id not in metadata:
                    raise ValueError(f"{database}: result references missing eval metadata {eval_id!r}")
                suite = metadata[eval_id].get("config", {}).get("params", {}).get("eval_track")
                context = json.loads(row["context"])
                identity = task_episode_identity(
                    {
                        "suite": suite,
                        "name": context.get("name"),
                        "episode_index": context.get("episode_index"),
                        "episode_config_sha256": context.get("episode_config_sha256"),
                    }
                )
                if context.get("name") != row["task_name"]:
                    raise ValueError(f"{database}: task name conflicts with recorded context for {identity}")
                if identity in attempts:
                    raise ValueError(f"duplicate attempted episode identity: {identity}")
                failed = bool(
                    row["status"] == "error" or row["failure_reason"] is not None or row["failure_detail"] is not None
                )
                attempts[identity] = {
                    "identity": _entry(identity),
                    "database": str(database),
                    "sid": str(row["sid"]),
                    "eid": str(row["eid"]),
                    "status": str(row["status"]),
                    "steps": int(row["steps"]),
                    "failure_reason": row["failure_reason"],
                    "failure_detail": row["failure_detail"],
                    "failed": failed,
                }
                if failed:
                    excluded_storage[database].append((str(row["sid"]), str(row["eid"])))
        finally:
            connection.close()
    return attempts, excluded_storage


def _clean_database(source: Path, destination: Path, excluded: list[tuple[str, str]]) -> None:
    if destination.exists():
        destination.unlink()
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        source_connection.close()
        destination_connection.close()

    connection = sqlite3.connect(destination)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for sid, eid in excluded:
            connection.execute("DELETE FROM step_rows WHERE sid = ? AND eid = ?", (sid, eid))
            connection.execute("DELETE FROM episode_results WHERE sid = ? AND eid = ?", (sid, eid))
        connection.commit()
        integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"{destination}: SQLite quick_check failed after cleaning: {integrity}")
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", nargs="+", required=True, type=Path)
    parser.add_argument("--base-completed-manifest", required=True, type=Path)
    parser.add_argument("--track-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    databases = [path.resolve() for path in args.db]
    for database in databases:
        if not database.is_file():
            raise FileNotFoundError(database)
    expected = _official_identities(args.track_dir.resolve())
    base_completed = load_completed_episode_identities(args.base_completed_manifest)
    unexpected_base = sorted(base_completed - expected)
    if unexpected_base:
        raise ValueError(f"base manifest contains identities absent from pinned tracks: {unexpected_base[:3]}")
    expected_attempts = expected - base_completed

    attempts, excluded_storage = _read_attempts(databases)
    observed = set(attempts)
    missing = sorted(expected_attempts - observed)
    unexpected = sorted(observed - expected_attempts)
    if missing or unexpected:
        raise ValueError(
            "run does not exactly cover the remaining pinned identities: "
            f"missing={len(missing)} first={missing[:3]}; unexpected={len(unexpected)} first={unexpected[:3]}"
        )
    failures = {identity for identity, attempt in attempts.items() if attempt["failed"]}

    output_dir = args.output_dir.resolve()
    clean_dir = output_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_databases: list[Path] = []
    sources: list[dict[str, Any]] = []
    for index, database in enumerate(databases):
        clean = clean_dir / f"shard{index}.sqlite"
        _clean_database(database, clean, excluded_storage[database])
        clean_databases.append(clean)
        sources.append(
            {
                "source": str(database),
                "source_sha256": _file_sha256(database),
                "clean": str(clean),
                "clean_sha256": _file_sha256(clean),
                "excluded_attempts": len(excluded_storage[database]),
            }
        )

    clean_results, clean_validation = load_recording_databases(clean_databases)
    valid_identities: set[EpisodeIdentity] = set()
    for result in clean_results:
        suite = result["config"]["params"]["eval_track"]
        for task in result["tasks"]:
            for episode in task["episodes"]:
                identity = task_episode_identity(
                    {
                        "suite": suite,
                        "name": task["task"],
                        "episode_index": episode["episode_index"],
                        "episode_config_sha256": episode["episode_config_sha256"],
                    }
                )
                if identity in valid_identities:
                    raise ValueError(f"duplicate valid identity after cleaning: {identity}")
                valid_identities.add(identity)
    expected_valid = expected_attempts - failures
    if valid_identities != expected_valid:
        raise ValueError("clean recording identities do not equal attempted identities minus failures")

    retry_manifest = {
        "schema_version": 1,
        "completed_episode_count": len(expected - failures),
        "completed_episodes": [_entry(identity) for identity in sorted(expected - failures)],
    }
    failure_records = [attempts[identity] for identity in sorted(failures)]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retry-manifest.json").write_text(json.dumps(retry_manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "failed-attempts.json").write_text(json.dumps(failure_records, indent=2) + "\n", encoding="utf-8")
    preparation = {
        "official_identities": len(expected),
        "base_completed_identities": len(base_completed),
        "attempted_identities": len(attempts),
        "valid_identities": len(valid_identities),
        "retry_identities": len(failures),
        "sources": sources,
        "clean_recording_validation": clean_validation,
    }
    (output_dir / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(preparation, indent=2))


if __name__ == "__main__":
    main()
