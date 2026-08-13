#!/usr/bin/env python3
"""Freeze validated VLABench progress and emit an exact continuation manifest."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
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


def _official_identities(track_dir: Path) -> tuple[set[EpisodeIdentity], dict[str, str]]:
    identities: set[EpisodeIdentity] = set()
    hashes: dict[str, str] = {}
    for suite in OFFICIAL_TRACKS:
        path = track_dir / f"{suite}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[suite] = _file_sha256(path)
        for task in load_vlabench_episode_tasks(path, suite=suite, episode_limit=50):
            identity = task_episode_identity(task)
            if identity in identities:
                raise ValueError(f"duplicate identity in pinned tracks: {identity}")
            identities.add(identity)
    return identities, hashes


def _snapshot_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        source_connection.close()
        destination_connection.close()


def _read_attempts(
    database: Path,
) -> tuple[list[tuple[EpisodeIdentity, tuple[str, str], bool, dict[str, Any]]], str]:
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
        attempts = []
        for row in connection.execute(
            """SELECT sid, eid, eval_id, task_name, status, steps, context,
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
            if context.get("name") != row["task_name"] or context.get("suite") != suite:
                raise ValueError(f"{database}: task context conflicts with metadata for {identity}")
            failed = bool(
                row["status"] == "error" or row["failure_reason"] is not None or row["failure_detail"] is not None
            )
            attempts.append(
                (
                    identity,
                    (str(row["sid"]), str(row["eid"])),
                    failed,
                    {
                        "identity": _entry(identity),
                        "source": str(database),
                        "sid": str(row["sid"]),
                        "eid": str(row["eid"]),
                        "status": str(row["status"]),
                        "steps": int(row["steps"]),
                        "failure_reason": row["failure_reason"],
                        "failure_detail": row["failure_detail"],
                    },
                )
            )
        return attempts, integrity
    finally:
        connection.close()


def _remove_failed_attempts(database: Path, storage_ids: list[tuple[str, str]]) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for sid, eid in storage_ids:
            connection.execute("DELETE FROM step_rows WHERE sid = ? AND eid = ?", (sid, eid))
            connection.execute("DELETE FROM episode_results WHERE sid = ? AND eid = ?", (sid, eid))
        connection.commit()
        integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if integrity != "ok":
            raise ValueError(f"{database}: SQLite quick_check failed after deriving clean copy: {integrity}")
    finally:
        connection.close()


def _identities_from_results(results: list[dict[str, Any]]) -> set[EpisodeIdentity]:
    identities: set[EpisodeIdentity] = set()
    for result in results:
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
                if identity in identities:
                    raise ValueError(f"duplicate valid identity across clean snapshots: {identity}")
                identities.add(identity)
    return identities


def checkpoint_progress(
    databases: list[Path],
    *,
    track_dir: Path,
    base_completed_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Snapshot live DBs without mutation and produce a validated resume point."""

    if not databases:
        raise ValueError("at least one recording database is required")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite recovery checkpoint: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    clean_dir = output_dir / "clean"
    raw_dir.mkdir()
    clean_dir.mkdir()

    expected, track_hashes = _official_identities(track_dir.resolve())
    base_completed = load_completed_episode_identities(base_completed_manifest)
    unexpected_base = sorted(base_completed - expected)
    if unexpected_base:
        raise ValueError(f"base manifest contains identities absent from pinned tracks: {unexpected_base[:3]}")

    failure_records: list[dict[str, Any]] = []
    attempted: Counter[EpisodeIdentity] = Counter()
    sources: list[dict[str, Any]] = []
    clean_databases: list[Path] = []
    for index, source in enumerate(databases):
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        raw = raw_dir / f"shard{index}.sqlite"
        clean = clean_dir / f"shard{index}.sqlite"
        _snapshot_database(source, raw)
        _snapshot_database(raw, clean)
        attempts, integrity = _read_attempts(raw)
        failed_storage: list[tuple[str, str]] = []
        for identity, storage_id, failed, record in attempts:
            if identity not in expected:
                raise ValueError(f"recording contains identity absent from pinned tracks: {identity}")
            if identity in base_completed:
                raise ValueError(f"recording re-attempted an identity in the base completed manifest: {identity}")
            attempted[identity] += 1
            if failed:
                failed_storage.append(storage_id)
                failure_records.append(record)
        _remove_failed_attempts(clean, failed_storage)
        clean_databases.append(clean)
        sources.append(
            {
                "source": str(source),
                "source_sha256_at_checkpoint": _file_sha256(raw),
                "raw_snapshot": str(raw),
                "raw_snapshot_sha256": _file_sha256(raw),
                "clean_snapshot": str(clean),
                "clean_snapshot_sha256": _file_sha256(clean),
                "sqlite_quick_check": integrity,
                "attempts": len(attempts),
                "excluded_failures": len(failed_storage),
            }
        )

    valid_results, clean_validation = load_recording_databases(clean_databases)
    valid = _identities_from_results(valid_results)
    completed = base_completed | valid
    pending = expected - completed

    manifest = {
        "schema_version": 1,
        "completed_episode_count": len(completed),
        "completed_episodes": [_entry(identity) for identity in sorted(completed)],
        "track_file_sha256": track_hashes,
    }
    (output_dir / "completed-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "failed-attempts.json").write_text(
        json.dumps(failure_records, indent=2) + "\n", encoding="utf-8"
    )
    progress = {
        "status": "complete" if not pending else "valid_partial",
        "official_identities": len(expected),
        "base_completed_identities": len(base_completed),
        "valid_checkpoint_identities": len(valid),
        "completed_identities": len(completed),
        "pending_identities": len(pending),
        "attempted_identity_count": len(attempted),
        "attempt_rows": sum(attempted.values()),
        "failed_attempt_rows": len(failure_records),
        "sources": sources,
        "clean_recording_validation": clean_validation,
    }
    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{_file_sha256(path)}  {path.relative_to(output_dir)}\n"
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        encoding="utf-8",
    )
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", nargs="+", required=True, type=Path)
    parser.add_argument("--track-dir", required=True, type=Path)
    parser.add_argument("--base-completed-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    progress = checkpoint_progress(
        args.db,
        track_dir=args.track_dir,
        base_completed_manifest=args.base_completed_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
