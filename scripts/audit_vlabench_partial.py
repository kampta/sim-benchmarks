#!/usr/bin/env python3
"""Validate partial XR-1 VLABench recording databases without accepting errors as scores."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim_benchmarks.benchmarks.vlabench_xr1 import (
    OFFICIAL_TRACKS,
    EpisodeIdentity,
    load_completed_episode_identities,
    load_vlabench_episode_tasks,
    task_episode_identity,
)
from sim_benchmarks.reporting.vlabench import METRICS


def _official_identities(track_dir: Path) -> set[EpisodeIdentity]:
    expected: set[EpisodeIdentity] = set()
    for suite in OFFICIAL_TRACKS:
        path = track_dir / f"{suite}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for task in load_vlabench_episode_tasks(path, suite=suite, episode_limit=50):
            identity = task_episode_identity(task)
            if identity in expected:
                raise ValueError(f"duplicate identity in pinned tracks: {identity}")
            expected.add(identity)
    return expected


def _metrics(value: str, source: str) -> dict[str, float]:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise TypeError(f"{source}: metrics must be an object")
    parsed: dict[str, float] = {}
    for key in METRICS:
        if key not in raw:
            raise ValueError(f"{source}: missing metric {key!r}")
        score = float(raw[key])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{source}: invalid {key}={score!r}")
        parsed[key] = score
    return parsed


def audit_partial(
    databases: list[Path],
    *,
    track_dir: Path,
    base_completed_manifest: Path,
) -> dict[str, Any]:
    expected = _official_identities(track_dir.resolve())
    base_completed = load_completed_episode_identities(base_completed_manifest)
    if not base_completed <= expected:
        raise ValueError("base completed manifest contains identities absent from pinned tracks")
    expected_remaining = expected - base_completed

    observed: set[EpisodeIdentity] = set()
    valid: set[EpisodeIdentity] = set()
    errors: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    track_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    success_count = 0
    failure_count = 0
    valid_steps = 0
    valid_elapsed_sec = 0.0

    for database in databases:
        database = database.resolve()
        if not database.is_file():
            raise FileNotFoundError(database)
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
            step_stats = {
                (str(row["sid"]), str(row["eid"])): (
                    int(row["count"]),
                    int(row["first_step"]),
                    int(row["last_step"]),
                )
                for row in connection.execute(
                    """SELECT sid, eid, COUNT(*) AS count, MIN(step_id) AS first_step,
                              MAX(step_id) AS last_step
                       FROM step_rows GROUP BY sid, eid"""
                )
            }
            database_results = 0
            for row in connection.execute(
                """SELECT sid, eid, eval_id, task_name, status, metrics, steps, elapsed_sec,
                          context, failure_reason, failure_detail
                   FROM episode_results"""
            ):
                database_results += 1
                storage_id = (str(row["sid"]), str(row["eid"]))
                stored_steps = step_stats.pop(storage_id, None)
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
                source = f"{database}:{storage_id[0]}/{storage_id[1]}"
                if context.get("name") != row["task_name"] or context.get("suite") != suite:
                    raise ValueError(f"{source}: stored context conflicts with task or track")
                if identity not in expected_remaining:
                    raise ValueError(f"{source}: identity is not an expected remaining episode: {identity}")
                if identity in observed:
                    raise ValueError(f"duplicate attempted identity across databases: {identity}")
                observed.add(identity)

                is_error = bool(
                    row["status"] == "error" or row["failure_reason"] is not None or row["failure_detail"] is not None
                )
                if is_error:
                    if row["status"] != "error" or not row["failure_reason"]:
                        raise ValueError(f"{source}: malformed error result")
                    if stored_steps is not None and stored_steps != (stored_steps[0], 0, stored_steps[0] - 1):
                        raise ValueError(f"{source}: error attempt has non-contiguous stored steps {stored_steps}")
                    errors.append(
                        {
                            "suite": identity[0],
                            "task_name": identity[1],
                            "episode_index": identity[2],
                            "episode_config_sha256": identity[3],
                            "failure_reason": row["failure_reason"],
                            "stored_step_rows": 0 if stored_steps is None else stored_steps[0],
                            "source": source,
                        }
                    )
                    track_counts[suite]["errors"] += 1
                    continue

                scores = _metrics(str(row["metrics"]), source)
                expected_status = "success" if bool(scores["success"]) else "fail"
                if row["status"] != expected_status:
                    raise ValueError(f"{source}: status conflicts with success metric")
                steps = int(row["steps"])
                if steps < 1 or stored_steps != (steps, 0, steps - 1):
                    raise ValueError(f"{source}: expected contiguous step rows {(steps, 0, steps - 1)}, found {stored_steps}")
                valid.add(identity)
                success_count += int(expected_status == "success")
                failure_count += int(expected_status == "fail")
                valid_steps += steps
                valid_elapsed_sec += float(row["elapsed_sec"])
                track_counts[suite][expected_status] += 1
            if step_stats:
                raise ValueError(f"{database}: found step rows without an episode result")
            sources.append(
                {
                    "path": str(database),
                    "sqlite_quick_check": integrity,
                    "episode_results": database_results,
                    "step_rows": connection.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0],
                }
            )
        finally:
            connection.close()

    return {
        "status": "valid_partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_identities": len(expected),
        "base_completed_identities": len(base_completed),
        "expected_remaining_identities": len(expected_remaining),
        "attempted_identities": len(observed),
        "valid_identities": len(valid),
        "total_valid_identities": len(base_completed) + len(valid),
        "error_identities": len(errors),
        "unattempted_identities": len(expected_remaining - observed),
        "pending_valid_identities": len(expected - base_completed - valid),
        "suite_completion_fraction": (len(base_completed) + len(valid)) / len(expected),
        "successes": success_count,
        "failures": failure_count,
        "valid_steps": valid_steps,
        "valid_elapsed_sec": valid_elapsed_sec,
        "tracks": {suite: dict(track_counts[suite]) for suite in OFFICIAL_TRACKS},
        "errors": errors,
        "recording_validation": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", nargs="+", required=True, type=Path)
    parser.add_argument("--track-dir", required=True, type=Path)
    parser.add_argument("--base-completed-manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_partial(
        args.db,
        track_dir=args.track_dir,
        base_completed_manifest=args.base_completed_manifest,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".next")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
