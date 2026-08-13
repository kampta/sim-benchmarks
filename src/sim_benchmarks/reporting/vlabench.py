"""Official task-macro aggregation for XR-1's VLABench evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sqlite3
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from sim_benchmarks.benchmarks.vlabench_xr1 import (
    OFFICIAL_TRACKS,
    EpisodeIdentity,
    load_vlabench_episode_tasks,
    task_episode_identity,
)

METRICS = ("success", "intention_score", "progress_score")
REPORTED_SOURCE = "https://arxiv.org/html/2607.15330v2#S5.T4"
REPORTED_PROTOCOL_EPISODES = 2500
XR1_REPORTED_VLABENCH: dict[str, dict[str, float]] = {
    "track_1_in_distribution": {"success": 0.756, "intention_score": 0.798, "progress_score": 0.850},
    "track_2_cross_category": {"success": 0.530, "intention_score": 0.664, "progress_score": 0.666},
    "track_3_common_sense": {"success": 0.484, "intention_score": 0.582, "progress_score": 0.583},
    "track_4_semantic_instruction": {"success": 0.558, "intention_score": 0.702, "progress_score": 0.668},
    "track_6_unseen_texture": {"success": 0.626, "intention_score": 0.748, "progress_score": 0.749},
    "overall": {"success": 0.591, "intention_score": 0.699, "progress_score": 0.703},
}
_STANDARD_TASKS = (
    "select_painting",
    "select_book",
    "select_drink",
    "select_chemistry_tube",
    "select_poker",
    "select_mahjong",
    "select_toy",
    "select_fruit",
    "add_condiment",
    "insert_flower",
)
OFFICIAL_EXPECTED_EPISODES: dict[str, dict[str, int]] = {
    "track_1_in_distribution": {task: 50 for task in _STANDARD_TASKS},
    "track_2_cross_category": {
        **{task: 50 for task in _STANDARD_TASKS},
        "insert_flower": 10,
    },
    "track_3_common_sense": {
        **{task: 50 for task in _STANDARD_TASKS if task not in {"select_poker", "select_mahjong"}},
        "select_nth_largest_poker": 50,
        "select_unique_type_mahjong": 50,
    },
    "track_4_semantic_instruction": {task: 50 for task in _STANDARD_TASKS},
    "track_6_unseen_texture": {task: 50 for task in _STANDARD_TASKS},
}
PINNED_PROTOCOL_EPISODES = sum(sum(tasks.values()) for tasks in OFFICIAL_EXPECTED_EPISODES.values())
UPSTREAM_RUNTIME_ERROR_PROTOCOL = (
    "Xiaomi's pinned VLABench Evaluator catches unstable-episode exceptions, omits those episodes from "
    "task_infos, and computes each task mean over the remaining records."
)


def load_runtime_error_records(
    paths: list[Path], *, validate_databases: bool = False
) -> list[dict[str, Any]]:
    """Load unresolved simulator errors, rejecting malformed or duplicate identities."""

    errors: list[dict[str, Any]] = []
    observed: set[EpisodeIdentity] = set()
    for path in paths:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, list):
            raise TypeError(f"{path}: runtime error records must be a list")
        for index, record in enumerate(decoded):
            if not isinstance(record, dict) or not isinstance(record.get("identity"), dict):
                raise TypeError(f"{path}: runtime error record {index} is malformed")
            identity_record = record["identity"]
            identity = task_episode_identity(
                {
                    "suite": identity_record.get("suite"),
                    "name": identity_record.get("task_name"),
                    "episode_index": identity_record.get("episode_index"),
                    "episode_config_sha256": identity_record.get("episode_config_sha256"),
                }
            )
            if identity in observed:
                raise ValueError(f"duplicate unresolved runtime-error identity: {identity}")
            if record.get("status") != "error" or not record.get("failure_reason"):
                raise ValueError(f"{path}: unresolved record is not a well-formed runtime error: {identity}")
            if validate_databases:
                database_value = record.get("database") or record.get("source")
                if not isinstance(database_value, str):
                    raise ValueError(f"{path}: runtime error lacks its preserved database: {identity}")
                database = Path(database_value).resolve()
                if not database.is_file():
                    raise FileNotFoundError(database)
                connection = sqlite3.connect(database)
                connection.row_factory = sqlite3.Row
                try:
                    integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
                    if integrity != "ok":
                        raise ValueError(f"{database}: SQLite quick_check failed: {integrity}")
                    row = connection.execute(
                        """SELECT status, failure_reason FROM episode_results
                           WHERE sid = ? AND eid = ?""",
                        (record.get("sid"), record.get("eid")),
                    ).fetchone()
                    if row is None or row["status"] != "error" or not row["failure_reason"]:
                        raise ValueError(f"{database}: unresolved error row is missing: {identity}")
                    step_stats = connection.execute(
                        """SELECT COUNT(*), MIN(step_id), MAX(step_id) FROM step_rows
                           WHERE sid = ? AND eid = ?""",
                        (record.get("sid"), record.get("eid")),
                    ).fetchone()
                    step_count = int(step_stats[0])
                    if step_count and (int(step_stats[1]), int(step_stats[2])) != (0, step_count - 1):
                        raise ValueError(f"{database}: runtime-error step rows are non-contiguous: {identity}")
                finally:
                    connection.close()
                record = dict(record)
                record["preserved_database"] = str(database)
                record["preserved_database_sha256"] = sha256(database.read_bytes()).hexdigest()
                record["sqlite_quick_check"] = integrity
                record["stored_step_rows"] = step_count
            observed.add(identity)
            errors.append(record)
    return errors


def _metrics_from_json(value: str, *, source: str) -> dict[str, float]:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise TypeError(f"{source}: metrics must decode to an object")
    metrics: dict[str, float] = {}
    for metric in METRICS:
        if metric not in raw:
            raise ValueError(f"{source}: missing metric {metric!r}")
        score = float(raw[metric])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{source}: invalid {metric} score {score!r}")
        metrics[metric] = score
    return metrics


def load_recording_databases(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load sharded v0.4.0 recording databases and validate every stored rollout."""

    if not paths:
        raise ValueError("at least one recording database is required")
    results: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    storage_ids: set[tuple[str, str]] = set()

    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if integrity != "ok":
                raise ValueError(f"{path}: SQLite quick_check failed: {integrity}")

            metadata: dict[str, dict[str, Any]] = {}
            for row in connection.execute("SELECT eval_id, metadata FROM eval_metadata"):
                decoded = json.loads(row["metadata"])
                if not isinstance(decoded, dict):
                    raise TypeError(f"{path}: metadata for {row['eval_id']} is not an object")
                metadata[str(row["eval_id"])] = decoded

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
            by_eval: dict[str, dict[str, list[dict[str, Any]]]] = {}
            episode_count = 0
            for row in connection.execute(
                """SELECT sid, eid, eval_id, task_name, episode_id, status, metrics,
                          steps, elapsed_sec, context, failure_reason, failure_detail
                   FROM episode_results"""
            ):
                episode_count += 1
                sid_eid = (str(row["sid"]), str(row["eid"]))
                if sid_eid in storage_ids:
                    raise ValueError(f"duplicate recording identity across databases: {sid_eid}")
                storage_ids.add(sid_eid)

                eval_id = str(row["eval_id"])
                if eval_id not in metadata:
                    raise ValueError(f"{path}: episode references missing eval metadata {eval_id!r}")
                config = metadata[eval_id].get("config", {})
                track = config.get("params", {}).get("eval_track")
                if track not in OFFICIAL_TRACKS:
                    raise ValueError(f"{path}: episode has invalid official track {track!r}")

                source = f"{path}:{sid_eid[0]}/{sid_eid[1]}"
                if row["failure_reason"] or row["failure_detail"] or row["status"] == "error":
                    raise ValueError(f"{source}: rollout contains a runtime failure")
                metrics = _metrics_from_json(str(row["metrics"]), source=source)
                expected_status = "success" if bool(metrics["success"]) else "fail"
                if row["status"] != expected_status:
                    raise ValueError(f"{source}: status {row['status']!r} conflicts with success metric")

                context = json.loads(row["context"])
                if not isinstance(context, dict):
                    raise TypeError(f"{source}: episode context is not an object")
                task_name = str(row["task_name"])
                if context.get("name") != task_name or context.get("suite") != track:
                    raise ValueError(f"{source}: stored task/track context is inconsistent")
                episode_index = context.get("episode_index")
                config_sha = context.get("episode_config_sha256")
                if not isinstance(episode_index, int) or not isinstance(config_sha, str):
                    raise TypeError(f"{source}: missing pinned episode identity")

                steps = int(row["steps"])
                stored_steps = step_stats.pop(sid_eid, None)
                expected_steps = (steps, 0, steps - 1)
                if steps < 1 or stored_steps != expected_steps:
                    raise ValueError(f"{source}: expected contiguous step rows {expected_steps}, found {stored_steps}")

                by_eval.setdefault(eval_id, {}).setdefault(task_name, []).append(
                    {
                        "episode_id": int(row["episode_id"]),
                        "episode_index": episode_index,
                        "episode_config_sha256": config_sha,
                        "metrics": metrics,
                        "status": str(row["status"]),
                        "steps": steps,
                        "elapsed_sec": float(row["elapsed_sec"]),
                        "recording_identity": {"sid": sid_eid[0], "eid": sid_eid[1]},
                    }
                )

            if step_stats:
                raise ValueError(f"{path}: found step rows without an episode result")
            for eval_id, tasks in by_eval.items():
                results.append(
                    {
                        "config": metadata[eval_id]["config"],
                        "tasks": [
                            {"task": task_name, "episodes": episodes} for task_name, episodes in sorted(tasks.items())
                        ],
                    }
                )
            validation.append(
                {
                    "path": str(path),
                    "sqlite_quick_check": integrity,
                    "episode_results": episode_count,
                    "step_rows": connection.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0],
                }
            )
        finally:
            connection.close()
    return results, validation


def _episode_key(track: str, task: str, episode: dict[str, Any]) -> tuple[Any, ...]:
    """Return the stable identity written by the XR-1 VLABench adapter."""

    return (
        track,
        task,
        episode.get("episode_index", episode.get("episode_idx", episode.get("episode_id"))),
        episode.get("episode_config_sha256"),
    )


def aggregate_official_vlabench(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool shard episodes, then reproduce Xiaomi's task/track macro averages."""

    pooled: dict[str, dict[str, dict[str, Any]]] = {}
    for result_index, result in enumerate(results):
        params = result.get("config", {}).get("params", {})
        track = params.get("eval_track")
        if track not in OFFICIAL_TRACKS:
            raise ValueError(f"aggregate is missing an official eval_track: {track!r}")
        track_tasks = pooled.setdefault(track, {})
        for task_index, task in enumerate(result.get("tasks", [])):
            name = task.get("task")
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid VLABench task entry: {name!r}")
            accumulator = track_tasks.setdefault(
                name,
                {
                    "episodes": {},
                    "summary_sums": {metric: 0.0 for metric in METRICS},
                    "summary_count": 0,
                },
            )
            episodes = task.get("episodes", [])
            if episodes:
                for episode_index, episode in enumerate(episodes):
                    key = _episode_key(track, name, episode)
                    if key[2:] == (None, None):
                        key += (result_index, task_index, episode_index)
                    scores = {metric: float(episode["metrics"][metric]) for metric in METRICS}
                    previous = accumulator["episodes"].setdefault(key, scores)
                    if previous != scores:
                        raise ValueError(f"conflicting duplicate VLABench episode: {key}")
            else:
                # Preserve support for aggregate-only inputs while preferring the
                # episode records emitted by sharded full-suite evaluations.
                count = int(task.get("num_episodes", 1))
                if count < 1:
                    raise ValueError(f"VLABench task {name} has invalid episode count: {count}")
                for metric in METRICS:
                    accumulator["summary_sums"][metric] += float(task[f"mean_{metric}"]) * count
                accumulator["summary_count"] += count

    tracks: dict[str, Any] = {}
    for track, track_tasks in pooled.items():
        task_scores: dict[str, dict[str, float | int]] = {}
        for name, accumulator in track_tasks.items():
            episode_scores = list(accumulator["episodes"].values())
            count = len(episode_scores) + accumulator["summary_count"]
            if not count:
                raise ValueError(f"VLABench task {name} contains no episode results")
            task_scores[name] = {
                metric: (sum(scores[metric] for scores in episode_scores) + accumulator["summary_sums"][metric])
                / count
                for metric in METRICS
            }
            task_scores[name]["num_episodes"] = count
        tracks[track] = {
            "tasks": task_scores,
            "macro": {
                metric: sum(scores[metric] for scores in task_scores.values()) / len(task_scores) for metric in METRICS
            },
            "num_tasks": len(task_scores),
            "num_episodes": sum(int(scores["num_episodes"]) for scores in task_scores.values()),
        }

    missing = [track for track in OFFICIAL_TRACKS if track not in tracks]
    if missing:
        raise ValueError(f"missing official VLABench track aggregates: {missing}")
    return {
        "aggregation": "macro_average_across_task_entries",
        "tracks": {track: tracks[track] for track in OFFICIAL_TRACKS},
        "overall": {
            metric: sum(
                float(task_scores[metric])
                for track in OFFICIAL_TRACKS
                for task_scores in tracks[track]["tasks"].values()
            )
            / sum(tracks[track]["num_tasks"] for track in OFFICIAL_TRACKS)
            for metric in METRICS
        },
        "overall_aggregation": "macro_average_across_all_task_entries",
        "num_episodes_total": sum(tracks[track]["num_episodes"] for track in OFFICIAL_TRACKS),
    }


def compare_to_reported(report: dict[str, Any]) -> dict[str, Any]:
    """Attach Xiaomi's Table 4 values and measured-minus-reported deltas."""

    measured = {
        **{track: report["tracks"][track]["macro"] for track in OFFICIAL_TRACKS},
        "overall": report["overall"],
    }
    return {
        "source": REPORTED_SOURCE,
        "units": "fraction; delta_percentage_points is 100 * (measured - reported)",
        "tracks": {
            key: {
                "measured": {metric: float(measured[key][metric]) for metric in METRICS},
                "reported": XR1_REPORTED_VLABENCH[key],
                "delta_percentage_points": {
                    metric: 100.0 * (float(measured[key][metric]) - XR1_REPORTED_VLABENCH[key][metric])
                    for metric in METRICS
                },
            }
            for key in (*OFFICIAL_TRACKS, "overall")
        },
    }


def compare_protocol_cardinality(report: dict[str, Any]) -> dict[str, Any]:
    """Record the released-track cardinality difference from the paper text."""

    measured_episodes = int(report.get("num_attempted_episodes", report["num_episodes_total"]))
    if measured_episodes != PINNED_PROTOCOL_EPISODES:
        raise ValueError(
            f"protocol comparison requires {PINNED_PROTOCOL_EPISODES} pinned episodes, found {measured_episodes}"
        )
    cross_category_flower = int(
        report["tracks"]["track_2_cross_category"]["tasks"].get("insert_flower", {}).get("num_episodes", 0)
    ) + sum(
        record["identity"]["suite"] == "track_2_cross_category"
        and record["identity"]["task_name"] == "insert_flower"
        for record in report.get("runtime_errors", [])
    )
    if cross_category_flower != 10:
        raise ValueError(
            "protocol comparison requires the released Cross-category insert_flower cardinality of 10, "
            f"found {cross_category_flower}"
        )
    return {
        "reported_source": REPORTED_SOURCE,
        "paper_described_episodes": REPORTED_PROTOCOL_EPISODES,
        "pinned_released_episodes": PINNED_PROTOCOL_EPISODES,
        "difference": PINNED_PROTOCOL_EPISODES - REPORTED_PROTOCOL_EPISODES,
        "reason": (
            "The paper describes 50 episodes for every task entry (2,500 total), but the pinned released "
            "track_2_cross_category file contains only 10 insert_flower configurations rather than 50."
        ),
        "comparison_scope": (
            "Every released pinned identity is attempted. Available-case metrics follow Xiaomi's released "
            "handling of unstable simulator exceptions; reported metrics are reproduced from Table 4 and may "
            "reflect the paper-described 2,500-rollout protocol."
        ),
    }
def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Render measured and reported full-suite/per-track metrics side by side."""

    comparison = report["reported_comparison"]
    metric_columns = (
        ("success", "SR"),
        ("progress_score", "PS"),
        ("intention_score", "IS"),
    )
    header = ["Scope", "Episodes"]
    for _, label in metric_columns:
        header.extend((f"Measured {label}", f"Reported {label}", f"Δ{label} (pp)"))
    lines = [
        "# Xiaomi-Robotics-1 VLABench reproduction",
        "",
        f"Reported reference: <{comparison['source']}>",
        "",
        (
            "Aggregation: macro average across task entries. The pinned five-track files contain "
            f"{report.get('num_attempted_episodes', report['num_episodes_total']):,} attempted episodes."
        ),
        (
            f"Protocol cardinality note: the paper describes {report['protocol_comparison']['paper_described_episodes']:,} "
            f"rollouts, while the pinned released tracks contain "
            f"{report['protocol_comparison']['pinned_released_episodes']:,}. "
            f"{report['protocol_comparison']['reason']}"
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| :--- | ---: | " + " | ".join("---:" for _ in range(len(header) - 2)) + " |",
    ]
    excluded = len(report.get("runtime_errors", []))
    if excluded:
        lines[4] += (
            f" Xiaomi-compatible available-case scoring includes {report['num_scored_episodes']:,} scored "
            f"episodes and excludes {excluded} preserved simulator error(s)."
        )
    for scope in (*OFFICIAL_TRACKS, "overall"):
        entry = comparison["tracks"][scope]
        episodes = (
            report.get("num_attempted_episodes", report["num_episodes_total"])
            if scope == "overall"
            else sum(OFFICIAL_EXPECTED_EPISODES[scope].values())
        )
        cells = [scope, f"{episodes:,}"]
        for metric, _ in metric_columns:
            cells.extend(
                (
                    f"{100.0 * entry['measured'][metric]:.2f}",
                    f"{100.0 * entry['reported'][metric]:.2f}",
                    f"{entry['delta_percentage_points'][metric]:+.2f}",
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        (
            "",
            (
                "All values except episode counts are percentages; deltas are measured minus reported "
                "in percentage points."
            ),
        )
    )
    if excluded:
        lines.extend(
            (
                "",
                f"Runtime-error handling: {UPSTREAM_RUNTIME_ERROR_PROTOCOL}",
                "",
                (
                    "A conservative zero-imputed sensitivity result is included in `report.json`; it is not "
                    "the Xiaomi-compatible primary comparison."
                ),
            )
        )
    return "\n".join(lines) + "\n"


def validate_official_episode_identities(results: list[dict[str, Any]], track_dir: Path) -> dict[str, Any]:
    """Match every recorded episode index and config digest to the pinned tracks."""

    track_dir = track_dir.resolve()
    expected: set[tuple[Any, ...]] = set()
    track_hashes: dict[str, str] = {}
    for track in OFFICIAL_TRACKS:
        path = track_dir / f"{track}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        track_hashes[track] = sha256(path.read_bytes()).hexdigest()
        for task in load_vlabench_episode_tasks(path, suite=track, episode_limit=50):
            expected.add(
                (
                    track,
                    task["name"],
                    task["episode_index"],
                    task["episode_config_sha256"],
                )
            )

    observed: Counter[tuple[Any, ...]] = Counter()
    for result in results:
        track = result.get("config", {}).get("params", {}).get("eval_track")
        if track not in OFFICIAL_TRACKS:
            raise ValueError(f"episode identity validation found invalid track {track!r}")
        for task in result.get("tasks", []):
            task_name = task.get("task")
            episodes = task.get("episodes")
            if not episodes:
                raise ValueError(
                    "exact episode identity validation requires per-episode inputs; "
                    f"{track}/{task_name} has only an aggregate"
                )
            for episode in episodes:
                observed[_episode_key(track, task_name, episode)] += 1

    observed_set = set(observed)
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    duplicates = sorted((key, count) for key, count in observed.items() if count != 1)
    problems: list[str] = []
    if missing:
        problems.append(f"missing {len(missing)} pinned identities (first: {missing[:3]})")
    if unexpected:
        problems.append(f"found {len(unexpected)} unexpected identities (first: {unexpected[:3]})")
    if duplicates:
        problems.append(f"found {len(duplicates)} duplicate identities (first: {duplicates[:3]})")
    if problems:
        raise ValueError("official episode identity validation failed:\n- " + "\n- ".join(problems))
    return {
        "status": "exact_complete",
        "expected_identities": len(expected),
        "observed_identities": sum(observed.values()),
        "track_file_sha256": track_hashes,
    }


def validate_official_coverage(report: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the report contains every pinned official episode."""

    problems: list[str] = []
    for track in OFFICIAL_TRACKS:
        expected = OFFICIAL_EXPECTED_EPISODES[track]
        actual = report["tracks"][track]["tasks"]
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        if missing:
            problems.append(f"{track}: missing tasks {missing}")
        if unexpected:
            problems.append(f"{track}: unexpected tasks {unexpected}")
        for task in sorted(set(expected) & set(actual)):
            count = int(actual[task]["num_episodes"])
            if count != expected[task]:
                problems.append(f"{track}/{task}: expected {expected[task]} episodes, found {count}")
    expected_total = sum(sum(tasks.values()) for tasks in OFFICIAL_EXPECTED_EPISODES.values())
    actual_total = int(report["num_episodes_total"])
    if actual_total != expected_total:
        problems.append(f"suite: expected {expected_total} episodes, found {actual_total}")
    if problems:
        raise ValueError("incomplete or incompatible official VLABench coverage:\n- " + "\n- ".join(problems))
    return {
        "status": "complete",
        "expected_episodes": expected_total,
        "observed_episodes": actual_total,
        "expected_tracks": list(OFFICIAL_TRACKS),
    }


def _runtime_error_identities(runtime_errors: list[dict[str, Any]]) -> set[EpisodeIdentity]:
    identities: set[EpisodeIdentity] = set()
    for record in runtime_errors:
        identity_record = record["identity"]
        identity = task_episode_identity(
            {
                "suite": identity_record.get("suite"),
                "name": identity_record.get("task_name"),
                "episode_index": identity_record.get("episode_index"),
                "episode_config_sha256": identity_record.get("episode_config_sha256"),
            }
        )
        if identity in identities:
            raise ValueError(f"duplicate unresolved runtime-error identity: {identity}")
        identities.add(identity)
    return identities


def validate_official_attempt_coverage(
    results: list[dict[str, Any]],
    runtime_errors: list[dict[str, Any]],
    track_dir: Path,
) -> dict[str, Any]:
    """Require each pinned identity to be either scored once or preserved as an error."""

    scored_validation = validate_official_episode_identities(results, track_dir) if not runtime_errors else None
    if scored_validation is not None:
        return {
            "status": "exact_complete",
            "expected_identities": scored_validation["expected_identities"],
            "scored_identities": scored_validation["observed_identities"],
            "runtime_error_identities": 0,
            "attempted_identities": scored_validation["observed_identities"],
            "track_file_sha256": scored_validation["track_file_sha256"],
        }

    track_dir = track_dir.resolve()
    expected: set[EpisodeIdentity] = set()
    track_hashes: dict[str, str] = {}
    for track in OFFICIAL_TRACKS:
        path = track_dir / f"{track}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        track_hashes[track] = sha256(path.read_bytes()).hexdigest()
        for task in load_vlabench_episode_tasks(path, suite=track, episode_limit=50):
            expected.add(task_episode_identity(task))

    scored: Counter[EpisodeIdentity] = Counter()
    for result in results:
        track = result.get("config", {}).get("params", {}).get("eval_track")
        for task in result.get("tasks", []):
            for episode in task.get("episodes", []):
                scored[_episode_key(track, task.get("task"), episode)] += 1
    duplicate_scored = sorted((identity, count) for identity, count in scored.items() if count != 1)
    errors = _runtime_error_identities(runtime_errors)
    overlap = sorted(set(scored) & errors)
    attempted = set(scored) | errors
    missing = sorted(expected - attempted)
    unexpected = sorted(attempted - expected)
    if duplicate_scored or overlap or missing or unexpected:
        raise ValueError(
            "official attempted-identity validation failed: "
            f"duplicates={duplicate_scored[:3]}; scored_error_overlap={overlap[:3]}; "
            f"missing={missing[:3]}; unexpected={unexpected[:3]}"
        )
    return {
        "status": "exact_complete_with_runtime_exclusions" if errors else "exact_complete",
        "expected_identities": len(expected),
        "scored_identities": len(scored),
        "runtime_error_identities": len(errors),
        "attempted_identities": len(attempted),
        "track_file_sha256": track_hashes,
    }


def zero_impute_runtime_errors(
    report: dict[str, Any], runtime_errors: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return a sensitivity analysis that assigns zero to every excluded error."""

    adjusted = copy.deepcopy(report)
    error_counts: Counter[tuple[str, str]] = Counter(
        (record["identity"]["suite"], record["identity"]["task_name"])
        for record in runtime_errors
    )
    for (track, task), error_count in error_counts.items():
        scores = adjusted["tracks"][track]["tasks"].get(task)
        if scores is None:
            scores = {metric: 0.0 for metric in METRICS}
            scores["num_episodes"] = 0
            adjusted["tracks"][track]["tasks"][task] = scores
        old_count = int(scores["num_episodes"])
        new_count = old_count + error_count
        for metric in METRICS:
            scores[metric] = float(scores[metric]) * old_count / new_count
        scores["num_episodes"] = new_count

    for track in OFFICIAL_TRACKS:
        tasks = adjusted["tracks"][track]["tasks"]
        adjusted["tracks"][track]["macro"] = {
            metric: sum(float(scores[metric]) for scores in tasks.values()) / len(tasks) for metric in METRICS
        }
        adjusted["tracks"][track]["num_tasks"] = len(tasks)
        adjusted["tracks"][track]["num_episodes"] = sum(
            int(scores["num_episodes"]) for scores in tasks.values()
        )
    all_tasks = [
        scores
        for track in OFFICIAL_TRACKS
        for scores in adjusted["tracks"][track]["tasks"].values()
    ]
    adjusted["overall"] = {
        metric: sum(float(scores[metric]) for scores in all_tasks) / len(all_tasks) for metric in METRICS
    }
    adjusted["num_episodes_total"] = sum(
        adjusted["tracks"][track]["num_episodes"] for track in OFFICIAL_TRACKS
    )
    return {
        "method": "assign_zero_to_each_unresolved_runtime_error",
        "tracks": {
            track: {
                "macro": adjusted["tracks"][track]["macro"],
                "num_episodes": adjusted["tracks"][track]["num_episodes"],
            }
            for track in OFFICIAL_TRACKS
        },
        "overall": adjusted["overall"],
        "num_episodes_total": adjusted["num_episodes_total"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "aggregates",
        nargs="*",
        type=Path,
        help="vla-eval *_aggregate.json files (five tracks, optionally split across shards)",
    )
    parser.add_argument(
        "--db",
        nargs="+",
        type=Path,
        help="sharded recording-*.sqlite files; validates every episode and step row",
    )
    parser.add_argument(
        "--track-dir",
        type=Path,
        help="directory containing the five pinned track JSON files for exact identity validation",
    )
    parser.add_argument(
        "--runtime-errors-json",
        nargs="+",
        type=Path,
        help="final unresolved error records emitted by prepare_vlabench_retries.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if bool(args.aggregates) == bool(args.db):
        parser.error("provide either aggregate JSON files or --db recording databases")
    if args.db and args.track_dir is None:
        parser.error("--track-dir is required with --db for exact pinned-episode validation")

    database_validation = None
    if args.db:
        results, database_validation = load_recording_databases(args.db)
    else:
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.aggregates]
    report = aggregate_official_vlabench(results)
    runtime_errors = load_runtime_error_records(args.runtime_errors_json or [], validate_databases=True)
    if runtime_errors:
        report["num_scored_episodes"] = report["num_episodes_total"]
        report["num_attempted_episodes"] = report["num_scored_episodes"] + len(runtime_errors)
        report["runtime_errors"] = runtime_errors
        report["scoring_protocol"] = {
            "primary": "xiaomi_compatible_available_case",
            "runtime_error_handling": UPSTREAM_RUNTIME_ERROR_PROTOCOL,
            "xiaomi_source_revision": "6bc75afb791a1938750fe5fc0aee2b0f28cf87e2",
            "vlabench_source_revision": "cf588fe60c0c7282174fe979f5913170cfe69017",
            "vlabench_evaluator_path": "VLABench/evaluation/evaluator/base.py",
            "sensitivity_analysis": "zero_imputation",
        }
        report["coverage_validation"] = {
            "status": "complete_with_runtime_exclusions",
            "expected_episodes": PINNED_PROTOCOL_EPISODES,
            "attempted_episodes": report["num_attempted_episodes"],
            "scored_episodes": report["num_scored_episodes"],
            "runtime_error_episodes": len(runtime_errors),
            "expected_tracks": list(OFFICIAL_TRACKS),
        }
        if report["num_attempted_episodes"] != PINNED_PROTOCOL_EPISODES:
            raise ValueError(
                f"attempt coverage requires {PINNED_PROTOCOL_EPISODES} identities, "
                f"found {report['num_attempted_episodes']}"
            )
        report["zero_imputed_sensitivity"] = zero_impute_runtime_errors(report, runtime_errors)
    else:
        report["coverage_validation"] = validate_official_coverage(report)
        report["num_scored_episodes"] = report["num_episodes_total"]
        report["num_attempted_episodes"] = report["num_episodes_total"]
    report["protocol_comparison"] = compare_protocol_cardinality(report)
    if args.track_dir is not None:
        report["episode_identity_validation"] = validate_official_attempt_coverage(
            results, runtime_errors, args.track_dir
        )
    if database_validation is not None:
        report["recording_validation"] = database_validation
    report["reported_comparison"] = compare_to_reported(report)
    if runtime_errors:
        report["zero_imputed_reported_comparison"] = compare_to_reported(
            report["zero_imputed_sensitivity"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_comparison_markdown(report), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
