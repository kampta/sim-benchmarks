from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from sim_benchmarks.benchmarks.vlabench_xr1 import OFFICIAL_TRACKS

_AUDIT_PATH = Path(__file__).parents[1] / "scripts" / "audit_vlabench_partial.py"
_AUDIT_SPEC = importlib.util.spec_from_file_location("audit_vlabench_partial", _AUDIT_PATH)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
audit_partial = _AUDIT_MODULE.audit_partial

_CHECKPOINT_PATH = Path(__file__).parents[1] / "scripts" / "checkpoint_vlabench_progress.py"
_CHECKPOINT_SPEC = importlib.util.spec_from_file_location("checkpoint_vlabench_progress", _CHECKPOINT_PATH)
assert _CHECKPOINT_SPEC is not None and _CHECKPOINT_SPEC.loader is not None
_CHECKPOINT_MODULE = importlib.util.module_from_spec(_CHECKPOINT_SPEC)
_CHECKPOINT_SPEC.loader.exec_module(_CHECKPOINT_MODULE)
checkpoint_progress = _CHECKPOINT_MODULE.checkpoint_progress


def _digest(config: dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _make_fixture(root: Path) -> tuple[Path, Path, Path]:
    track_dir = root / "tracks"
    track_dir.mkdir()
    identities = []
    for index, suite in enumerate(OFFICIAL_TRACKS):
        config = {"seed": index}
        task_name = f"task_{index}"
        (track_dir / f"{suite}.json").write_text(json.dumps({task_name: [config]}), encoding="utf-8")
        identities.append((suite, task_name, 0, _digest(config)))

    manifest = root / "completed.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_episodes": [
                    {
                        "suite": suite,
                        "task_name": task_name,
                        "episode_index": episode_index,
                        "episode_config_sha256": config_sha,
                    }
                    for suite, task_name, episode_index, config_sha in identities[2:]
                ],
            }
        ),
        encoding="utf-8",
    )

    database = root / "recording.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE eval_metadata (eval_id TEXT PRIMARY KEY, safe_name TEXT NOT NULL, metadata TEXT NOT NULL);
        CREATE TABLE episode_results (
            sid TEXT NOT NULL, eid TEXT NOT NULL, eval_id TEXT NOT NULL, task_name TEXT,
            episode_id INTEGER, status TEXT, metrics TEXT, steps INTEGER, elapsed_sec REAL,
            context TEXT, jsonl_path TEXT, failure_reason TEXT, failure_detail TEXT,
            PRIMARY KEY (sid, eid)
        );
        CREATE TABLE step_rows (
            sid TEXT NOT NULL, eid TEXT NOT NULL, step_id INTEGER NOT NULL, fields TEXT NOT NULL,
            PRIMARY KEY (sid, eid, step_id)
        );
        """
    )
    for suite in OFFICIAL_TRACKS[:2]:
        metadata = {"config": {"params": {"eval_track": suite}}}
        connection.execute("INSERT INTO eval_metadata VALUES (?, ?, ?)", (suite, suite, json.dumps(metadata)))

    for index, (suite, task_name, episode_index, config_sha) in enumerate(identities[:2]):
        error = index == 1
        context = {
            "name": task_name,
            "suite": suite,
            "episode_index": episode_index,
            "episode_config_sha256": config_sha,
        }
        connection.execute(
            "INSERT INTO episode_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"sid{index}",
                f"eid{index}",
                suite,
                task_name,
                0,
                "error" if error else "success",
                json.dumps(
                    {"success": False}
                    if error
                    else {"success": True, "intention_score": 0.75, "progress_score": 1.0}
                ),
                0 if error else 2,
                0.0 if error else 3.5,
                json.dumps(context),
                None,
                "exception" if error else None,
                "synthetic physics error" if error else None,
            ),
        )
    connection.executemany(
        "INSERT INTO step_rows VALUES (?, ?, ?, ?)",
        [("sid0", "eid0", 0, "{}"), ("sid0", "eid0", 1, "{}"), ("sid1", "eid1", 0, "{}")],
    )
    connection.commit()
    connection.close()
    return database, track_dir, manifest


def test_partial_audit_separates_valid_scores_from_preserved_errors(tmp_path: Path) -> None:
    database, track_dir, manifest = _make_fixture(tmp_path)

    report = audit_partial([database], track_dir=track_dir, base_completed_manifest=manifest)

    assert report["status"] == "valid_partial"
    assert report["official_identities"] == 5
    assert report["base_completed_identities"] == 3
    assert report["attempted_identities"] == 2
    assert report["valid_identities"] == 1
    assert report["total_valid_identities"] == 4
    assert report["pending_valid_identities"] == 1
    assert report["error_identities"] == 1
    assert report["unattempted_identities"] == 0
    assert report["successes"] == 1
    assert report["failures"] == 0
    assert report["valid_steps"] == 2
    assert report["errors"][0]["stored_step_rows"] == 1


def test_partial_audit_rejects_noncontiguous_valid_steps(tmp_path: Path) -> None:
    database, track_dir, manifest = _make_fixture(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM step_rows WHERE sid = 'sid0' AND step_id = 1")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="contiguous step rows"):
        audit_partial([database], track_dir=track_dir, base_completed_manifest=manifest)


def test_recovery_checkpoint_preserves_raw_errors_and_emits_exact_resume_manifest(tmp_path: Path) -> None:
    database, track_dir, manifest = _make_fixture(tmp_path)
    output_dir = tmp_path / "checkpoint"

    progress = checkpoint_progress(
        [database],
        track_dir=track_dir,
        base_completed_manifest=manifest,
        output_dir=output_dir,
    )

    assert progress["status"] == "valid_partial"
    assert progress["official_identities"] == 5
    assert progress["base_completed_identities"] == 3
    assert progress["valid_checkpoint_identities"] == 1
    assert progress["completed_identities"] == 4
    assert progress["pending_identities"] == 1
    assert progress["failed_attempt_rows"] == 1

    completed = json.loads((output_dir / "completed-manifest.json").read_text())
    assert completed["completed_episode_count"] == 4
    assert len(completed["completed_episodes"]) == 4
    failures = json.loads((output_dir / "failed-attempts.json").read_text())
    assert failures[0]["failure_reason"] == "exception"

    raw = sqlite3.connect(output_dir / "raw" / "shard0.sqlite")
    clean = sqlite3.connect(output_dir / "clean" / "shard0.sqlite")
    original = sqlite3.connect(database)
    try:
        assert raw.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0] == 2
        assert raw.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0] == 3
        assert clean.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0] == 1
        assert clean.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0] == 2
        assert original.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0] == 2
        assert original.execute("SELECT COUNT(*) FROM step_rows").fetchone()[0] == 3
    finally:
        raw.close()
        clean.close()
        original.close()

    checksum_lines = (output_dir / "SHA256SUMS").read_text().splitlines()
    assert any(line.endswith("  completed-manifest.json") for line in checksum_lines)
