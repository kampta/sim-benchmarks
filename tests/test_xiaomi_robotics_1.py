from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

from sim_benchmarks.benchmarks.vlabench_xr1 import (
    XR1VLABenchBenchmark,
    load_completed_episode_identities,
    load_vlabench_episode_tasks,
    make_xr1_vlabench_observation,
    resolve_episode_max_steps,
    task_episode_identity,
)
from sim_benchmarks.model_servers.xiaomi_robotics_1 import (
    XiaomiRobotics1VLABenchServer,
    XR1VLABenchProfile,
    build_vlabench_messages,
    normalize_vlabench_actions,
    plan_vlabench_actions,
    prepare_vlabench_observation,
)
from sim_benchmarks.provenance.artifacts import verify_checkpoint_files
from sim_benchmarks.reporting.vlabench import (
    OFFICIAL_EXPECTED_EPISODES,
    aggregate_official_vlabench,
    compare_protocol_cardinality,
    compare_to_reported,
    load_recording_databases,
    load_runtime_error_records,
    render_comparison_markdown,
    validate_official_attempt_coverage,
    validate_official_coverage,
    validate_official_episode_identities,
    zero_impute_runtime_errors,
)


def observation(image_size: int = 2) -> dict[str, Any]:
    return {
        "images": {
            "front": np.full((image_size, image_size, 3), 30, dtype=np.uint8),
            "base": np.full((image_size, image_size, 3), 10, dtype=np.uint8),
            "left_wrist": np.full((image_size, image_size, 3), 40, dtype=np.uint8),
        },
        "task_description": "select the red fruit",
        "state": np.arange(7, dtype=np.float32),
    }


class _InferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


class _FakeTorch:
    @staticmethod
    def inference_mode() -> _InferenceMode:
        return _InferenceMode()


class _FakeProcessor:
    def __init__(self, decoded: np.ndarray) -> None:
        self.decoded = decoded
        self.template_kwargs: dict[str, Any] | None = None
        self.decode_robot_type: str | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.template_kwargs = {"messages": messages, **kwargs}
        return {"input_ids": np.asarray([[1]], dtype=np.int64)}

    def decode_action(self, actions: Any, *, robot_type: str) -> np.ndarray:
        del actions
        self.decode_robot_type = robot_type
        return self.decoded


class _FakeModel:
    def __init__(self) -> None:
        self.inputs: dict[str, Any] | None = None

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        self.inputs = inputs
        return SimpleNamespace(actions=np.zeros((1, 10, 60), dtype=np.float32))


def fake_server(decoded: np.ndarray) -> XiaomiRobotics1VLABenchServer:
    server = object.__new__(XiaomiRobotics1VLABenchServer)
    PredictModelServer.__init__(server, chunk_size=5)
    server.profile = XR1VLABenchProfile(image_size=2)
    server.model_path = "test"
    server.revision = "0" * 40
    server._torch = _FakeTorch()
    server._device = "cuda"
    server._dtype = "bfloat16"
    server._processor = _FakeProcessor(decoded)
    server._model = _FakeModel()
    return server


class XiaomiRobotics1CodecTests(unittest.TestCase):
    def test_preparation_preserves_explicit_camera_order_and_pads_state(self) -> None:
        images, instruction, state = prepare_vlabench_observation(
            observation(),
            XR1VLABenchProfile(image_size=2),
        )
        self.assertEqual([np.asarray(image)[0, 0, 0] for image in images], [30, 10, 40])
        self.assertEqual(instruction, "select the red fruit")
        self.assertEqual(state.shape, (60,))
        np.testing.assert_array_equal(state[:7], np.arange(7, dtype=np.float32))
        np.testing.assert_array_equal(state[7:], np.zeros(53, dtype=np.float32))

    def test_preparation_fails_on_missing_camera_or_wrong_state(self) -> None:
        obs = observation()
        del obs["images"]["left_wrist"]
        with self.assertRaisesRegex(ValueError, "left_wrist"):
            prepare_vlabench_observation(obs, XR1VLABenchProfile(image_size=2))

        obs = observation()
        obs["state"] = np.zeros(8, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"shape \(7,\)"):
            prepare_vlabench_observation(obs, XR1VLABenchProfile(image_size=2))

    def test_prompt_matches_cot_modes(self) -> None:
        images = [object(), object(), object()]
        no_cot = build_vlabench_messages(images, "pick", cot=False)
        self.assertEqual(no_cot[-1]["content"][0]["text"], "<cot></cot>")
        self.assertTrue(no_cot[0]["content"][-1]["text"].endswith("pick /no_cot"))
        self.assertEqual(
            [part.get("text") for part in no_cot[0]["content"]].count("\n# Left-Wrist View\n"),
            1,
        )

        cot = build_vlabench_messages(images, "pick", cot=True)
        self.assertEqual(len(cot), 1)
        self.assertTrue(cot[0]["content"][-1]["text"].endswith("pick /cot"))

    def test_action_decode_slices_dimensions_and_thresholds_gripper(self) -> None:
        decoded = np.zeros((1, 10, 60), dtype=np.float32)
        decoded[0, :5, 6] = np.asarray([0.19, 0.2, -2.0, 1.0, 0.0])
        actions = normalize_vlabench_actions(decoded, XR1VLABenchProfile())
        self.assertEqual(actions.shape, (10, 7))
        np.testing.assert_array_equal(actions[:5, 6], [-1.0, 1.0, -1.0, 1.0, -1.0])

    def test_action_decode_fails_closed_on_shape_horizon_and_nonfinite_values(self) -> None:
        profile = XR1VLABenchProfile()
        for invalid in (
            np.zeros((2, 10, 60), dtype=np.float32),
            np.zeros((5, 60), dtype=np.float32),
            np.zeros((10, 6), dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                normalize_vlabench_actions(invalid, profile)
        invalid = np.zeros((10, 60), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            normalize_vlabench_actions(invalid, profile)

    def test_action_plan_accumulates_deltas_from_replan_state(self) -> None:
        decoded = np.zeros((10, 60), dtype=np.float32)
        decoded[:, 0] = 0.1
        decoded[:, 3] = np.pi / 2
        decoded[:, 6] = 0.3
        state = np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        planned = plan_vlabench_actions(decoded, state, XR1VLABenchProfile())
        np.testing.assert_allclose(planned[:3, 0], [1.1, 1.2, 1.3], atol=1e-6)
        np.testing.assert_allclose(planned[0, 1:3], [1.6, 3.78], atol=1e-6)
        np.testing.assert_allclose(planned[[0, 2], 3], [np.pi / 2, -np.pi / 2], atol=1e-6)
        self.assertAlmostEqual(abs(float(planned[1, 3])), np.pi, places=6)
        np.testing.assert_array_equal(planned[:, 6], np.ones(10, dtype=np.float32))

    def test_benchmark_profile_maps_raw_camera_and_state_contract(self) -> None:
        rgb = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
        raw = {"rgb": rgb, "ee_state": np.asarray([1, 2, 3, 0, 0, 0, 1, 0.5], dtype=np.float32)}
        result = make_xr1_vlabench_observation(
            raw,
            "pick",
            np.asarray([0.0, -0.4, 0.78], dtype=np.float32),
            lambda _: np.asarray([0.0, 2 * np.pi, -3 * np.pi]),
        )
        self.assertEqual([result["images"][key][0, 0, 0] for key in ("front", "base", "left_wrist")], [30, 10, 40])
        np.testing.assert_allclose(result["state"][:3], [1.0, 2.4, 2.22], atol=1e-6)
        np.testing.assert_allclose(result["state"][3:6], [0.0, 0.0, -np.pi], atol=1e-6)
        self.assertEqual(result["state"][6], 0.5)

    def test_official_track_is_flattened_into_deterministic_episode_tasks(self) -> None:
        track = {
            "select_fruit": [{"seed": 1}, {"seed": 2}],
            "select_toy": [{"seed": 3}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/track.json"
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(track, stream)
            tasks = load_vlabench_episode_tasks(
                path,
                suite="track_1_in_distribution",
                selected_tasks=["select_fruit"],
                episode_limit=1,
            )
        self.assertEqual(
            {key: value for key, value in tasks[0].items() if key != "episode_config_sha256"},
            {
                "name": "select_fruit",
                "suite": "track_1_in_distribution",
                "episode_index": 0,
                "episode_config": {"seed": 1},
            },
        )
        self.assertRegex(tasks[0]["episode_config_sha256"], r"^[0-9a-f]{64}$")

    def test_resume_manifest_filters_only_the_exact_four_part_identity(self) -> None:
        track = "track_1_in_distribution"
        config = {"seed": 1}
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track_dir = root / "configs" / "evaluation" / "tracks"
            track_dir.mkdir(parents=True)
            (track_dir / f"{track}.json").write_text(
                json.dumps({"select_fruit": [config]}), encoding="utf-8"
            )
            manifest = root / "completed.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "completed_episodes": [
                            {
                                "suite": "track_2_cross_category",
                                "task_name": "select_fruit",
                                "episode_index": 0,
                                "episode_config_sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            benchmark = XR1VLABenchBenchmark(
                eval_track=track,
                episode_limit=1,
                completed_episode_manifest=str(manifest),
            )
            with mock.patch.dict(os.environ, {"VLABENCH_ROOT": str(root)}):
                tasks = benchmark.get_tasks()
            self.assertEqual(len(tasks), 1)

            identity = task_episode_identity(tasks[0])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["completed_episodes"].append(
                {
                    "suite": identity[0],
                    "task_name": identity[1],
                    "episode_index": identity[2],
                    "episode_config_sha256": identity[3],
                }
            )
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.dict(os.environ, {"VLABENCH_ROOT": str(root)}):
                self.assertEqual(benchmark.get_tasks(), [])

    def test_resume_manifest_rejects_duplicates_and_unknown_same_track_identity(self) -> None:
        track = "track_1_in_distribution"
        entry = {
            "suite": track,
            "task_name": "select_fruit",
            "episode_index": 0,
            "episode_config_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "completed.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "completed_episodes": [entry, entry]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate completed episode"):
                load_completed_episode_identities(manifest)

            track_dir = root / "configs" / "evaluation" / "tracks"
            track_dir.mkdir(parents=True)
            (track_dir / f"{track}.json").write_text(
                json.dumps({"select_fruit": [{"seed": 1}]}), encoding="utf-8"
            )
            manifest.write_text(
                json.dumps({"schema_version": 1, "completed_episodes": [entry]}),
                encoding="utf-8",
            )
            benchmark = XR1VLABenchBenchmark(
                eval_track=track,
                episode_limit=1,
                completed_episode_manifest=str(manifest),
            )
            with (
                mock.patch.dict(os.environ, {"VLABENCH_ROOT": str(root)}),
                self.assertRaisesRegex(ValueError, "unknown identities"),
            ):
                benchmark.get_tasks()

    def test_episode_horizon_matches_official_default_and_task_override(self) -> None:
        self.assertEqual(resolve_episode_max_steps({}), 200)
        self.assertEqual(
            resolve_episode_max_steps({"evaluation": {"max_episode_length": 300}}),
            300,
        )

    def test_checkpoint_verifier_checks_size_and_digest(self) -> None:
        payload = b"immutable checkpoint shard"

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/model.safetensors"
            with open(path, "wb") as stream:
                stream.write(payload)
            checkpoint = {
                "weight_files": [
                    {
                        "path": "model.safetensors",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
            checks = verify_checkpoint_files(Path(directory), checkpoint)
            self.assertTrue(checks[0]["size_ok"])
            self.assertTrue(checks[0]["sha256_ok"])

            checkpoint["weight_files"][0]["size"] += 1
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                verify_checkpoint_files(Path(directory), checkpoint)

    def test_server_predict_builds_official_request_and_base_trims_chunk(self) -> None:
        decoded = np.zeros((1, 10, 60), dtype=np.float32)
        decoded[..., 6] = 0.3
        server = fake_server(decoded)
        ctx = SessionContext("session", "episode")

        result = server.predict(observation(), ctx)
        self.assertEqual(result["actions"].shape, (10, 7))
        normalized = server._normalize_result(result, ctx)
        self.assertEqual(normalized["actions"].shape, (5, 7))
        self.assertEqual(server._processor.decode_robot_type, "vlabench_choice")
        self.assertEqual(server._processor.template_kwargs["state"].shape, (1, 1, 60))
        self.assertEqual(server._model.inputs["task_id"], "vlabench_choice")
        self.assertEqual(server._model.inputs["seed"], 42)

    def test_profile_rejects_incompatible_action_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 7"):
            XR1VLABenchProfile(action_dim=8)
        with self.assertRaisesRegex(ValueError, "execute_horizon"):
            XR1VLABenchProfile(execute_horizon=11)

    def test_vlabench_report_uses_task_macro_not_episode_weighting(self) -> None:
        tracks = (
            "track_1_in_distribution",
            "track_2_cross_category",
            "track_3_common_sense",
            "track_4_semantic_instruction",
            "track_6_unseen_texture",
        )
        results = []
        for index, track in enumerate(tracks):
            results.append(
                {
                    "config": {"params": {"eval_track": track}},
                    "tasks": [
                        {
                            "task": f"task_{index}_a",
                            "mean_success": 1,
                            "mean_intention_score": 0.5,
                            "mean_progress_score": 0.25,
                        },
                        {
                            "task": f"task_{index}_b",
                            "mean_success": 0,
                            "mean_intention_score": 0.25,
                            "mean_progress_score": 0.75,
                        },
                    ],
                }
            )
        report = aggregate_official_vlabench(results)
        self.assertEqual(report["aggregation"], "macro_average_across_task_entries")
        self.assertEqual(report["overall_aggregation"], "macro_average_across_all_task_entries")
        self.assertEqual(report["overall"]["success"], 0.5)
        self.assertEqual(report["overall"]["intention_score"], 0.375)
        self.assertEqual(report["tracks"][tracks[0]]["macro"]["progress_score"], 0.5)

    def test_vlabench_report_pools_shards_and_deduplicates_episodes(self) -> None:
        results = []
        for track in (
            "track_1_in_distribution",
            "track_2_cross_category",
            "track_3_common_sense",
            "track_4_semantic_instruction",
            "track_6_unseen_texture",
        ):
            for episode_index, success in ((0, False), (1, True)):
                results.append(
                    {
                        "config": {"params": {"eval_track": track}},
                        "tasks": [
                            {
                                "task": "select_fruit",
                                "episodes": [
                                    {
                                        "episode_index": episode_index,
                                        "episode_config_sha256": str(episode_index) * 64,
                                        "metrics": {
                                            "success": success,
                                            "intention_score": 0.5 + 0.5 * success,
                                            "progress_score": 0.25 + 0.5 * success,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )
        results.append(results[0])  # A resumed merge must not double-count an episode.

        report = aggregate_official_vlabench(results)

        self.assertEqual(report["num_episodes_total"], 10)
        self.assertEqual(report["overall"]["success"], 0.5)
        first_track = report["tracks"]["track_1_in_distribution"]
        self.assertEqual(first_track["num_episodes"], 2)
        self.assertEqual(first_track["tasks"]["select_fruit"]["intention_score"], 0.75)

    def test_vlabench_report_compares_all_tracks_to_published_table(self) -> None:
        report = {
            "tracks": {
                track: {"macro": {"success": 0.5, "intention_score": 0.6, "progress_score": 0.7}}
                for track in (
                    "track_1_in_distribution",
                    "track_2_cross_category",
                    "track_3_common_sense",
                    "track_4_semantic_instruction",
                    "track_6_unseen_texture",
                )
            },
            "overall": {"success": 0.5, "intention_score": 0.6, "progress_score": 0.7},
        }

        comparison = compare_to_reported(report)

        self.assertEqual(set(comparison["tracks"]), {*report["tracks"], "overall"})
        self.assertAlmostEqual(
            comparison["tracks"]["track_1_in_distribution"]["delta_percentage_points"]["success"],
            -25.6,
        )
        self.assertAlmostEqual(
            comparison["tracks"]["overall"]["delta_percentage_points"]["progress_score"],
            -0.3,
        )

    def test_vlabench_comparison_markdown_orders_sr_ps_is_and_includes_all_scopes(self) -> None:
        report = {
            "tracks": {
                track: {
                    "macro": {"success": 0.5, "intention_score": 0.6, "progress_score": 0.7},
                    "num_episodes": sum(OFFICIAL_EXPECTED_EPISODES[track].values()),
                    "tasks": {
                        task: {"num_episodes": count}
                        for task, count in OFFICIAL_EXPECTED_EPISODES[track].items()
                    },
                }
                for track in OFFICIAL_EXPECTED_EPISODES
            },
            "overall": {"success": 0.5, "intention_score": 0.6, "progress_score": 0.7},
            "num_episodes_total": 2460,
        }
        report["reported_comparison"] = compare_to_reported(report)
        report["protocol_comparison"] = compare_protocol_cardinality(report)

        markdown = render_comparison_markdown(report)

        self.assertIn("| Scope | Episodes | Measured SR | Reported SR | ΔSR (pp)", markdown)
        self.assertLess(markdown.index("Measured PS"), markdown.index("Measured IS"))
        self.assertIn("| track_2_cross_category | 460 |", markdown)
        self.assertIn("| overall | 2,460 |", markdown)
        self.assertIn("| track_6_unseen_texture |", markdown)
        self.assertIn("paper describes 2,500 rollouts", markdown)
        self.assertIn("pinned released tracks contain 2,460", markdown)

    def test_vlabench_protocol_comparison_records_released_cardinality_gap(self) -> None:
        report = {
            "tracks": {
                track: {"tasks": {task: {"num_episodes": count} for task, count in expected.items()}}
                for track, expected in OFFICIAL_EXPECTED_EPISODES.items()
            },
            "num_episodes_total": 2460,
        }

        comparison = compare_protocol_cardinality(report)

        self.assertEqual(comparison["paper_described_episodes"], 2500)
        self.assertEqual(comparison["pinned_released_episodes"], 2460)
        self.assertEqual(comparison["difference"], -40)
        self.assertIn("insert_flower", comparison["reason"])

    def test_vlabench_coverage_gate_requires_every_pinned_episode(self) -> None:
        tracks = {
            track: {"tasks": {task: {"num_episodes": count} for task, count in expected.items()}}
            for track, expected in OFFICIAL_EXPECTED_EPISODES.items()
        }
        expected_total = sum(sum(tasks.values()) for tasks in OFFICIAL_EXPECTED_EPISODES.values())
        report = {"tracks": tracks, "num_episodes_total": expected_total}

        coverage = validate_official_coverage(report)

        self.assertEqual(coverage["status"], "complete")
        self.assertEqual(coverage["observed_episodes"], 2460)

        tracks["track_2_cross_category"]["tasks"]["insert_flower"]["num_episodes"] = 9
        report["num_episodes_total"] -= 1
        with self.assertRaisesRegex(ValueError, "track_2_cross_category/insert_flower"):
            validate_official_coverage(report)

    def test_vlabench_recording_loader_validates_episode_and_step_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "recording-test.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE eval_metadata (
                    eval_id TEXT PRIMARY KEY, safe_name TEXT NOT NULL, metadata TEXT NOT NULL
                );
                CREATE TABLE episode_results (
                    sid TEXT NOT NULL, eid TEXT NOT NULL, eval_id TEXT NOT NULL,
                    task_name TEXT, episode_id INTEGER, status TEXT, metrics TEXT,
                    steps INTEGER, elapsed_sec REAL, context TEXT, jsonl_path TEXT,
                    failure_reason TEXT, failure_detail TEXT,
                    PRIMARY KEY (sid, eid)
                );
                CREATE TABLE step_rows (
                    sid TEXT NOT NULL, eid TEXT NOT NULL, step_id INTEGER NOT NULL,
                    fields TEXT NOT NULL, PRIMARY KEY (sid, eid, step_id)
                );
                """
            )
            metadata = {"config": {"params": {"eval_track": "track_1_in_distribution"}}}
            context = {
                "name": "select_fruit",
                "suite": "track_1_in_distribution",
                "episode_index": 7,
                "episode_config_sha256": "a" * 64,
            }
            metrics = {"success": True, "intention_score": 0.75, "progress_score": 1.0}
            connection.execute(
                "INSERT INTO eval_metadata VALUES (?, ?, ?)",
                ("eval", "safe", json.dumps(metadata)),
            )
            connection.execute(
                "INSERT INTO episode_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "sid",
                    "eid",
                    "eval",
                    "select_fruit",
                    0,
                    "success",
                    json.dumps(metrics),
                    2,
                    3.5,
                    json.dumps(context),
                    None,
                    None,
                    None,
                ),
            )
            connection.executemany(
                "INSERT INTO step_rows VALUES (?, ?, ?, ?)",
                [("sid", "eid", 0, "{}"), ("sid", "eid", 1, "{}")],
            )
            connection.commit()
            connection.close()

            results, validation = load_recording_databases([database])

            self.assertEqual(validation[0]["sqlite_quick_check"], "ok")
            self.assertEqual(validation[0]["episode_results"], 1)
            self.assertEqual(validation[0]["step_rows"], 2)
            episode = results[0]["tasks"][0]["episodes"][0]
            self.assertEqual(episode["episode_index"], 7)
            self.assertEqual(episode["metrics"]["intention_score"], 0.75)

            connection = sqlite3.connect(database)
            connection.execute("DELETE FROM step_rows WHERE step_id = 1")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "contiguous step rows"):
                load_recording_databases([database])

    def test_vlabench_exact_identity_validation_uses_pinned_track_contents(self) -> None:
        tracks = (
            "track_1_in_distribution",
            "track_2_cross_category",
            "track_3_common_sense",
            "track_4_semantic_instruction",
            "track_6_unseen_texture",
        )
        with tempfile.TemporaryDirectory() as directory:
            track_dir = Path(directory)
            results = []
            for index, track in enumerate(tracks):
                episode_config = {"seed": index}
                (track_dir / f"{track}.json").write_text(
                    json.dumps({"select_fruit": [episode_config]}), encoding="utf-8"
                )
                digest = hashlib.sha256(
                    json.dumps(episode_config, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                results.append(
                    {
                        "config": {"params": {"eval_track": track}},
                        "tasks": [
                            {
                                "task": "select_fruit",
                                "episodes": [
                                    {
                                        "episode_index": 0,
                                        "episode_config_sha256": digest,
                                    }
                                ],
                            }
                        ],
                    }
                )

            validation = validate_official_episode_identities(results, track_dir)

            self.assertEqual(validation["status"], "exact_complete")
            self.assertEqual(validation["observed_identities"], 5)
            results.append(results[0])
            with self.assertRaisesRegex(ValueError, "duplicate identities"):
                validate_official_episode_identities(results, track_dir)

    def test_vlabench_attempt_coverage_accepts_preserved_runtime_error(self) -> None:
        tracks = tuple(OFFICIAL_EXPECTED_EPISODES)
        with tempfile.TemporaryDirectory() as directory:
            track_dir = Path(directory)
            results = []
            identities = []
            for index, track in enumerate(tracks):
                config = {"seed": index}
                (track_dir / f"{track}.json").write_text(
                    json.dumps({"select_fruit": [config]}), encoding="utf-8"
                )
                digest = hashlib.sha256(
                    json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                identities.append((track, digest))
                if index < len(tracks) - 1:
                    results.append(
                        {
                            "config": {"params": {"eval_track": track}},
                            "tasks": [
                                {
                                    "task": "select_fruit",
                                    "episodes": [
                                        {"episode_index": 0, "episode_config_sha256": digest}
                                    ],
                                }
                            ],
                        }
                    )
            error = {
                "identity": {
                    "suite": identities[-1][0],
                    "task_name": "select_fruit",
                    "episode_index": 0,
                    "episode_config_sha256": identities[-1][1],
                },
                "status": "error",
                "failure_reason": "exception",
            }

            validation = validate_official_attempt_coverage(results, [error], track_dir)

            self.assertEqual(validation["status"], "exact_complete_with_runtime_exclusions")
            self.assertEqual(validation["scored_identities"], 4)
            self.assertEqual(validation["runtime_error_identities"], 1)
            self.assertEqual(validation["attempted_identities"], 5)

    def test_vlabench_runtime_error_loader_and_zero_imputation(self) -> None:
        error = {
            "identity": {
                "suite": "track_1_in_distribution",
                "task_name": "select_fruit",
                "episode_index": 7,
                "episode_config_sha256": "a" * 64,
            },
            "status": "error",
            "failure_reason": "exception",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errors.json"
            path.write_text(json.dumps([error]), encoding="utf-8")
            self.assertEqual(load_runtime_error_records([path]), [error])

            database = Path(directory) / "raw.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE episode_results (
                    sid TEXT, eid TEXT, status TEXT, failure_reason TEXT
                );
                CREATE TABLE step_rows (
                    sid TEXT, eid TEXT, step_id INTEGER, fields TEXT
                );
                INSERT INTO episode_results VALUES ('sid', 'eid', 'error', 'exception');
                INSERT INTO step_rows VALUES ('sid', 'eid', 0, '{}');
                INSERT INTO step_rows VALUES ('sid', 'eid', 1, '{}');
                """
            )
            connection.commit()
            connection.close()
            strict_error = {**error, "database": str(database), "sid": "sid", "eid": "eid"}
            path.write_text(json.dumps([strict_error]), encoding="utf-8")

            validated = load_runtime_error_records([path], validate_databases=True)

            self.assertEqual(validated[0]["sqlite_quick_check"], "ok")
            self.assertEqual(validated[0]["stored_step_rows"], 2)
            self.assertEqual(len(validated[0]["preserved_database_sha256"]), 64)

        report = {
            "tracks": {
                track: {
                    "tasks": {
                        "select_fruit": {
                            "success": 0.5,
                            "intention_score": 0.5,
                            "progress_score": 0.5,
                            "num_episodes": 1,
                        }
                    },
                    "macro": {metric: 0.5 for metric in ("success", "intention_score", "progress_score")},
                    "num_tasks": 1,
                    "num_episodes": 1,
                }
                for track in OFFICIAL_EXPECTED_EPISODES
            },
            "overall": {metric: 0.5 for metric in ("success", "intention_score", "progress_score")},
            "num_episodes_total": 5,
        }

        adjusted = zero_impute_runtime_errors(report, [error])

        self.assertEqual(adjusted["num_episodes_total"], 6)
        self.assertEqual(adjusted["tracks"]["track_1_in_distribution"]["macro"]["success"], 0.25)
        self.assertAlmostEqual(adjusted["overall"]["success"], 0.45)


if __name__ == "__main__":
    unittest.main()
