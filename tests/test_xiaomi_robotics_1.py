from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

from sim_benchmarks.benchmarks.vlabench_xr1 import make_xr1_vlabench_observation
from sim_benchmarks.model_servers.xiaomi_robotics_1 import (
    XiaomiRobotics1VLABenchServer,
    XR1VLABenchProfile,
    build_vlabench_messages,
    normalize_vlabench_actions,
    plan_vlabench_actions,
    prepare_vlabench_observation,
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


if __name__ == "__main__":
    unittest.main()
