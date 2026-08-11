"""VLABench observation profile required by Xiaomi-Robotics-1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from vla_eval.benchmarks.base import StepResult
from vla_eval.benchmarks.vlabench.benchmark import VLABenchBenchmark
from vla_eval.specs import IMAGE_RGB, LANGUAGE, POSITION_ABSOLUTE, ROTATION_EULER, DimSpec
from vla_eval.types import Action, Observation, Task

from sim_benchmarks.model_servers.xiaomi_robotics_1 import XR1_GRIPPER, XR1_MODEL_STATE

DEFAULT_POSITION_OFFSET = (0.0, -0.4, 0.78)


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def make_xr1_vlabench_observation(
    raw_obs: dict[str, Any],
    instruction: str,
    position_offset: np.ndarray,
    quaternion_to_euler: Callable[[np.ndarray], Any],
) -> Observation:
    """Convert raw VLABench cameras/state to the explicit XR-1 profile."""

    rgb = raw_obs.get("rgb")
    if rgb is None or len(rgb) < 4:
        raise ValueError("VLABench raw observation must contain at least four RGB cameras")
    base_image, _, front_image, wrist_image = rgb[:4]

    ee_state = np.asarray(raw_obs.get("ee_state"), dtype=np.float32).reshape(-1)
    if ee_state.size < 8:
        raise ValueError(f"expected ee_state=[xyz, quaternion, gripper] with at least 8 values, got {ee_state.size}")
    if not np.all(np.isfinite(ee_state[:8])):
        raise ValueError("VLABench ee_state contains NaN or Inf")
    euler = np.asarray(quaternion_to_euler(ee_state[3:7]), dtype=np.float32).reshape(-1)
    if euler.size < 3 or not np.all(np.isfinite(euler[:3])):
        raise ValueError("quaternion_to_euler did not return three finite values")

    model_state = np.concatenate(
        [ee_state[:3] - position_offset, _wrap_to_pi(euler[:3]), ee_state[7:8]],
        dtype=np.float32,
    )
    return {
        "images": {
            "front": np.asarray(front_image),
            "base": np.asarray(base_image),
            "left_wrist": np.asarray(wrist_image),
        },
        "task_description": instruction,
        "state": model_state,
    }


class XR1VLABenchBenchmark(VLABenchBenchmark):
    """Pinned harness VLABench adapter with XR-1's named observation profile."""

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "franka",
        max_steps: int = 200,
        position_offset: tuple[float, float, float] = DEFAULT_POSITION_OFFSET,
    ) -> None:
        super().__init__(tasks=tasks, robot=robot, max_steps=max_steps)
        self._xr1_position_offset = np.asarray(position_offset, dtype=np.float32)
        if self._xr1_position_offset.shape != (3,):
            raise ValueError("position_offset must contain exactly three values")

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        del task
        if not isinstance(raw_obs, dict):
            raise TypeError(f"expected VLABench raw observation mapping, got {type(raw_obs).__name__}")
        from VLABench.utils.utils import quaternion_to_euler

        return make_xr1_vlabench_observation(
            raw_obs,
            self._instruction,
            self._xr1_position_offset,
            quaternion_to_euler,
        )

    def step(self, action: Action) -> StepResult:
        """Convert the official fixed absolute-pose plan for the base adapter."""

        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raise ValueError("XR-1 action must contain 'actions' or 'action'")
        raw_action = np.asarray(raw_action, dtype=np.float64)
        if raw_action.shape != (7,) or not np.all(np.isfinite(raw_action)):
            raise ValueError(f"XR-1 absolute action must contain seven finite values, got {raw_action.shape}")

        ee_state = self._last_ee_state
        if ee_state is None:
            ee_state = np.concatenate([self._env.get_ee_pos(), self._env.get_ee_quat(), [0.0]])
        from VLABench.utils.utils import quaternion_to_euler

        current_euler = np.asarray(quaternion_to_euler(np.asarray(ee_state)[3:7]), dtype=np.float64)
        delta_action = np.concatenate(
            [raw_action[:3] - np.asarray(ee_state)[:3], raw_action[3:6] - current_euler[:3], raw_action[6:7]]
        )
        return super().step({"actions": delta_action})

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_ABSOLUTE, "rotation": ROTATION_EULER, "gripper": XR1_GRIPPER}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "front": IMAGE_RGB,
            "base": IMAGE_RGB,
            "left_wrist": IMAGE_RGB,
            "state": XR1_MODEL_STATE,
            "language": LANGUAGE,
        }

    def get_metadata(self) -> dict[str, Any]:
        return {
            **super().get_metadata(),
            "observation_profile": "xiaomi_robotics_1_vlabench_v1",
            "position_offset": self._xr1_position_offset.tolist(),
        }
