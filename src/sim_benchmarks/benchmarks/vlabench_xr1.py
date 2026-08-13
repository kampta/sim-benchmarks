"""VLABench observation profile required by Xiaomi-Robotics-1."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from vla_eval.benchmarks.base import StepResult
from vla_eval.benchmarks.vlabench.benchmark import VLABenchBenchmark
from vla_eval.specs import IMAGE_RGB, LANGUAGE, POSITION_ABSOLUTE, ROTATION_EULER, DimSpec
from vla_eval.types import Action, Observation, Task

from sim_benchmarks.model_servers.xiaomi_robotics_1 import XR1_GRIPPER, XR1_MODEL_STATE

DEFAULT_POSITION_OFFSET = (0.0, -0.4, 0.78)
OFFICIAL_DEFAULT_MAX_STEPS = 200
OFFICIAL_TRACKS = (
    "track_1_in_distribution",
    "track_2_cross_category",
    "track_3_common_sense",
    "track_4_semantic_instruction",
    "track_6_unseen_texture",
)


def resolve_vlabench_track_path(eval_track: str, root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve one immutable track file from the pinned VLABench checkout."""

    if eval_track not in OFFICIAL_TRACKS:
        raise ValueError(f"unknown official VLABench track {eval_track!r}; expected one of {OFFICIAL_TRACKS}")
    root_value = root if root is not None else os.environ.get("VLABENCH_ROOT")
    if not root_value:
        raise OSError("VLABENCH_ROOT must be set to load an official VLABench track")
    vlabench_root = Path(root_value)
    candidates = (
        vlabench_root / "configs" / "evaluation" / "tracks" / f"{eval_track}.json",
        vlabench_root / "VLABench" / "configs" / "evaluation" / "tracks" / f"{eval_track}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("VLABench track file not found; tried: " + ", ".join(map(str, candidates)))


def load_vlabench_episode_tasks(
    path: str | os.PathLike[str],
    *,
    suite: str,
    selected_tasks: list[str] | None = None,
    episode_limit: int = 50,
) -> list[Task]:
    """Flatten an official track into deterministic, independently shardable episodes."""

    if episode_limit < 1:
        raise ValueError("episode_limit must be positive")
    with Path(path).open(encoding="utf-8") as stream:
        track = json.load(stream)
    if not isinstance(track, dict):
        raise TypeError(f"VLABench track must be a task mapping, got {type(track).__name__}")

    names = list(track) if selected_tasks is None else selected_tasks
    missing = [name for name in names if name not in track]
    if missing:
        raise ValueError(f"tasks absent from VLABench track {suite}: {missing}")

    tasks: list[Task] = []
    for name in names:
        episodes = track[name]
        if not isinstance(episodes, list):
            raise TypeError(f"VLABench track entry {name!r} must be a list")
        for episode_index, episode_config in enumerate(episodes[:episode_limit]):
            if not isinstance(episode_config, dict):
                raise TypeError(f"VLABench episode {name}[{episode_index}] must be a mapping")
            tasks.append(
                {
                    "name": name,
                    "suite": suite,
                    "episode_index": episode_index,
                    "episode_config_sha256": sha256(
                        json.dumps(episode_config, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "episode_config": episode_config,
                }
            )
    return tasks


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def resolve_episode_max_steps(task_config: dict[str, Any]) -> int:
    """Match Xiaomi's evaluator default and per-task horizon override."""

    return int(
        task_config.get("evaluation", {}).get(
            "max_episode_length",
            OFFICIAL_DEFAULT_MAX_STEPS,
        )
    )


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
        max_steps: int = 300,
        position_offset: tuple[float, float, float] = DEFAULT_POSITION_OFFSET,
        eval_track: str | None = None,
        episode_limit: int = 50,
        seed: int = 42,
        intention_score_threshold: float = 0.1,
    ) -> None:
        super().__init__(tasks=tasks, robot=robot, max_steps=max_steps)
        self._xr1_position_offset = np.asarray(position_offset, dtype=np.float32)
        if self._xr1_position_offset.shape != (3,):
            raise ValueError("position_offset must contain exactly three values")
        if eval_track is not None and eval_track not in OFFICIAL_TRACKS:
            raise ValueError(f"unknown official VLABench track {eval_track!r}")
        if episode_limit < 1:
            raise ValueError("episode_limit must be positive")
        self._selected_tasks = tasks
        self._eval_track = eval_track
        self._episode_limit = episode_limit
        self._seed = seed
        self._intention_score_threshold = intention_score_threshold
        self._episode_max_steps = max_steps
        self._episode_step = 0

    def get_tasks(self) -> list[Task]:
        if self._eval_track is None:
            return super().get_tasks()
        track_path = resolve_vlabench_track_path(self._eval_track)
        return load_vlabench_episode_tasks(
            track_path,
            suite=self._eval_track,
            selected_tasks=self._selected_tasks,
            episode_limit=self._episode_limit,
        )

    def reset(self, task: Task) -> Any:
        """Load the exact episode configuration used by Xiaomi's dispatcher."""

        self._ensure_vlabench()
        from VLABench.configs import name2config
        from VLABench.envs import load_env
        from VLABench.utils.utils import find_key_by_value

        task_name = task["name"]
        episode_config = task.get("episode_config")
        episode_index = int(task.get("episode_index", task.get("episode_idx", 0)))
        if episode_config is None:
            np.random.seed(self._seed + episode_index)
            random.seed(self._seed + episode_index)

        if self._env is not None:
            with suppress(Exception):
                self._env.close()
        self._env = load_env(
            task_name,
            robot=self._robot,
            episode_config=episode_config,
            random_init=episode_config is None,
            eval=False,
            run_mode="eval",
        )
        # Xiaomi's pinned Evaluator resets once more after load_env's internal
        # reset; preserve that behavior for episode-level parity.
        self._env.reset()
        self._current_task = task_name
        self._episode_step = 0

        task_series = find_key_by_value(name2config, task_name)
        task_config_path = Path(os.environ["VLABENCH_ROOT"]) / "configs" / "task_config.json"
        with task_config_path.open(encoding="utf-8") as stream:
            all_task_configs = json.load(stream)
        task_config = all_task_configs.get(task_series, {}) if task_series is not None else {}
        self._episode_max_steps = resolve_episode_max_steps(task_config)

        obs = self._env.get_observation(require_pcd=False)
        self._instruction = self._env.task.get_instruction()
        self._last_ee_state = obs.get("ee_state", None)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

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
        result = super().step({"actions": delta_action})
        self._episode_step += 1
        return result

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done or self._episode_step >= self._episode_max_steps

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        return {
            "success": bool(step_result.info.get("success", False)),
            "intention_score": float(self._env.get_intention_score(threshold=self._intention_score_threshold)),
            "progress_score": float(self._env.get_task_progress()),
        }

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "intention_score": "mean", "progress_score": "mean"}

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
            "eval_track": self._eval_track,
            "episode_limit": self._episode_limit,
            "seed": self._seed,
            "official_tracks": list(OFFICIAL_TRACKS),
        }
