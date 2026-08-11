# /// script
# requires-python = "~=3.12"
# dependencies = [
#     "sim-benchmarks",
#     "torch==2.8.0",
#     "torchvision==0.23.0",
#     "transformers==4.57.1",
#     "flash-attn==2.8.3",
#     "pillow>=10",
#     "numpy>=1.26",
# ]
#
# [tool.uv.sources]
# sim-benchmarks = { path = "../../..", editable = true }
#
# [tool.uv.extra-build-dependencies]
# flash-attn = [{ requirement = "torch", match-runtime = true }]
#
# [tool.uv]
# exclude-newer = "2026-08-11T00:00:00Z"
# ///
"""Xiaomi-Robotics-1 server for the released VLABench checkpoint.

The official example splits preprocessing and inference across an unsafe
pickle-over-TCP connection. This implementation keeps both in one process and
uses vla-eval's WebSocket/MessagePack transport.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, POSITION_ABSOLUTE, ROTATION_EULER, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "XiaomiRobotics/Xiaomi-Robotics-1-VLABench"
DEFAULT_MODEL_REVISION = "2dfc33b390478f71737eacb4748333e6d8638a06"
VLABENCH_IMAGE_KEYS = ("front", "base", "left_wrist")

XR1_MODEL_STATE = DimSpec(
    "state",
    7,
    "eef_robot_relative_pos3_euler3_gripper1",
    description="Position relative to [0, -0.4, 0.78], world-frame Euler XYZ, scalar gripper.",
)
XR1_GRIPPER = DimSpec(
    "gripper",
    1,
    "raw",
    (-1.0, 1.0),
    description="Binarized for VLABench: +1 opens and -1 closes; decoded threshold is 0.2.",
)


@dataclass(frozen=True)
class XR1VLABenchProfile:
    """Immutable preprocessing and action-decoding contract."""

    robot_type: str = "vlabench_choice"
    state_dim: int = 60
    action_dim: int = 7
    model_action_horizon: int = 10
    execute_horizon: int = 5
    image_size: int = 480
    cot: bool = False
    request_seed: int = 42
    gripper_threshold: float = 0.2
    position_offset: tuple[float, float, float] = (0.0, -0.4, 0.78)

    def __post_init__(self) -> None:
        if not self.robot_type:
            raise ValueError("robot_type cannot be empty")
        if self.state_dim < 7:
            raise ValueError("state_dim must be at least 7")
        if self.action_dim != 7:
            raise ValueError("the VLABench embodiment requires exactly 7 executable action dimensions")
        if self.model_action_horizon < self.execute_horizon or self.execute_horizon < 1:
            raise ValueError("execute_horizon must be in [1, model_action_horizon]")
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if len(self.position_offset) != 3:
            raise ValueError("position_offset must contain exactly three values")


def _to_pil(image: Any, image_size: int) -> Any:
    from PIL import Image

    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected an ndarray image, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected an HWC image with 1, 3, or 4 channels, got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains NaN or Inf")

    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.max(image)) if image.size else 0.0
        upper = 1.0 if maximum <= 1.0 + 1e-6 else 255.0
        image = np.clip(image, 0.0, upper)
        if upper == 1.0:
            image = image * 255.0
    image = image.astype(np.uint8, copy=False)

    if image.shape[-1] == 1:
        result = Image.fromarray(image[..., 0], mode="L").convert("RGB")
    else:
        mode = "RGB" if image.shape[-1] == 3 else "RGBA"
        result = Image.fromarray(image, mode=mode).convert("RGB")
    if result.size != (image_size, image_size):
        result = result.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return result


def prepare_vlabench_observation(
    obs: Observation,
    profile: XR1VLABenchProfile,
) -> tuple[list[Any], str, np.ndarray]:
    """Validate and convert one canonical harness observation."""

    images = obs.get("images")
    if not isinstance(images, Mapping):
        raise TypeError("observation 'images' must be a camera-name mapping")
    missing = [key for key in VLABENCH_IMAGE_KEYS if key not in images]
    if missing:
        raise ValueError(f"observation is missing required XR-1 cameras: {missing}")
    pil_images = [_to_pil(images[key], profile.image_size) for key in VLABENCH_IMAGE_KEYS]

    instruction = obs.get("task_description")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("observation 'task_description' must be a non-empty string")

    state = np.asarray(obs.get("state"), dtype=np.float32)
    if state.shape != (7,):
        raise ValueError(f"XR-1 VLABench state must have shape (7,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("XR-1 VLABench state contains NaN or Inf")
    state_padded = np.zeros(profile.state_dim, dtype=np.float32)
    state_padded[:7] = state
    return pil_images, instruction.strip(), state_padded


def build_vlabench_messages(images: list[Any], instruction: str, *, cot: bool) -> list[dict[str, Any]]:
    """Reproduce the checkpoint's view labels and prompt template."""

    if len(images) != 3:
        raise ValueError(f"XR-1 prompt requires exactly 3 images, got {len(images)}")
    suffix = "/cot" if cot else "/no_cot"
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
        {"type": "image", "image": images[0]},
        {"type": "text", "text": "\n# Base View\n"},
        {"type": "image", "image": images[1]},
        {"type": "text", "text": "\n# Left-Wrist View\n"},
        {"type": "image", "image": images[2]},
        {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} {suffix}"},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if not cot:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]})
    return messages


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def normalize_vlabench_actions(decoded: Any, profile: XR1VLABenchProfile) -> np.ndarray:
    """Fail closed on decoded shape and convert the gripper to benchmark convention."""

    chunk = _as_numpy(decoded)
    if chunk.ndim == 3:
        if chunk.shape[0] != 1:
            raise ValueError(f"expected decoded batch size 1, got shape {chunk.shape}")
        chunk = chunk[0]
    elif chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2:
        raise ValueError(f"expected decoded actions with shape [T, D], got {chunk.shape}")
    if chunk.shape[0] < profile.model_action_horizon:
        raise ValueError(
            f"decoded horizon {chunk.shape[0]} is shorter than model horizon {profile.model_action_horizon}"
        )
    if chunk.shape[1] < profile.action_dim:
        raise ValueError(f"decoded action dimension {chunk.shape[1]} is smaller than {profile.action_dim}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("decoded actions contain NaN or Inf")

    actions = chunk[: profile.model_action_horizon, : profile.action_dim].copy()
    actions[:, 6] = np.where(actions[:, 6] >= profile.gripper_threshold, 1.0, -1.0)
    return actions


def plan_vlabench_actions(
    decoded: Any,
    model_state: np.ndarray,
    profile: XR1VLABenchProfile,
) -> np.ndarray:
    """Accumulate decoded deltas into the official fixed absolute-pose plan."""

    state = np.asarray(model_state, dtype=np.float32)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError(f"model_state must contain seven finite values, got shape {state.shape}")
    deltas = normalize_vlabench_actions(decoded, profile)
    current = state.copy()
    current[:3] += np.asarray(profile.position_offset, dtype=np.float32)
    planned = np.empty_like(deltas)
    for index, delta in enumerate(deltas):
        current[:6] += delta[:6]
        current[3:6] = (current[3:6] + np.pi) % (2.0 * np.pi) - np.pi
        planned[index, :6] = current[:6]
        planned[index, 6] = delta[6]
    return planned


class XiaomiRobotics1VLABenchServer(PredictModelServer):
    """Direct Hugging Face XR-1 inference for the VLABench embodiment."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        revision: str = DEFAULT_MODEL_REVISION,
        *,
        robot_type: str = "vlabench_choice",
        state_dim: int = 60,
        action_dim: int = 7,
        model_action_horizon: int = 10,
        chunk_size: int = 5,
        image_size: int = 480,
        cot: bool = False,
        request_seed: int = 42,
        gripper_threshold: float = 0.2,
        position_offset: tuple[float, float, float] = (0.0, -0.4, 0.78),
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        self.profile = XR1VLABenchProfile(
            robot_type=robot_type,
            state_dim=state_dim,
            action_dim=action_dim,
            model_action_horizon=model_action_horizon,
            execute_horizon=chunk_size,
            image_size=image_size,
            cot=cot,
            request_seed=request_seed,
            gripper_threshold=gripper_threshold,
            position_offset=position_offset,
        )
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.revision = revision

        import torch
        from transformers import AutoModel, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("XR-1 evaluation requires CUDA, but torch.cuda.is_available() is false")
        if not hasattr(torch, dtype):
            raise ValueError(f"unsupported torch dtype: {dtype}")
        self._torch = torch
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)

        logger.info("Loading XR-1 checkpoint %s at revision %s", model_path, revision)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )
        self._model = AutoModel.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            dtype=self._dtype,
        ).to(device=self._device, dtype=self._dtype)
        self._model.eval()
        logger.info("XR-1 checkpoint loaded on %s", self._device)

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

    def _move_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    value = value.to(device=self._device, dtype=self._dtype)
                else:
                    value = value.to(device=self._device)
            moved[key] = value
        return moved

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        del ctx  # XR-1 is stateless across replans; PredictModelServer owns the chunk buffer.
        images, instruction, state = prepare_vlabench_observation(obs, self.profile)
        messages = build_vlabench_messages(images, instruction, cot=self.profile.cot)
        encoded = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state.reshape(1, 1, self.profile.state_dim),
            robot_type=self.profile.robot_type,
        )
        model_inputs = dict(encoded)
        model_inputs["task_id"] = self.profile.robot_type
        model_inputs["seed"] = self.profile.request_seed
        model_inputs = self._move_inputs(model_inputs)

        with self._torch.inference_mode():
            outputs = self._model(**model_inputs)
        if not hasattr(outputs, "actions"):
            raise RuntimeError("XR-1 model output does not contain an 'actions' field")
        raw_actions = outputs.actions
        if hasattr(raw_actions, "detach"):
            raw_actions = raw_actions.detach()
        if hasattr(raw_actions, "cpu"):
            raw_actions = raw_actions.cpu()
        decoded = self._processor.decode_action(raw_actions, robot_type=self.profile.robot_type)
        return {"actions": plan_vlabench_actions(decoded, state[:7], self.profile)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(XiaomiRobotics1VLABenchServer)
