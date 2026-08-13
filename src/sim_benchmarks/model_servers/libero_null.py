"""Null LIBERO policy used only to calibrate evaluator overhead."""

from __future__ import annotations

from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation


class LIBERONullModelServer(PredictModelServer):
    """Return ten no-motion, open-gripper actions per policy query.

    This is an infrastructure probe, not a model baseline: its purpose is to
    measure simulator, serialization, WebSocket, and recording overhead without
    conflating those costs with GPU inference.
    """

    def __init__(self, chunk_size: int = 10, **kwargs: Any) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)
        self._actions = np.zeros((chunk_size, 7), dtype=np.float32)
        self._actions[:, -1] = -1.0

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "agentview": IMAGE_RGB,
            "wrist": IMAGE_RGB,
            "state": STATE_EEF_POS_AA_GRIP,
            "language": LANGUAGE,
        }

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return {"actions": self._actions.copy()}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(LIBERONullModelServer)
