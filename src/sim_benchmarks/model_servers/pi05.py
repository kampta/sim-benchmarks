"""Pinned π0.5 launcher for the vla-eval LeRobot bridge.

The released LIBERO processor points at a gated PaliGemma repository. This
launcher changes only that processor lookup to use Physical Intelligence's
anonymously published, checksum-pinned tokenizer. Policy inference, checkpoint
loading, normalization, camera mapping, action chunking, and transport remain
the pinned LeRobot v0.6.0 and vla-eval v0.4.0 implementations.
"""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any

import lerobot.policies as _policies
from lerobot.policies.factory import make_pre_post_processors as _make_pre_post_processors

from sim_benchmarks.model_servers.client_chunking import ClientChunkDeliveryMixin

_TOKENIZER_ROOT = Path("/data/shared1/models/paligemma-tokenizer-official")
_TOKENIZER_MODEL_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"


def _validate_tokenizer() -> None:
    tokenizer_model = _TOKENIZER_ROOT / "tokenizer.model"
    if not tokenizer_model.is_file():
        raise FileNotFoundError(f"missing pinned PaliGemma tokenizer: {tokenizer_model}")
    digest = hashlib.sha256(tokenizer_model.read_bytes()).hexdigest()
    if digest != _TOKENIZER_MODEL_SHA256:
        raise ValueError(f"PaliGemma tokenizer checksum mismatch: expected {_TOKENIZER_MODEL_SHA256}, got {digest}")


def make_pre_post_processors(*args: Any, **kwargs: Any) -> Any:
    """Build the saved processor pipeline with the pinned local tokenizer."""

    _validate_tokenizer()
    overrides = dict(kwargs.get("preprocessor_overrides") or {})
    overrides["tokenizer_processor"] = {"tokenizer_name": str(_TOKENIZER_ROOT)}
    kwargs["preprocessor_overrides"] = overrides
    return _make_pre_post_processors(*args, **kwargs)


_policies.make_pre_post_processors = make_pre_post_processors

LeRobotModelServer = import_module("vla_eval.model_servers.lerobot").LeRobotModelServer
run_server = import_module("vla_eval.model_servers.serve").run_server


class Pi05ModelServer(ClientChunkDeliveryMixin, LeRobotModelServer):
    """LeRobot bridge with opt-in transport and masked-camera optimizations."""

    def __init__(
        self,
        *,
        compact_masked_cameras: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if compact_masked_cameras:
            # The released LIBERO config declares ``empty_camera_0``. LeRobot
            # creates an all--1 image for it, sets its attention mask to false,
            # but still runs SigLIP and carries the masked image tokens through
            # the prefix transformer. Removing a block that is false in the
            # padding mask preserves the valid-token computation while avoiding
            # that vision and attention work.
            config = self._policy.config
            empty_keys = [key for key in config.input_features if key.startswith("observation.images.empty_camera_")]
            if not empty_keys:
                raise ValueError("compact_masked_cameras requested, but the checkpoint declares no empty cameras")
            for key in empty_keys:
                config.input_features.pop(key)
            config.empty_cameras = 0
            self._expected_image_keys = [key for key in self._expected_image_keys if key not in empty_keys]


if __name__ == "__main__":
    run_server(Pi05ModelServer)
