"""Chunk-aware, local-transport fast path for the pinned vla-eval CLI.

Run this module in place of ``vla_eval.cli.main``.  All upstream command-line
arguments remain unchanged::

    python -m sim_benchmarks.fast_eval run --config ... --no-docker --yes

The runner negotiates complete action chunks in EPISODE_START, applies every
physical action locally, checks termination after every action, and requests a
new observation only at the next policy inference boundary.  Images default to
lossless raw transport because localhost PNG encoding is slower than copying
the 256x256 arrays.  Set ``SIM_BENCHMARKS_IMAGE_FORMAT=png`` when bandwidth is
more important than local CPU time.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import numpy as np
from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.base import EpisodeRunner
from vla_eval.types import Action, EpisodeResult, Task

from sim_benchmarks.model_servers.client_chunking import (
    ACTION_DELIVERY_PROTOCOL,
    ACTION_DELIVERY_REQUEST_KEY,
    ACTION_DELIVERY_RESPONSE_KEY,
)


class ChunkSyncEpisodeRunner(EpisodeRunner):
    """Synchronous runner that executes negotiated action chunks client-side."""

    def __init__(self, *, require_action_chunks: bool = True) -> None:
        self.require_action_chunks = require_action_chunks

    @staticmethod
    def _actions_from_response(response: Action) -> list[Action]:
        protocol = response.get(ACTION_DELIVERY_RESPONSE_KEY)
        if protocol != ACTION_DELIVERY_PROTOCOL:
            return [response]

        raw_actions = response.get("actions")
        actions = np.asarray(raw_actions)
        if actions.ndim != 2 or len(actions) == 0:
            raise RuntimeError(
                f"{ACTION_DELIVERY_PROTOCOL} requires a non-empty [horizon, action_dim] array; "
                f"got shape {actions.shape}"
            )
        return [{**response, "actions": action} for action in actions]

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        await benchmark.start_episode(task, recorder=recorder)
        obs_dict = await benchmark.get_observation()

        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        ep_payload: dict[str, Any] = {
            "task": task_info,
            ACTION_DELIVERY_REQUEST_KEY: ACTION_DELIVERY_PROTOCOL,
        }
        if recorder is not None and recorder.is_active:
            ep_payload["recording"] = {
                "sid": recorder.sid,
                "eid": recorder.eid,
                "eval_id": recorder.eval_id,
                "db_path": recorder.db_path,
            }
        await conn.start_episode(ep_payload)

        action_steps = 0
        inference_steps = itertools.count()
        for _ in inference_steps:
            response = await conn.act(obs_dict)
            negotiated = response.get(ACTION_DELIVERY_RESPONSE_KEY) == ACTION_DELIVERY_PROTOCOL
            if self.require_action_chunks and not negotiated:
                raise RuntimeError(
                    "model server did not negotiate client action chunks; use a ClientChunkDeliveryMixin "
                    "server or the stock vla-eval runner"
                )

            actions = self._actions_from_response(response)
            for action in actions:
                await benchmark.apply_action(action)
                action_steps += 1
                if await benchmark.is_done() or (max_steps is not None and action_steps >= max_steps):
                    break

            if await benchmark.is_done() or (max_steps is not None and action_steps >= max_steps):
                break
            obs_dict = await benchmark.get_observation()

        elapsed = await benchmark.get_time()
        metrics = await benchmark.get_result()
        episode_result: EpisodeResult = {
            "metrics": metrics,
            "steps": action_steps,
            "elapsed_sec": round(elapsed, 3),
        }
        await conn.end_episode(episode_result)
        return episode_result


def main() -> None:
    image_format = os.environ.get("SIM_BENCHMARKS_IMAGE_FORMAT", "raw")
    if image_format not in {"raw", "jpeg", "png"}:
        raise ValueError("SIM_BENCHMARKS_IMAGE_FORMAT must be raw, jpeg, or png")

    from vla_eval.protocol.numpy_codec import set_image_format

    set_image_format(image_format)

    # vla-eval v0.4.0 constructs SyncEpisodeRunner directly.  Replace that
    # constructor before delegating to its CLI so config parsing, recording,
    # sharding, reporting, and error isolation remain pinned upstream behavior.
    import vla_eval.orchestrator as orchestrator_module

    orchestrator_module.SyncEpisodeRunner = ChunkSyncEpisodeRunner

    from vla_eval.cli.main import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
