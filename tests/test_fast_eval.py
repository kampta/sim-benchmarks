from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext

from sim_benchmarks.fast_eval import ChunkSyncEpisodeRunner
from sim_benchmarks.model_servers.client_chunking import (
    ACTION_DELIVERY_PROTOCOL,
    ACTION_DELIVERY_REQUEST_KEY,
    ACTION_DELIVERY_RESPONSE_KEY,
    ClientChunkDeliveryMixin,
)


class _Benchmark:
    def __init__(self, done_after: int) -> None:
        self.done_after = done_after
        self.applied: list[np.ndarray] = []
        self.observation_calls = 0

    async def start_episode(self, task: dict[str, Any], recorder: Any = None) -> None:
        return None

    async def get_observation(self) -> dict[str, Any]:
        self.observation_calls += 1
        return {"observation_index": self.observation_calls}

    async def apply_action(self, action: dict[str, Any]) -> None:
        self.applied.append(np.asarray(action["actions"]))

    async def is_done(self) -> bool:
        return len(self.applied) >= self.done_after

    async def get_time(self) -> float:
        return 1.2345

    async def get_result(self) -> dict[str, Any]:
        return {"success": await self.is_done()}


class _Connection:
    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon
        self.start_payload: dict[str, Any] | None = None
        self.ended: dict[str, Any] | None = None

    async def start_episode(self, payload: dict[str, Any]) -> None:
        self.start_payload = payload

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "actions": np.arange(self.horizon * 7, dtype=np.float32).reshape(self.horizon, 7),
            ACTION_DELIVERY_RESPONSE_KEY: ACTION_DELIVERY_PROTOCOL,
        }

    async def end_episode(self, result: dict[str, Any]) -> None:
        self.ended = result


class _BaseChunkServer:
    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        return None

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        return None

    def _try_serve_from_buffer(self, ctx: SessionContext) -> np.ndarray:
        return np.ones(7, dtype=np.float32)

    async def _process_and_send(self, result: dict[str, Any], ctx: SessionContext) -> None:
        await ctx.send_action({"base": True})


class _ChunkServer(ClientChunkDeliveryMixin, _BaseChunkServer):
    pass


def test_chunk_runner_checks_done_after_each_physical_action() -> None:
    benchmark = _Benchmark(done_after=3)
    connection = _Connection(horizon=5)
    runner = ChunkSyncEpisodeRunner()

    result = asyncio.run(runner.run_episode(benchmark, {"name": "task"}, connection, max_steps=20))

    assert len(benchmark.applied) == 3
    assert benchmark.observation_calls == 1
    assert result == {"metrics": {"success": True}, "steps": 3, "elapsed_sec": 1.234}
    assert connection.start_payload is not None
    assert connection.start_payload[ACTION_DELIVERY_REQUEST_KEY] == ACTION_DELIVERY_PROTOCOL
    assert connection.ended == result


def test_chunk_runner_preserves_control_step_limit() -> None:
    benchmark = _Benchmark(done_after=100)
    connection = _Connection(horizon=10)
    runner = ChunkSyncEpisodeRunner()

    result = asyncio.run(runner.run_episode(benchmark, {"name": "task"}, connection, max_steps=7))

    assert len(benchmark.applied) == 7
    assert result["steps"] == 7
    assert result["metrics"] == {"success": False}


def test_chunk_server_negotiates_per_session_and_preserves_stock_fallback() -> None:
    async def exercise() -> None:
        sent: list[dict[str, Any]] = []
        server = _ChunkServer()
        ctx = SessionContext("session", "episode")

        async def send_action(action: dict[str, Any]) -> None:
            sent.append(action)

        ctx._send_action_fn = send_action

        await server.on_episode_start({}, ctx)
        np.testing.assert_array_equal(server._try_serve_from_buffer(ctx), np.ones(7, dtype=np.float32))
        await server._process_and_send({"actions": np.zeros((2, 7), dtype=np.float32)}, ctx)
        assert sent.pop() == {"base": True}

        await server.on_episode_start({ACTION_DELIVERY_REQUEST_KEY: ACTION_DELIVERY_PROTOCOL}, ctx)
        assert server._try_serve_from_buffer(ctx) is None
        chunk = np.zeros((2, 7), dtype=np.float32)
        await server._process_and_send({"actions": chunk}, ctx)
        assert sent[-1][ACTION_DELIVERY_RESPONSE_KEY] == ACTION_DELIVERY_PROTOCOL
        np.testing.assert_array_equal(sent[-1]["actions"], chunk)

        await server.on_episode_end({}, ctx)
        np.testing.assert_array_equal(server._try_serve_from_buffer(ctx), np.ones(7, dtype=np.float32))

    asyncio.run(exercise())
