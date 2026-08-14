"""Opt-in delivery of complete action chunks to benchmark clients.

The pinned vla-eval server normally buffers a policy's action chunk on the
server and asks the simulator to send another observation for every buffered
action.  Those observations are intentionally ignored by the policy.  This
mixin moves that buffer to a chunk-aware episode runner without changing the
default wire behavior for existing clients.
"""

from __future__ import annotations

from typing import Any

from vla_eval.model_servers.base import SessionContext

ACTION_DELIVERY_PROTOCOL = "sim-benchmarks.client-action-chunk.v1"
ACTION_DELIVERY_REQUEST_KEY = "action_delivery"
ACTION_DELIVERY_RESPONSE_KEY = "action_delivery"


class ClientChunkDeliveryMixin:
    """Negotiate per-session client-side action-chunk execution.

    The client requests :data:`ACTION_DELIVERY_PROTOCOL` in EPISODE_START.
    Sessions that do not request it retain upstream vla-eval's server-side
    buffering exactly.  This keeps the model server compatible with the stock
    runner and makes accidental use of a chunk-aware runner fail closed.
    """

    def _client_chunk_sessions(self) -> set[str]:
        sessions = getattr(self, "_sim_benchmarks_client_chunk_sessions", None)
        if sessions is None:
            sessions = set()
            self._sim_benchmarks_client_chunk_sessions = sessions
        return sessions

    def _uses_client_chunk_delivery(self, ctx: SessionContext) -> bool:
        return ctx.session_id in self._client_chunk_sessions()

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        request = config.get(ACTION_DELIVERY_REQUEST_KEY)
        if request == ACTION_DELIVERY_PROTOCOL:
            self._client_chunk_sessions().add(ctx.session_id)
        else:
            self._client_chunk_sessions().discard(ctx.session_id)
        await super().on_episode_start(config, ctx)  # type: ignore[misc]

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        try:
            await super().on_episode_end(result, ctx)  # type: ignore[misc]
        finally:
            self._client_chunk_sessions().discard(ctx.session_id)

    def _try_serve_from_buffer(self, ctx: SessionContext):
        if self._uses_client_chunk_delivery(ctx):
            return None
        return super()._try_serve_from_buffer(ctx)  # type: ignore[misc]

    async def _process_and_send(self, result: dict[str, Any], ctx: SessionContext) -> None:
        if self._uses_client_chunk_delivery(ctx):
            await ctx.send_action({**result, ACTION_DELIVERY_RESPONSE_KEY: ACTION_DELIVERY_PROTOCOL})
            return
        await super()._process_and_send(result, ctx)  # type: ignore[misc]
