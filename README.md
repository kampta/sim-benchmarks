# sim-benchmarks

An upstream-tracking extension of
[AllenAI's VLA evaluation harness](https://github.com/allenai/vla-evaluation-harness)
for current, non-saturated robot-policy evaluation.

The project deliberately separates:

- the rollout data plane: WebSocket + MessagePack, inherited from `vla-eval`;
- the control plane: CLI first, with REST/MCP orchestration added only after the
  simulation protocol is stable;
- benchmark adapters: isolated and pinned per upstream repository;
- embodiment adapters: explicit, versioned observation/action transformations;
- evaluation evidence: an adapter is not marked supported until a published
  reference result has been reproduced.

## Upstream pin

`vla-eval` is pinned to tag `v0.4.0`, commit
`2680ab2fafe981c2dba63c6c1a4e7bb4415dbb56`. See [UPSTREAM.md](UPSTREAM.md).

## Initial benchmark scope

The first new integrations are Colosseum V2, VLA-Arena, DOMINO, and EBench.
Before claiming those or existing harness adapters as supported, follow the
reproduction targets in `manifests/reproduction/targets.json`.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install uv
.venv/bin/uv sync --dev
.venv/bin/uv run pytest
```

This repository currently contains the pinned foundation, strict interface
contract, embodiment-adapter API, benchmark source manifests, dataset catalog,
and reproduction plan. Environment-specific rollout adapters are added only
after their reference protocol is captured and validated.
