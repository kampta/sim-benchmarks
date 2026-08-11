# Evaluation roadmap

## Phase 0: trustworthy foundation

- Pin `vla-eval` v0.4.0 and all benchmark sources by commit.
- Treat `integrated` and `score reproduced` as different states.
- Reproduce high-value existing adapters before advertising support.
- Record exact episode manifests, seeds, checkpoints, dataset versions, and
  container digests.

## Phase 1: simulation portfolio

Add adapters in this order:

1. Colosseum V2: controlled visual, language, and action perturbations.
2. VLA-Arena: safety, distractors, extrapolation, and long-horizon tasks.
3. DOMINO: dynamic manipulation and real-time responsiveness.
4. EBench: mobile, dexterous, and long-horizon capability diagnosis.

An adapter advances through `planned`, `scaffolded`, `smoke-tested`,
`reproduced`, and `supported`. Only the last two may appear in reported
benchmark tables.

## Phase 2: strict interfaces

- Negotiate a versioned observation/action contract before an episode.
- Fail on incompatible control modes, frames, units, dimensions, history, or
  action horizons.
- Route every non-identity conversion through a named, versioned
  `EmbodimentAdapter`.
- Preserve WebSocket + MessagePack for rollout traffic.

## Phase 3: control plane

Expose job-level REST and MCP operations for listing, launching, monitoring,
cancelling, and comparing evaluations. MCP is not used for per-step images or
actions.

## Phase 4: additional runners

- Physical runner: safety supervisor, watchdog, heartbeats, rate limits,
  intervention logging, and emergency stop outside the policy process.
- World-model runner: generated-future and actionability contracts for
  WorldArena and RoboWM-Bench.

