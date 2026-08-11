# Evaluation roadmap

All phases are constrained by [the repository scope](SCOPE.md): this project
evaluates robot policies in simulation and does not train policies, evaluate
video generators or world models, or operate physical robots.

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

## Phase 4: maintained evaluation suites

- Publish versioned benchmark suites with frozen task and episode manifests.
- Add paired policy comparisons, confidence intervals, failure taxonomies, and
  per-capability scorecards.
- Continuously rerun reproduced baselines to detect simulator, dependency, and
  protocol drift.
