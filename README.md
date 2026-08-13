# sim-benchmarks

A reproducible evaluation suite for vision-language-action (VLA) policies,
built as a thin extension of
[AllenAI's VLA evaluation harness](https://github.com/allenai/vla-evaluation-harness).
The goal is a useful, non-saturated benchmark portfolio—not a larger list of
adapters with unknown correctness.

## Scope

This repository has one job: run robot policies against simulation benchmarks
and produce comparable, reproducible evaluation results. In scope are benchmark
adapters, policy inference adapters, embodiment mappings, evaluation suites,
metrics, rollout transport, provenance, and reporting.

It is not a policy-training repository, a dataset survey, a video-tokenizer or
generative-model testbed, a world-model benchmark, or a physical-robot runtime.
Training datasets and normalization statistics may be referenced only to record
policy provenance and prevent evaluation leakage. See [the scope contract](docs/SCOPE.md).

## Status

This repository is at the foundation stage. It contains:

- an exact pin to `vla-eval` v0.4.0;
- strict, fail-closed observation/action interface negotiation;
- an explicit embodiment-adapter API;
- pinned source manifests for the first new benchmarks;
- benchmark-data provenance and a reference-result reproduction plan.

No benchmark is called supported until a released policy reproduces a published
reference result under the same task, seed, horizon, camera, and metric
protocol. See [the reproducibility policy](docs/REPRODUCIBILITY.md).

## Architecture

```text
benchmark environment                 policy process
┌──────────────────────┐              ┌──────────────────────┐
│ task/episode manifest│              │ checkpoint + processor│
│ simulator + metrics  │── WS/MsgPack─│ PredictModelServer    │
│ embodiment adapter   │              │ action chunks         │
└──────────────────────┘              └──────────────────────┘
            │
            └── CLI now; REST/MCP later for jobs and reports only
```

Images and actions stay on the harness's WebSocket/MessagePack rollout data
plane. REST and MCP are reserved for slower control-plane operations such as
listing suites, launching jobs, monitoring runs, and comparing reports. Every
non-identity observation or action conversion belongs to a named, versioned
embodiment adapter.

## Pinned base

`vla-eval` is pinned to tag `v0.4.0`, commit
`2680ab2fafe981c2dba63c6c1a4e7bb4415dbb56`. The pin has 18 benchmark
packages, including RoboCasa and VLABench. It does **not** include the newer
RoboCasa365 or RoboDojo adapters currently on harness `main`; those must be
reviewed and backported explicitly. See [UPSTREAM.md](UPSTREAM.md).

## Benchmark portfolio

The first new benchmark adapters, in planned implementation order, are:

1. **Colosseum V2** — controlled visual, language, and action perturbations.
2. **VLA-Arena** — safety, distractors, extrapolation, and long-horizon tasks.
3. **DOMINO** — dynamic manipulation, collision-aware scoring, and latency.
4. **EBench** — mobile, dexterous, and long-horizon capability diagnosis.

RoboCasa, RoboCasa365, VLABench, RoboDojo, RoboTwin, MIKASA-Robo, RoboMME,
MolmoSpaces, RoboCerebra, and others remain valuable suite candidates. We will
prefer generalization, perturbation, dynamics, memory, safety, and long-horizon
coverage over saturated headline scores. LIBERO can remain a compatibility
smoke test, but is not a primary differentiating benchmark.

Source-specific integration contracts are in [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md),
and the longer sequence is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Next baseline: Xiaomi-Robotics-1

[Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) (XR-1)
is a policy family, not a benchmark. It is a useful first baseline because its
released checkpoints cover VLABench, RoboCasa, and RoboCasa365. Xiaomi also
publishes a RoboDojo headline result, but does not list a corresponding public
checkpoint.

No XR-1 benchmark score has been reproduced by this repository yet. The values
below are upstream reference targets, not our measurements:

| Benchmark | Xiaomi-published reference | Reproduced here | Current state |
|---|---:|---:|---|
| VLABench | 59.1% headline SR | — | One pinned 200-step XR-1 episode now passes end-to-end evaluation/recording; the full five-track score remains outstanding. |
| RoboCasa | 74.5% headline; detailed evaluation guide reports 74.2% over 24 × 100 episodes | — | Pinned harness adapter must be audited against RoboCasa v0.2. |
| RoboCasa365 | 57.4% headline; detailed guide reports 1,432/2,500 = 57.28% | — | Adapter must be reviewed/backported and the `target50` protocol frozen. |
| RoboDojo | 13.93% headline | — | No XR-1 checkpoint is published; no adapter is present in pinned v0.4.0. |

The headline values come from Xiaomi's
[pinned benchmark table](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/6bc75afb791a1938750fe5fc0aee2b0f28cf87e2/README.md#-benchmark).
The more precise RoboCasa and RoboCasa365 values come from the pinned
[RoboCasa](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/6bc75afb791a1938750fe5fc0aee2b0f28cf87e2/eval_robocasa/README.md#results)
and
[RoboCasa365](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/6bc75afb791a1938750fe5fc0aee2b0f28cf87e2/eval_robocasa365/README.md#results)
evaluation guides. We will preserve both reported values until an exact
episode-level reproduction explains the rounding/configuration difference.
Xiaomi does not report XR-1 results for Colosseum V2, VLA-Arena, DOMINO,
EBench, or the other portfolio candidates; those entries are **not evaluated**,
not zero scores.

The first end-to-end target is the released XR-1 VLABench checkpoint because
VLABench already exists in the pinned harness. This is not merely a model-server
addition: the v0.4.0 VLABench adapter exposes only one camera and success, while
XR-1's published protocol selects three named views from the four-camera raw
observation, uses robot state and official tracks, and reports
intention/progress metrics. We will fix and validate that benchmark adapter
before claiming reproduction.

Active XR-1 TODOs, in execution order:

- [x] Pin the XR-1 source, checkpoint revisions, dependencies, and upstream
  reference scores.
- [x] Add a fail-closed XR-1 `PredictModelServer` over WebSocket/MessagePack and
  unit-test the VLABench observation/action contract.
- [x] Finish the pinned VLABench container/assets and checkpoint transfer, verify
  their official digests, and pass one complete
  reset → inference → action → step → recording smoke test.
- [x] Add the five official VLABench tracks, deterministic episode manifests,
  task-specific horizons, and SR/IS/PS aggregation.
- [x] Audit the adapter against Xiaomi's pinned VLABench evaluator at
  `6bc75afb791a1938750fe5fc0aee2b0f28cf87e2`, including camera order, prompt,
  state frame, action accumulation, five-step replanning, gripper command,
  reset behavior, horizons, metrics, and task-macro aggregation.
- [ ] Run the complete five-track suite and reproduce the published 59.1% SR.
- [ ] Audit RoboCasa v0.2 cameras, controller, horizons, seeds, and action
  semantics; reproduce the published 74.5% headline / 74.2% detailed result.
- [ ] Add or backport RoboCasa365, freeze its official `target50` suite, and
  reproduce 1,432 successes from 2,500 episodes within declared tolerance.
- [ ] Add RoboDojo only when a usable XR-1 checkpoint and an exact reproducible
  simulation protocol are available.
- [ ] Attempt XR-1 on other portfolio benchmarks only after defining a valid
  embodiment adapter and declaring whether benchmark data entered training.

We will not reuse XR-1's example pickle-over-TCP transport. Exact source,
checkpoint, dependency, and protocol details are recorded in
[the XR-1 integration note](docs/models/XIAOMI_ROBOTICS_1.md).

## Evaluation data integrity

The machine-readable catalog at
[`src/sim_benchmarks/manifests/datasets/catalog.json`](src/sim_benchmarks/manifests/datasets/catalog.json)
tracks only data released with benchmarks in this evaluation portfolio. It uses
three explicit data boundaries:

- `train_id`: official demonstrations or generated data allowed for training;
- `eval_id`: unseen episodes from the same task distribution;
- `eval_ood`: held-out objects, layouts, scenes, language, dynamics, or task
  compositions.

Evaluation episode manifests must be excluded from training by identity and
content hash even when a policy was intentionally trained in distribution.
This repository records that provenance; it does not download broad pretraining
corpora or implement policy training.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install uv
.venv/bin/uv sync --dev
.venv/bin/uv run pytest
.venv/bin/uv run ruff check .
```

The README tracks the active baseline checklist; the complete implementation
queue and acceptance criteria live in [TODO.md](TODO.md). A checked box means
the artifact exists; benchmark support still follows the stricter state
definitions in the reproducibility policy.
