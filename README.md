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
released checkpoints cover VLABench, RoboCasa, and RoboCasa365.

The first end-to-end target is the released XR-1 VLABench checkpoint because
VLABench already exists in the pinned harness. This is not merely a model-server
addition: the v0.4.0 VLABench adapter exposes only one camera and success, while
XR-1's published protocol selects three named views from the four-camera raw
observation, uses robot state and official tracks, and reports
intention/progress metrics. We will fix and validate that benchmark adapter
before claiming reproduction.

The intended order is:

1. add a safe XR-1 `PredictModelServer` using WebSocket/MessagePack;
2. upgrade and reproduce the official VLABench evaluation protocol;
3. reproduce XR-1 on the existing RoboCasa adapter;
4. add/backport RoboCasa365 and reproduce its official `target50` protocol;
5. consider RoboDojo after its environment path and a suitable public
   checkpoint are available.

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

The active implementation queue and acceptance criteria live in
[TODO.md](TODO.md). A checked box means the artifact exists; benchmark support
still follows the stricter state definitions in the reproducibility policy.
