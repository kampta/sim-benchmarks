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
| VLABench | 59.1% headline SR | — | Full pinned run active with exact-identity recovery checkpoints; the current validated count is recorded in the latest checkpoint manifest. |
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
- [x] Preserve and validate the first 55 scored episodes, then add fail-closed
  resume filtering keyed by `(track, task, episode index, config SHA-256)` so
  no completed rollout is repeated or confused with an identical config in a
  different track.
- [x] Start the remaining 2,405 episodes across four isolated model servers
  and four five-core simulator shards, with periodic SQLite backups and a
  finalizer that preserves simulator errors, retries only their exact pinned
  identities up to three times, and publishes only after exact 2,460-identity
  coverage passes.
- [x] Add read-only recovery checkpoints for an interrupted multi-day run:
  snapshot every live SQLite database, retain raw simulator errors, validate
  derived scored copies, and emit the exact completed-identity manifest needed
  to continue only the missing episodes. A detached monitor creates and
  verifies these checkpoints every six hours by default and once more when the
  runner exits. The active long run uses an hourly interval.
- [x] Parameterize the four-shard runner for crash continuation. It validates
  any supplied completed manifest against all pinned track identities before
  model startup and requires fresh run/result roots, allowing a checkpoint to
  resume only its missing episodes without editing the benchmark config.
- [x] Make continuation supervisors fail fast: detect any client exit with
  `wait -n`, require three consecutive model-server health failures before
  interruption, stop sibling shards through the cleanup trap, and let the
  recovery monitor snapshot the resulting closed databases.
- [x] Exercise crash continuation after a native simulator watchdog terminated
  one first-generation shard: freeze 101 valid identities (55 original plus 46
  resumed), retain the raw failed attempt separately, and restart all 2,359
  pending identities across four shards. Parameterize auditing, checkpointing,
  retry mounts, and final aggregation so the prior clean checkpoint is included
  exactly once in the eventual 2,460-identity report.
- [x] Match Xiaomi/VLABench's released handling of unstable simulator episodes:
  require exact attempted-identity coverage, preserve every raw exception and
  partial step trace, report the upstream-compatible available-case task macro,
  and include a conservative zero-imputed sensitivity result. This was needed
  after one pinned `select_book` identity reproduced `mjWARN_BADQACC` at the
  same step in two independent generations with fixed request seed 42.
- [x] Measure protocol-equivalent serving topologies on GB10 instead of assuming
  replica scaling. Four replicas/four simulators averaged 5.18 seconds per
  simulator step; two replicas/four simulators averaged 5.14 seconds per step
  with half the model memory; one serialized replica/four simulators was slower
  and produced sustained request backpressure. Continue with two replicas.
- [x] Validate the accumulated multi-generation union independently of its
  continuation manifests: the original 55 identities exactly match their four
  recording databases, and clean continuation databases add unique valid
  identities with no duplicates or unexpected config hashes. Preserve the
  continuously refreshed audit and source checksums with the run artifacts.
- [x] Prepare Spark2 for an exact distributed continuation: synchronize the
  pinned code, mamba runtime, track files, checkpoint, and byte-identical
  VLABench image; add global shard offsets and distributed finalization; and
  install a fail-closed cutover that waits for the existing Spark2 workload to
  disappear and memory to recover before checkpointing Spark1 and launching
  non-overlapping shards 0–3 and 4–7.
- [x] Make dual-host recovery and finalization generation-aware: snapshot the
  four-shard original/prior generations separately from an eight-shard resumed
  generation, audit and checkpoint all shared recordings on Spark1, wait for
  both runner sessions, and automatically run the exact-coverage finalizer.
- [x] Keep two-node recovery copies: hourly checkpoints are checksum-verified
  under Spark2-backed `/data/shared2`, then staged, re-verified, and atomically
  mirrored to Spark1-local `/data/shared1` before being marked latest.
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

## Fast evaluator validation: pi0.5

Before adding more baselines, we are using the public
[`lerobot/pi05_libero_finetuned_v044`](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)
checkpoint to measure and tune end-to-end evaluator throughput. This is a
LIBERO-trained checkpoint, so LIBERO-Object is a valid native-policy acceptance
test. It remains a compatibility and performance gate—not a primary scientific
benchmark, and not evidence that pi0.5 supports another benchmark's embodiment.

The checkpoint, LeRobot v0.6.0 source, normalization files, and LIBERO container
are pinned by revision or digest in the
[`pi05_libero` model manifest](src/sim_benchmarks/manifests/models/pi05_libero.json).
The checkpoint's saved processor names a gated Hugging Face PaliGemma repo. The
launcher instead uses Physical Intelligence's anonymously published tokenizer,
pinned locally by SHA-256, so evaluation does not depend on gated access or the
network.
The server uses the same WebSocket/MessagePack data plane as every other policy.
It returns action chunks and the harness executes ten actions before replanning.

The released LIBERO image is AMD64-only. On this ARM64 GB10, use the dedicated
native mamba environment at `/data/shared1/envs/libero` with LIBERO commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01` and pass `--no-docker`. Do not use
QEMU emulation for scored runs. A null-policy calibration completed all ten
LIBERO-Object tasks (one full-horizon episode each) in 21 seconds wall time with
ten shards: 2,800 steps, zero runtime errors, and ten valid SQLite recordings.
This measures evaluator capacity only; its 0% policy score is not a baseline.
The scored pi0.5 Object run will be compared against LeRobot's 99.0% target,
[AllenAI's 100/100 reproduction](https://github.com/allenai/vla-evaluation-harness/blob/2680ab2fafe981c2dba63c6c1a4e7bb4415dbb56/docs/reproductions/lerobot.md),
and the official OpenPI checkpoint's 98.2%.

The exact AllenAI protocol now reproduces at **100/100**: ten tasks x ten
episodes, seed 7, 13,457 simulator steps, zero errors, and 349 seconds rollout
wall time after 145 seconds of model startup. All ten SQLite databases pass
`PRAGMA quick_check` and contain one step row per recorded simulator step.
Warm compiled throughput is 10/10 episodes in 39 seconds. Results are preserved
under `/data/shared2/user/kampta/logs/sim_benchmarks/libero/pi05_reproduction`.

Active pi0.5/evaluator TODOs:

- [x] Pin the LeRobot source, checkpoint files, normalization statistics, model
  license, benchmark image, camera mapping, and action-chunk protocol.
- [x] Create the shared `pi05_v060` mamba environment with the exact LeRobot
  v0.6.0 source used by AllenAI's reproduction, without modifying the harness.
- [x] Replace the AMD64-only benchmark image on GB10 with a pinned native ARM64
  LIBERO mamba runtime; validate reset, EGL render, observation construction,
  stepping, WebSocket transport, and SQLite persistence.
- [x] Run the null-policy ten-shard capacity gate: 10 episodes and 2,800 steps
  in 21 seconds wall time, with zero errors and valid per-shard databases.
- [x] Re-run the LeRobot v0.6.0 acceptance gates: the model loader reports all
  keys loaded; the smoke episode succeeded in 129 steps; and the ten-task gate
  scored 10/10 with ten valid databases. Its 93-second wall time includes
  contention from another GPU workload, so it is not the final speed number.
- [x] Validate compiled inference on GB10 with CUDA 13's system `ptxas`; the
  Triton-bundled CUDA 12.8 assembler cannot target `sm_121a`.
- [x] Reproduce AllenAI's 100/100 protocol (10 tasks x 10 episodes): 100/100,
  zero errors, 13,457 steps, and 349 seconds rollout wall time.
- [ ] Select the fastest stable shard count from measured results on this GB10;
  do not assume that more simulator containers are always faster.
- [ ] Run the standard 500-episode LIBERO-Object suite with the validated
  ten-shard native runner and compare it with the published targets.
- [ ] Turn the validated launcher/config/result layout into the template for
  subsequent baselines; add pi0.5 to other benchmarks only with a declared,
  tested embodiment adapter or a benchmark-native checkpoint.

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
