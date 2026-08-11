# Implementation queue

Items are ordered by dependency. An adapter progresses through `planned`,
`scaffolded`, `smoke-tested`, `reproduced`, and `supported`; checking off code
creation alone does not advance it to reproduced.

## Repository handoff

- [x] Pin `vla-eval` v0.4.0 by full commit SHA.
- [x] Add strict interface contracts and an embodiment-adapter boundary.
- [x] Add first-wave benchmark, dataset, and reproduction manifests.
- [x] Document architecture, support policy, and the initial baseline.
- [x] Constrain the repository to simulation policy evaluation; keep training,
  tokenizer/generative-model, world-model, and physical-robot work elsewhere.
- [x] Create an authenticated writable GitHub repository/remote and push this branch.
- [ ] Add CI for unit tests, Ruff, manifest validation, and pin-drift detection.

## XR-1 milestone 1: VLABench

- [x] Pin Xiaomi-Robotics-1 source revision, license, released checkpoints, and
  official evaluation dependencies in a model manifest.
- [x] Add an XR-1 `PredictModelServer` that loads the Hugging Face
  `AutoModel`/`AutoProcessor` with `trust_remote_code=True` and declares its
  complete observation/action interface.
- [x] Add a minimal no-checkpoint unit test for input validation, image ordering,
  state padding, action decoding, and fail-closed shape handling.
- [x] Package XR-1 runtime dependencies separately: Python 3.12, PyTorch 2.8.0,
  torchvision 0.23, Transformers 4.57.1, and flash-attn 2.8.3.
- [x] Never expose the upstream example pickle-over-TCP server; serve inference
  through the harness WebSocket/MessagePack protocol.
- [x] Extend VLABench observations from one camera to the three named XR-1 views
  selected from its four raw cameras, with deterministic ordering.
- [x] Expose the end-effector/robot state required by the XR-1 processor and
  verify its padding and normalization semantics.
- [ ] Add the official VLABench tracks and deterministic episode manifests.
- [ ] Preserve success rate, intention score, and progress score instead of
  reporting success alone.
- [x] Match the released policy's 10-action prediction / 5-action execution
  behavior through explicit action-chunk configuration.
- [ ] Smoke-test reset, observation serialization, one inference, action decode,
  stepping, termination, and recording.
- [ ] Reproduce the released `XiaomiRobotics/Xiaomi-Robotics-1-VLABench`
  checkpoint against the official task protocol and record confidence intervals.

## XR-1 milestone 2: RoboCasa

- [ ] Freeze the upstream RoboCasa v0.2 task list, 100 episodes per task, seeds,
  controller, cameras, and horizons used by Xiaomi's published evaluation.
- [ ] Add the RoboCasa embodiment profile and verify delta-pose, Euler rotation,
  gripper sign, action scaling, and action-chunk semantics.
- [ ] Audit the pinned harness adapter against that protocol and fix differences
  without silently changing its compatibility mode.
- [ ] Reproduce `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa` and store the exact
  checkpoint digest, episode manifest, logs, and aggregate/per-task scores.

## XR-1 milestone 3: RoboCasa365

- [ ] Review the newer harness adapter and upstream RoboCasa365 implementation;
  backport only after recording source revisions and behavioral differences.
- [ ] Implement the official `target50` suite: five tasks, 50 trials per task,
  seed 7, four-frame observation history at interval 2, crop ratio 0.95, and
  16 executed actions per query.
- [ ] Add train/evaluation manifest collision checks for RoboCasa365 data.
- [ ] Reproduce `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` before advertising
  adapter support.

## Existing high-value adapters

- [ ] Reproduce RoboMME with its released baseline and memory protocol.
- [ ] Reproduce MolmoSpaces with the released MolmoBot policy.
- [ ] Reproduce RoboTwin with a released policy and canonical episode manifest.
- [ ] Audit and reproduce RoboCasa and VLABench as described above.
- [ ] Upgrade BEHAVIOR-1K to its current protocol before reproduction.
- [ ] Keep LIBERO-family runs as regression/compatibility checks, not primary
  model-ranking evidence.

## First-wave new adapters

- [ ] Colosseum V2: perturbation-stratified adapter and ACT/LeRobot references.
- [ ] VLA-Arena: capability/safety metrics and OpenPI/OpenVLA references.
- [ ] DOMINO: synchronous and latency-aware live runners with PUMA reference.
- [ ] EBench: GenManip service bridge and one published baseline reproduction.

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for per-benchmark exit criteria.

## Protocol and infrastructure

- [ ] Integrate interface negotiation into the harness HELLO/episode-start path;
  mismatched dimensions, frames, units, rates, history, and horizons must fail.
- [ ] Persist negotiated interfaces and embodiment-adapter versions in results.
- [ ] Add deterministic episode-manifest hashing and training/eval collision
  checks.
- [ ] Add container and asset digest capture plus paired statistical reports.
- [ ] Add REST/MCP orchestration only after simulation rollout semantics stabilize.
- [ ] Add versioned cross-benchmark suites and paired statistical reports after
  their constituent adapters are reproduced.
