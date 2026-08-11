# Xiaomi-Robotics-1 integration

## Classification and pin

Xiaomi-Robotics-1 (XR-1) is a VLA model family and baseline, not an evaluation
benchmark. Its official repository is pinned here at commit
`6bc75afb791a1938750fe5fc0aee2b0f28cf87e2` (Apache-2.0 source license). Do not
build reproducibility claims from a moving branch.

Released task-fine-tuned checkpoints:

- `XiaomiRobotics/Xiaomi-Robotics-1-VLABench`
- `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa`
- `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365`

The source pin and checkpoint repository revisions are machine-readable in
`src/sim_benchmarks/manifests/models/xiaomi_robotics_1.json`. Weight and dataset
terms are recorded from their model cards and must still be checked
independently of the repository's source-code license before use. Downloaded
artifact digests are captured when a reproduction environment is assembled.

## Integration decision

Implement XR-1 directly as a `vla_eval.model_servers.predict.PredictModelServer`.
Load the checkpoint's Hugging Face `AutoModel` and `AutoProcessor` in-process,
then use the existing harness WebSocket/MessagePack data plane. Do not wrap or
expose the repository's example length-prefixed TCP/pickle transport: pickle is
unsafe across a trust boundary and the second protocol would bypass harness
negotiation, recording, and chunk handling.

The server must declare and validate:

- required named RGB camera streams and their deterministic order;
- language field and robot/embodiment selector;
- robot-state dimensions, history, padding, units, and normalization;
- raw model output shape versus executable action dimensions;
- delta-position and delta-Euler frames/units plus gripper convention;
- predicted, retained, and executed action-chunk lengths.

Any benchmark-specific conversion belongs in an embodiment profile/adapter,
not in an implicit camera-name or array-slice convention inside the server.

## Implemented scaffold

The initial VLABench slice now includes:

- `XiaomiRobotics1VLABenchServer`, with the checkpoint and remote-code revision
  pinned, optional heavy dependencies isolated in PEP 723 metadata, and direct
  WebSocket/MessagePack serving through `PredictModelServer`;
- `XR1VLABenchBenchmark`, which selects and names `front`, `base`, and
  `left_wrist` from the four raw cameras and converts end-effector state to the
  checkpoint's robot-relative position/Euler/gripper representation;
- strict validation of images, state, decoded batch/horizon/action dimensions,
  finite values, and gripper semantics;
- accumulation of decoded deltas into the same fixed five-step absolute-pose
  plan as Xiaomi's evaluator, avoiding drift from recomputing against measured
  IK state between planned actions;
- model-server and one-episode benchmark smoke configurations;
- no-checkpoint tests for the full preprocessing and action-decoding path.

Start the two processes from the repository root:

```bash
# GPU process; the first run creates the isolated XR-1 environment and downloads weights.
uv run --script src/sim_benchmarks/model_servers/xiaomi_robotics_1.py \
  --config configs/model_servers/xiaomi_robotics_1/vlabench.yaml

# VLABench environment process. This remains a contract smoke test, not a score.
vla-eval run --config configs/benchmarks/vlabench/xr1_smoke.yaml --no-docker
```

The environment command requires VLABench and its assets. A pinned container is
still needed before the smoke test is portable.

## Compatibility and order

| Evaluation | In pinned harness v0.4.0 | Released XR-1 checkpoint | Work required |
|---|---:|---:|---|
| VLABench | Yes | Yes | Upgrade observations, tracks, metrics, and chunk protocol; then reproduce. |
| RoboCasa | Yes | Yes | Audit v0.2 protocol and add an XR-1 embodiment profile; then reproduce. |
| RoboCasa365 | No | Yes | Review/backport or implement an adapter, freeze `target50`, then reproduce. |
| RoboDojo | No | No checkpoint listed above | Defer until environment and baseline artifacts support a reproducible run. |

VLABench is first because it joins an existing pinned benchmark adapter to a
released checkpoint. RoboCasa365 is the first XR-1-related *benchmark addition*.

## VLABench protocol gap

The v0.4.0 harness adapter currently sends one `primary` camera, language, and a
7D action, and records success only. Xiaomi's released evaluation path requires
four raw camera images, selects three of them for its prompt, adds end-effector
state, predicts a 10-step action chunk from a 60-dimensional raw action
representation, executes the first seven action dimensions, and replans after
five actions. Its reports include success, intention, and progress metrics
across official tracks.

The new XR-1 observation profile closes the model input and action-contract gap.
The smoke configuration is still not a reproduction because the official track
enumeration, deterministic episode manifests, intention/progress metrics, and a
pinned VLABench runtime remain outstanding.

## RoboCasa365 reproduction target

The official XR-1 evaluation configuration to preserve uses the `target50`
suite (five tasks), 50 trials per task, seed 7, four observations sampled at
history interval 2, image crop ratio 0.95, and 16 executed actions per policy
query. These fields belong in a committed episode/config manifest and must not
be inferred from defaults at runtime.

## Dependencies and acceptance

The pinned source specifies Python 3.12, PyTorch 2.8.0, torchvision 0.23,
Transformers 4.57.1, and flash-attn 2.8.3 for its released evaluation path.
Keep these optional model-server dependencies out of the lightweight core
environment and capture the GPU/container digest in results.

XR-1 support requires all of the following:

1. protocol unit tests and an environment/model smoke test;
2. an immutable checkpoint revision/digest;
3. an exact benchmark task/episode manifest;
4. published-protocol parity for observations, actions, horizons, and metrics;
5. an episode-level reference reproduction within a declared tolerance.

Official sources:

- <https://github.com/XiaomiRobotics/Xiaomi-Robotics-1>
- <https://huggingface.co/XiaomiRobotics>
