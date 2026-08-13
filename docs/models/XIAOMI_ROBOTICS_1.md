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
# Build the native simulator image after extracting and verifying the pinned
# asset layer. The named context must contain an `assets/` directory.
docker build --progress=plain \
  --build-context vlabench_assets=/data/shared1/cache/sim_benchmarks/vlabench/extracted \
  -f docker/Dockerfile.vlabench \
  -t sim-benchmarks/vlabench:cf588fe .

# GPU process in the dedicated Mamba environment.
HF_HOME=/data/shared1/cache/huggingface/kampta \
TORCH_HOME=/data/shared1/cache/torch/kampta \
PYTHONUNBUFFERED=1 \
  /data/shared1/envs/xr1/bin/python \
  src/sim_benchmarks/model_servers/xiaomi_robotics_1.py \
  --config configs/model_servers/xiaomi_robotics_1/vlabench_gb10.yaml

# VLABench environment process. This remains a one-episode smoke test, not a score.
.venv/bin/vla-eval run \
  --config configs/benchmarks/vlabench/xr1_smoke.yaml \
  --record-video
```

The environment image is built from `docker/Dockerfile.vlabench`; both VLABench
source and the RRT dependency are pinned by commit. Its object and scene assets
come from the single architecture-neutral layer of AllenAI's immutable released
VLABench image. The image manifest and layer digests are pinned in the benchmark
manifest. The layer must be SHA-256 verified before extraction and supplied as
the `vlabench_assets` BuildKit named context, so the native build neither pulls
the AMD64 runtime nor depends on mutable Google Drive archives.

Verify and extract the pinned layer before the Docker build:

```bash
/data/shared1/envs/xr1/bin/python -m sim_benchmarks.provenance.vlabench_assets \
  /data/shared1/cache/sim_benchmarks/vlabench/sha256-9aaf46e753389b05e3b4fc3cdc62bc3ba3d577d5d1e1bbbfa0145d9361657b17.tar.gz \
  --manifest src/sim_benchmarks/manifests/benchmarks/vlabench.json

mkdir -p /data/shared1/cache/sim_benchmarks/vlabench/extracted
tar --extract --gzip --no-same-owner --no-same-permissions --strip-components=3 \
  --file=/data/shared1/cache/sim_benchmarks/vlabench/sha256-9aaf46e753389b05e3b4fc3cdc62bc3ba3d577d5d1e1bbbfa0145d9361657b17.tar.gz \
  --directory=/data/shared1/cache/sim_benchmarks/vlabench/extracted \
  app/VLABench/VLABench/assets
```

Before starting the model server, verify the downloaded shards against the
sizes and SHA-256 digests frozen in the model manifest:

```bash
HF_HOME=/data/shared1/cache/huggingface/kampta \
TORCH_HOME=/data/shared1/cache/torch/kampta \
  mamba run -p /data/shared1/envs/xr1 python -m sim_benchmarks.provenance.artifacts \
  /data/shared1/models/Xiaomi-Robotics-1-VLABench \
  --manifest src/sim_benchmarks/manifests/models/xiaomi_robotics_1.json \
  --benchmark vlabench
```

### GB10 environment deviation

The current reproduction host is ARM64 with an NVIDIA GB10 (`sm_121`). Xiaomi's
released install command pins Python 3.12, PyTorch 2.8.0/CUDA 12.8,
torchvision 0.23, Transformers 4.57.1, and flash-attn 2.8.3. PyTorch's ARM64
CUDA 12.8 index does not publish the 2.8.0 wheel and flash-attn 2.8.3 does not
provide an `sm_121` binary. The dedicated `/data/shared1/envs/xr1` Mamba
environment therefore uses Python 3.10.20 with NVIDIA's ARM64 PyTorch
2.9.1/CUDA 13.0 and torchvision 0.24.1 wheels, while preserving Transformers
4.57.1 exactly. The installed PyTorch wheel is tagged
`cp310-cp310-manylinux_2_28_aarch64`.

For the official run, flash-attn 2.8.3 is compiled locally with CUDA 13 for
`sm_120`; that kernel executes on the forward-compatible `sm_121` GB10. Its
BF16/head-dimension-128 kernel and a full XR-1 inference fixture were validated
before evaluation. Against eager attention on the same fixed observation, it
preserved all discrete gripper actions, had mean absolute action difference
`1.48e-4` (maximum `1.14e-3`), and reduced isolated warm inference from 9.63 to
8.89 seconds. Build the cached ARM64 wheel with:

```bash
MAX_JOBS=2 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS=120 \
  /data/shared1/envs/xr1/bin/pip wheel \
  --cache-dir /data/shared1/cache/torch/kampta/pip \
  --no-build-isolation --no-deps flash-attn==2.8.3 \
  -w /data/shared1/cache/torch/kampta/wheels
```

The resulting wheel used for this run has SHA-256
`a86284cb5241ff0f5d8802e36052459905a8b2f1e25ba177e12b7aaa52c522a4`.
This remains a declared platform port, not an exact dependency reproduction;
all benchmark and model protocol pins remain unchanged.
The input environment specification is committed at
`environments/xr1-gb10.yaml`; create it under the shared environment and package
cache roots with:

```bash
CONDA_PKGS_DIRS=/data/shared1/conda_pkgs \
  mamba env create -p /data/shared1/envs/xr1 \
  -f environments/xr1-gb10.yaml
```

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

The XR-1 profile closes the model input and action-contract gap. It also reads
the five official pinned track files, flattens their fixed episode configs into
independently shardable harness tasks, preserves per-task episode horizons, and
reports success, intention, and progress. The smoke configuration selects the
first fixed `select_fruit` episode from `track_1_in_distribution`.

### Audit against Xiaomi's evaluator

Xiaomi's VLABench evaluator is available in the pinned source revision
`6bc75afb791a1938750fe5fc0aee2b0f28cf87e2`. The harness adapter was audited
field by field against `eval_vlabench/main.py`, `eval_vlabench/dispatch.py`, and
`eval_vlabench/merge_results.py`. Both paths use raw cameras 2, 0, and 3 as the
Ego, Base, and Left-Wrist views; 480-pixel bilinear resizing; the same prompt;
robot-relative XYZ plus world-frame Euler and gripper state padded to 60
dimensions; a deterministic request seed of 42; a raw `[10, 60]` action; the
first seven decoded dimensions; cumulative world-frame targets; five executed
actions per replan; a 0.2 gripper threshold; one MuJoCo substep; the second
environment reset; task-specific episode horizons; and SR/IS/PS collection.

The integration intentionally differs only in transport and artifact handling:
WebSocket/MessagePack replaces trusted-local pickle/TCP, recordings are stored
transactionally in SQLite, and videos can be disabled. Those differences do not
alter observations, actions, simulator transitions, or metrics. The pinned
VLABench revision is also Xiaomi's current upstream revision
`cf588fe60c0c7282174fe979f5913170cfe69017`.

The pinned smoke completed on 2026-08-12 with native ARM64 image
`sha256:6e6a6a5865678bc3bf2f38148324cd54181cf38493875a5ac5c3d3020118f646`.
All three checkpoint shards and the simulator asset layer matched their frozen
sizes and SHA-256 digests. The episode ran for its 200-step horizon, produced
201 recorded 480×480 frames, and wrote an integrity-checked SQLite recording.
The policy did not solve this single episode: SR 0.0, intention score 1.0, and
progress score 0.5. This validates the full evaluation path but is not a
benchmark score or evidence for the published 59.1% result. The complete
five-track reproduction remains outstanding.

After the smoke test, run the complete 2,460-episode protocol with:

```bash
.venv/bin/vla-eval run --config configs/benchmarks/vlabench/xr1_official.yaml
```

The harness aggregate is episode-weighted, whereas Xiaomi's published report
macro-averages all task entries (important because track 2 has 460 episodes).
For a sharded run, build the comparable report directly from the recording
databases. This command checks SQLite integrity, rejects runtime-error rows,
requires a contiguous stored step trace for every episode, and matches every
episode index and configuration digest against the copied pinned track files:

```bash
.venv/bin/python -m sim_benchmarks.reporting.vlabench \
  --db /data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812/shard*/recording-*.sqlite \
  --track-dir /data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812/provenance/tracks \
  --output /data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812/xr1_official_macro.json
```

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
