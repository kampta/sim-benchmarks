#!/usr/bin/env bash
set -euo pipefail

repo_root=${XR1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
bundle=${XR1_MIGRATION_BUNDLE:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_migration_bundle_20260813T234213Z}
model_root=${XR1_MODEL_ROOT:-/data/shared1/models/Xiaomi-Robotics-1-VLABench}
asset_root=${XR1_VLABENCH_ASSET_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/extracted}
asset_archive=${XR1_VLABENCH_ASSET_ARCHIVE:-/data/shared1/cache/sim_benchmarks/vlabench/sha256-9aaf46e753389b05e3b4fc3cdc62bc3ba3d577d5d1e1bbbfa0145d9361657b17.tar.gz}
env_prefix=${XR1_ENV_PREFIX:-/data/shared1/envs/xr1-rtx-x86_64}
image=${XR1_VLABENCH_IMAGE:-sim-benchmarks/vlabench:cf588fe-cuda128-amd64}
gpu_ids=${XR1_GPU_IDS:-0}
minimum_gpu_memory_mib=${XR1_MIN_GPU_MEMORY_MIB:-20000}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
report_root=${XR1_PREPARE_REPORT_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_host_prepare_${timestamp}}
stable_ready=${XR1_STABLE_READY_FILE:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_host_ready_$(hostname -s)}

cd "${repo_root}"
mkdir -p "${report_root}"

if [[ "$(uname -m)" != x86_64 ]]; then
  echo "XR-1 RTX preparation requires x86_64; found $(uname -m)" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "refusing to prepare an RTX host from a dirty worktree" >&2
  exit 2
fi
if [[ -n "${XR1_EXPECTED_GIT_COMMIT:-}" ]] \
  && [[ "$(git rev-parse HEAD)" != "${XR1_EXPECTED_GIT_COMMIT}" ]]; then
  echo "git revision does not match XR1_EXPECTED_GIT_COMMIT" >&2
  exit 2
fi
for executable in docker nvidia-smi sha256sum; do
  command -v "${executable}" >/dev/null || { echo "missing executable: ${executable}" >&2; exit 2; }
done
mamba_bin=${XR1_MAMBA:-$(command -v mamba || true)}
if [[ -z "${mamba_bin}" || ! -x "${mamba_bin}" ]]; then
  echo "mamba is required to create ${env_prefix}" >&2
  exit 2
fi
for required_path in "${bundle}/BUNDLE_SHA256SUMS" "${bundle}/base/completed-manifest.json" \
  "${asset_root}/assets/obj/meshes" "${asset_root}/assets/scenes" "${asset_archive}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "missing required artifact: ${required_path}" >&2
    exit 2
  fi
done

(cd "${bundle}" && sha256sum -c BUNDLE_SHA256SUMS) >"${report_root}/bundle-verification.log"
printf '%s  %s\n' \
  '9aaf46e753389b05e3b4fc3cdc62bc3ba3d577d5d1e1bbbfa0145d9361657b17' \
  "${asset_archive}" | sha256sum -c - >"${report_root}/asset-archive-verification.log"
(
  cd "${model_root}"
  printf '%s  %s\n' \
    'aa20a195689d0066e24c0e95f4147279763523252f3a6c67254f698ce63f01c4' model-00001-of-00003.safetensors \
    'b774074e20cf9e6880f64b137ce668087339014cb346d1f42b702d97edd4ca62' model-00002-of-00003.safetensors \
    'fae7341347fc0a661780e56c684d24464161b918d2430e18c65e237161ee885e' model-00003-of-00003.safetensors \
    | sha256sum -c -
) >"${report_root}/checkpoint-verification.log"

nvidia-smi -q >"${report_root}/nvidia-smi-q.txt"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version \
  --format=csv,noheader,nounits >"${report_root}/gpus.csv"
IFS=, read -r -a requested_gpus <<<"${gpu_ids}"
for gpu_id in "${requested_gpus[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "XR1_GPU_IDS must be a comma-separated list of integer GPU IDs" >&2
    exit 2
  fi
  total_mib=$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  if [[ ! "${total_mib}" =~ ^[0-9]+$ ]] || [[ "${total_mib}" -lt "${minimum_gpu_memory_mib}" ]]; then
    echo "GPU ${gpu_id} has ${total_mib:-unknown} MiB; require at least ${minimum_gpu_memory_mib} MiB" >&2
    exit 2
  fi
done

export CONDA_PKGS_DIRS=/data/shared1/conda_pkgs
export HF_HOME=/data/shared1/cache/huggingface
export TORCH_HOME=/data/shared1/cache/torch
mkdir -p "${CONDA_PKGS_DIRS}" "${HF_HOME}" "${TORCH_HOME}"
if [[ ! -x "${env_prefix}/bin/python" ]]; then
  "${mamba_bin}" env create --prefix "${env_prefix}" --file environments/xr1-rtx-x86_64.yaml
fi
env_python="${env_prefix}/bin/python"
"${env_python}" - <<'PY' >"${report_root}/python-stack.txt"
import flash_attn
import torch
import torchvision
import transformers

assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.23.0", torchvision.__version__
assert transformers.__version__ == "4.57.1", transformers.__version__
assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
assert torch.cuda.is_available()
assert torch.version.cuda == "12.8", torch.version.cuda
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu_{index}={torch.cuda.get_device_name(index)}")
PY
"${env_python}" -m pip check >"${report_root}/pip-check.txt"

DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
  --build-arg 'CUDA_BASE=nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2' \
  --build-context "vlabench_assets=${asset_root}" \
  --file docker/Dockerfile.vlabench --tag "${image}" . \
  >"${report_root}/docker-build.log" 2>&1
docker image inspect "${image}" >"${report_root}/docker-image-inspect.json"
image_arch=$(docker image inspect --format '{{.Architecture}}' "${image}")
image_id=$(docker image inspect --format '{{.Id}}' "${image}")
vlabench_revision=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision.vlabench"}}' "${image}")
if [[ "${image_arch}" != amd64 || "${vlabench_revision}" != cf588fe60c0c7282174fe979f5913170cfe69017 ]]; then
  echo "built image failed architecture/revision validation" >&2
  exit 2
fi
docker run --rm -i --gpus "device=${requested_gpus[0]}" --entrypoint /opt/venv/bin/python "${image}" - <<'PY' \
  >"${report_root}/container-import-smoke.txt"
import dm_control
import mujoco
import VLABench
print(f"mujoco={mujoco.__version__}")
print(f"vlabench={VLABench.__file__}")
PY

PYTHONPATH="${repo_root}/src" "${env_python}" - \
  "${bundle}/base/completed-manifest.json" \
  "${bundle}/original/provenance/tracks" <<'PY' >"${report_root}/resume-manifest-validation.txt"
import sys
from pathlib import Path
from sim_benchmarks.benchmarks.vlabench_xr1 import (
    OFFICIAL_TRACKS,
    load_completed_episode_identities,
    load_vlabench_episode_tasks,
    task_episode_identity,
)

manifest, track_dir = map(Path, sys.argv[1:])
completed = load_completed_episode_identities(manifest)
expected = {
    task_episode_identity(task)
    for suite in OFFICIAL_TRACKS
    for task in load_vlabench_episode_tasks(track_dir / f"{suite}.json", suite=suite, episode_limit=50)
}
assert len(expected) == 2460, len(expected)
assert len(completed) == 150, len(completed)
assert completed <= expected
print("completed=150")
print("pending=2310")
print("suite=2460")
PY

git rev-parse HEAD >"${report_root}/git-commit.txt"
git status --porcelain=v1 >"${report_root}/git-status.txt"
"${mamba_bin}" list --prefix "${env_prefix}" --explicit >"${report_root}/conda-explicit.txt"
"${env_python}" -m pip freeze >"${report_root}/pip-freeze.txt"
sha256sum "${report_root}"/* >"${report_root}/SHA256SUMS"
git_commit=$(git rev-parse HEAD)
printf 'ready=1\nhost=%s\ngit_commit=%s\nimage=%s\nimage_id=%s\nenv=%s\nbundle=%s\nreport=%s\n' \
  "$(hostname)" "${git_commit}" "${image}" "${image_id}" "${env_prefix}" "${bundle}" "${report_root}" \
  | tee "${report_root}/READY" "${stable_ready}"
