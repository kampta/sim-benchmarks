#!/usr/bin/env bash
set -euo pipefail

repo_root=${XR1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
bundle=${XR1_MIGRATION_BUNDLE:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_migration_bundle_20260813T234213Z}
env_prefix=${XR1_ENV_PREFIX:-/data/shared1/envs/xr1-rtx-x86_64}
xr1_python=${XR1_PYTHON:-${env_prefix}/bin/python}
vla_eval=${XR1_VLA_EVAL:-${env_prefix}/bin/vla-eval}
model_config=${XR1_MODEL_CONFIG:-configs/model_servers/xiaomi_robotics_1/vlabench_rtx.yaml}
benchmark_config=${XR1_PERF_CONFIG:-configs/benchmarks/vlabench/xr1_rtx_perf.yaml}
image=${XR1_VLABENCH_IMAGE:-sim-benchmarks/vlabench:cf588fe-cuda128-amd64}
model_gpu=${XR1_MODEL_GPU_ID:-0}
sim_gpu=${XR1_SIM_GPU_ID:-${model_gpu}}
cpu_spec=${XR1_CPUS:-0-15}
base_port=${XR1_BASE_PORT:-18000}
minimum_speedup=${XR1_MINIMUM_SPEEDUP:-2.0}
expected_gb10_seconds_per_step=5.672262096774194
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${XR1_PERF_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_perf_${timestamp}}
result_root=${XR1_PERF_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/xr1_rtx_perf_${timestamp}}
ready_file=${XR1_READY_FILE:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_host_ready_$(hostname -s)}
gate_file=${XR1_THROUGHPUT_GATE_FILE:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_throughput_gate_$(hostname -s).json}
eval_id=xr1-rtx-perf-${timestamp}

cd "${repo_root}"
if [[ "$(uname -m)" != x86_64 ]]; then
  echo "XR-1 RTX benchmark requires x86_64; found $(uname -m)" >&2
  exit 2
fi
for required in "${ready_file}" "${xr1_python}" "${vla_eval}" "${model_config}" "${benchmark_config}" \
  "${bundle}/BUNDLE_SHA256SUMS"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing prepared-host input: ${required}" >&2
    exit 2
  fi
done
grep -qx 'ready=1' "${ready_file}" || { echo "invalid RTX READY file" >&2; exit 2; }
ready_image=$(awk -F= '$1 == "image" {print substr($0, index($0, "=") + 1)}' "${ready_file}")
ready_image_id=$(awk -F= '$1 == "image_id" {print substr($0, index($0, "=") + 1)}' "${ready_file}")
ready_commit=$(awk -F= '$1 == "git_commit" {print substr($0, index($0, "=") + 1)}' "${ready_file}")
if [[ "${ready_image}" != "${image}" ]]; then
  echo "READY image ${ready_image} does not match requested image ${image}" >&2
  exit 2
fi
current_image_id=$(docker image inspect --format '{{.Id}}' "${image}")
if [[ "${ready_image_id}" != "${current_image_id}" || "${ready_commit}" != "$(git rev-parse HEAD)" ]]; then
  echo "RTX READY file is stale for the current image or git commit; rerun host preparation" >&2
  exit 2
fi
if [[ -e "${run_root}" ]] || [[ -e "${result_root}" ]]; then
  echo "refusing to reuse throughput run/result roots: ${run_root} ${result_root}" >&2
  exit 2
fi
mkdir -p "${run_root}" "${result_root}"
(cd "${bundle}" && sha256sum -c BUNDLE_SHA256SUMS) >"${run_root}/bundle-verification.log"

read -r gb10_episodes gb10_steps gb10_elapsed gb10_seconds_per_step < <(
  "${xr1_python}" - "${bundle}" "${expected_gb10_seconds_per_step}" <<'PY'
import glob
import json
import sqlite3
import sys

bundle = sys.argv[1]
expected_seconds_per_step = float(sys.argv[2])
rows = {}
for database in glob.glob(f"{bundle}/original/shard*/recording-*.sqlite"):
    connection = sqlite3.connect(database)
    for status, steps, elapsed, context_json in connection.execute(
        "SELECT status, steps, elapsed_sec, context FROM episode_results WHERE task_name = 'select_painting'"
    ):
        context = json.loads(context_json)
        if context.get("suite") != "track_1_in_distribution" or context.get("episode_index") not in {0, 1, 2}:
            continue
        identity = int(context["episode_index"])
        assert identity not in rows, identity
        rows[identity] = (status, int(steps), float(elapsed))
    connection.close()
assert set(rows) == {0, 1, 2}, rows
assert all(status != "error" for status, _, _ in rows.values())
steps = sum(row[1] for row in rows.values())
elapsed = sum(row[2] for row in rows.values())
seconds_per_step = elapsed / steps
assert steps == 248, steps
assert abs(seconds_per_step - expected_seconds_per_step) < 1e-12, seconds_per_step
print(len(rows), steps, elapsed, seconds_per_step)
PY
)
export HF_HOME=/data/shared1/cache/huggingface
export HF_HUB_CACHE=/data/shared1/cache/huggingface/hub
export HF_MODULES_CACHE=/data/shared1/cache/huggingface/kampta/modules
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/shared1/cache/torch
export TORCH_EXTENSIONS_DIR=/data/shared1/cache/torch/extensions/kampta/xr1-rtx-x86_64
export TORCHINDUCTOR_CACHE_DIR=/data/shared1/cache/torch/inductor/xr1-rtx-x86_64
export PYTHONPATH="${repo_root}/src"
export PYTHONUNBUFFERED=1
export XR1_VLABENCH_IMAGE="${image}"
export XR1_SIM_GPU_IDS="${sim_gpu}"
mkdir -p "${HF_MODULES_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

server_pid=
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${model_gpu}" "${xr1_python}" \
  src/sim_benchmarks/model_servers/xiaomi_robotics_1.py \
  --config "${model_config}" --port "${base_port}" \
  >"${run_root}/server.log" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${base_port}/config" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "XR-1 server exited during startup; see ${run_root}/server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" -ne 1 ]]; then
  echo "XR-1 server did not become ready" >&2
  exit 1
fi

"${vla_eval}" run --config "${benchmark_config}" \
  --server-url "ws://127.0.0.1:${base_port}" \
  --output-dir "${result_root}" --eval-id "${eval_id}" \
  --cpus "${cpu_spec}" --gpus "${sim_gpu}" --yes \
  >"${run_root}/eval.log" 2>&1
database=${result_root}/recording-${eval_id}.sqlite
if [[ ! -f "${database}" ]]; then
  echo "throughput benchmark did not produce ${database}" >&2
  exit 1
fi
read -r episodes errors steps elapsed < <(
  sqlite3 -separator ' ' "${database}" \
    "SELECT COUNT(*), COALESCE(SUM(status = 'error'), 0), COALESCE(SUM(steps), 0), COALESCE(SUM(elapsed_sec), 0) FROM episode_results;"
)
if [[ "${episodes}" -ne 3 || "${errors}" -ne 0 || "${steps}" -le 0 ]]; then
  echo "invalid throughput sample: episodes=${episodes} errors=${errors} steps=${steps}" >&2
  exit 1
fi
image_id=${current_image_id}

"${xr1_python}" - "${gate_file}" "${database}" "${episodes}" "${steps}" "${elapsed}" \
  "${gb10_episodes}" "${gb10_steps}" "${gb10_elapsed}" "${gb10_seconds_per_step}" \
  "${minimum_speedup}" "${image}" "${image_id}" "$(git rev-parse HEAD)" <<'PY'
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

gate_path, database_path = map(Path, sys.argv[1:3])
episodes, steps = map(int, sys.argv[3:5])
elapsed = float(sys.argv[5])
gb10_episodes, gb10_steps = map(int, sys.argv[6:8])
gb10_elapsed, gb10_seconds_per_step, minimum_speedup = map(float, sys.argv[8:11])
image, image_id, commit = sys.argv[11:14]
seconds_per_step = elapsed / steps
speedup = gb10_seconds_per_step / seconds_per_step
material_speedup = speedup >= minimum_speedup
database_sha256 = hashlib.sha256(database_path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "purpose": "XR-1 RTX continuation throughput gate; never include in benchmark metrics",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "git_commit": commit,
    "image": image,
    "image_id": image_id,
    "sample": "track_1_in_distribution/select_painting episodes 0-2",
    "episodes": episodes,
    "steps": steps,
    "elapsed_sec": elapsed,
    "seconds_per_step": seconds_per_step,
    "gb10_episodes_same_sample": gb10_episodes,
    "gb10_steps_same_sample": gb10_steps,
    "gb10_elapsed_sec_same_sample": gb10_elapsed,
    "gb10_seconds_per_step_same_sample": gb10_seconds_per_step,
    "speedup_over_gb10": speedup,
    "minimum_required_speedup": minimum_speedup,
    "material_speedup": material_speedup,
    "recording_database": str(database_path),
    "recording_database_sha256": database_sha256,
}
gate_path.parent.mkdir(parents=True, exist_ok=True)
gate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if not material_speedup:
    raise SystemExit(1)
PY

cp "${gate_file}" "${run_root}/throughput-gate.json"
sha256sum "${database}" "${run_root}/throughput-gate.json" >"${run_root}/SHA256SUMS"
printf 'RTX throughput gate passed: %s\n' "${gate_file}"
