#!/usr/bin/env bash
set -euo pipefail

repo_root=${XR1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
xr1_python=/data/shared1/envs/xr1/bin/python
vla_eval="${repo_root}/.venv/bin/vla-eval"
benchmark_config=configs/benchmarks/vlabench/xr1_official_resume.yaml
manifest=${XR1_COMPLETED_MANIFEST:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812/resume/completed-55.json}
run_root=${XR1_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_20260813}
result_root=${XR1_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_20260813}
track_dir=/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812/provenance/tracks
shard_count=${XR1_SHARD_COUNT:-4}
local_shard_count=${XR1_LOCAL_SHARD_COUNT:-${shard_count}}
shard_offset=${XR1_SHARD_OFFSET:-0}
server_count=${XR1_SERVER_COUNT:-${local_shard_count}}
base_port=${XR1_BASE_PORT:-18000}
eval_prefix=${XR1_EVAL_PREFIX:-xr1-full-resume}
eval_id="${eval_prefix}-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${run_root}/server_logs" "${run_root}/shards" "${run_root}/snapshots" "${result_root}"
cd "${repo_root}"

if ! [[ "${shard_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "XR1_SHARD_COUNT must be a positive integer" >&2
  exit 2
fi
if ! [[ "${local_shard_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "XR1_LOCAL_SHARD_COUNT must be a positive integer" >&2
  exit 2
fi
if ! [[ "${shard_offset}" =~ ^[0-9]+$ ]] \
  || [[ $((shard_offset + local_shard_count)) -gt "${shard_count}" ]]; then
  echo "XR1_SHARD_OFFSET + XR1_LOCAL_SHARD_COUNT must fit within XR1_SHARD_COUNT" >&2
  exit 2
fi
if ! [[ "${server_count}" =~ ^[1-9][0-9]*$ ]] || [[ "${server_count}" -gt "${local_shard_count}" ]]; then
  echo "XR1_SERVER_COUNT must be a positive integer no greater than XR1_LOCAL_SHARD_COUNT" >&2
  exit 2
fi
printf 'global_shards=%s\nlocal_shards=%s\nshard_offset=%s\nservers=%s\nbase_port=%s\n' \
  "${shard_count}" "${local_shard_count}" "${shard_offset}" "${server_count}" "${base_port}" \
  >"${run_root}/topology.txt"

exec 9>"${run_root}/runner.lock"
if ! flock -n 9; then
  echo "another XR-1 resume runner holds ${run_root}/runner.lock" >&2
  exit 1
fi
if find "${result_root}" -name 'recording-*.sqlite' -print -quit | grep -q .; then
  echo "refusing to mix a new run with existing recordings under ${result_root}" >&2
  exit 1
fi

manifest_count=$(PYTHONPATH="${repo_root}/src" "${repo_root}/.venv/bin/python" - "${manifest}" "${track_dir}" <<'PY'
import sys
from pathlib import Path

from sim_benchmarks.benchmarks.vlabench_xr1 import (
    OFFICIAL_TRACKS,
    load_completed_episode_identities,
    load_vlabench_episode_tasks,
    task_episode_identity,
)

manifest = Path(sys.argv[1]).resolve()
track_dir = Path(sys.argv[2]).resolve()
completed = load_completed_episode_identities(manifest)
expected = {
    task_episode_identity(task)
    for suite in OFFICIAL_TRACKS
    for task in load_vlabench_episode_tasks(track_dir / f"{suite}.json", suite=suite, episode_limit=50)
}
unexpected = sorted(completed - expected)
if unexpected:
    raise ValueError(f"resume manifest contains identities absent from pinned tracks: {unexpected[:3]}")
if len(completed) >= len(expected):
    raise ValueError(f"resume manifest leaves no work: completed={len(completed)} official={len(expected)}")
print(len(completed))
PY
)
export XR1_COMPLETED_MANIFEST="$(realpath "${manifest}")"
printf 'validated completed manifest identities=%s pending=%s\n' \
  "${manifest_count}" "$((2460 - manifest_count))" >"${run_root}/resume-validation.txt"
printf '%s\n' "${eval_id}" >"${run_root}/eval_id.txt"

export HF_HOME=/data/shared1/cache/huggingface
export HF_HUB_CACHE=/data/shared1/cache/huggingface/hub
export HF_MODULES_CACHE=/data/shared1/cache/huggingface/kampta/modules
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/shared1/cache/torch
export TORCH_EXTENSIONS_DIR=/data/shared1/cache/torch/extensions/kampta/xr1
export TORCHINDUCTOR_CACHE_DIR=/data/shared1/cache/torch/inductor/xr1_cuda130
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PYTHONPATH="${repo_root}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${HF_MODULES_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

server_pids=()
client_pids=()
monitor_pid=
watchdog_pid=
cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -n "${watchdog_pid}" ]]; then
    kill "${watchdog_pid}" 2>/dev/null || true
  fi
  for pid in "${client_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${server_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for server in $(seq 0 $((server_count - 1))); do
  port=$((base_port + server))
  "${xr1_python}" src/sim_benchmarks/model_servers/xiaomi_robotics_1.py \
    --config configs/model_servers/xiaomi_robotics_1/vlabench_gb10.yaml \
    --port "${port}" \
    >"${run_root}/server_logs/server${server}.log" 2>&1 &
  server_pids+=("$!")
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/config" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pids[-1]}" 2>/dev/null; then
      echo "XR-1 server ${server} exited during startup; see server log" >&2
      exit 1
    fi
    sleep 5
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "XR-1 server ${server} did not become ready" >&2
    exit 1
  fi
done

for local_shard in $(seq 0 $((local_shard_count - 1))); do
  shard=$((shard_offset + local_shard))
  port=$((base_port + (local_shard % server_count)))
  shard_result="${result_root}/shard${shard}"
  mkdir -p "${shard_result}"
  "${vla_eval}" run \
    --config "${benchmark_config}" \
    --server-url "ws://127.0.0.1:${port}" \
    --output-dir "${shard_result}" \
    --eval-id "${eval_id}" \
    --shard-id "${shard}" --num-shards "${shard_count}" \
    --cpus 0-19 --yes \
    >"${run_root}/shards/shard${shard}.log" 2>&1 &
  client_pids+=("$!")
done

snapshot_recordings() {
  local timestamp database shard snapshot_dir episodes successes steps
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  snapshot_dir="${run_root}/snapshots/latest"
  mkdir -p "${snapshot_dir}"
  : >"${run_root}/status.txt"
  for local_shard in $(seq 0 $((local_shard_count - 1))); do
    shard=$((shard_offset + local_shard))
    database="${result_root}/shard${shard}/recording-${eval_id}.sqlite"
    if [[ -f "${database}" ]]; then
      sqlite3 "${database}" ".backup '${snapshot_dir}/shard${shard}.next.sqlite'"
      mv -f "${snapshot_dir}/shard${shard}.next.sqlite" "${snapshot_dir}/shard${shard}.sqlite"
      read -r episodes successes steps < <(
        sqlite3 -separator ' ' "${database}" \
          "SELECT COUNT(*), COALESCE(SUM(status = 'success'), 0), COALESCE(SUM(steps), 0) FROM episode_results;"
      )
      printf '%s shard=%s episodes=%s successes=%s steps=%s\n' \
        "${timestamp}" "${shard}" "${episodes}" "${successes}" "${steps}" >>"${run_root}/status.txt"
    fi
  done
}

(
  while :; do
    snapshot_recordings || true
    any_alive=0
    for pid in "${client_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    [[ "${any_alive}" -eq 1 ]] || break
    sleep 600
  done
) >"${run_root}/snapshot.log" 2>&1 &
monitor_pid=$!

# The harness isolates episode errors by design, so a dead model server would
# otherwise let its client record errors for every remaining identity. Require
# three consecutive failed health passes before stopping the whole supervisor;
# the cleanup trap closes every client/server and the external checkpoint
# monitor then snapshots the closed databases for exact continuation.
supervisor_pid=${BASHPID}
(
  consecutive_failures=0
  while :; do
    any_client_alive=0
    for pid in "${client_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_client_alive=1
      fi
    done
    [[ "${any_client_alive}" -eq 1 ]] || exit 0

    healthy=1
    for server in $(seq 0 $((server_count - 1))); do
      port=$((base_port + server))
      if ! kill -0 "${server_pids[server]}" 2>/dev/null \
        || ! curl --max-time 5 -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        healthy=0
      fi
    done
    if [[ "${healthy}" -eq 1 ]]; then
      consecutive_failures=0
    else
      consecutive_failures=$((consecutive_failures + 1))
      printf '%s unhealthy_pass=%s/3\n' \
        "$(date -Is)" "${consecutive_failures}" >"${run_root}/watchdog-status.txt"
    fi
    if [[ "${consecutive_failures}" -ge 3 ]]; then
      printf '%s fatal: model-server health failed three consecutive passes\n' \
        "$(date -Is)" >"${run_root}/watchdog-status.txt"
      kill -TERM "${supervisor_pid}"
      exit 1
    fi
    sleep 30
  done
) >"${run_root}/watchdog.log" 2>&1 &
watchdog_pid=$!

exit_code=0
remaining_pids=("${client_pids[@]}")
while [[ "${#remaining_pids[@]}" -gt 0 ]]; do
  finished_pid=
  client_rc=0
  wait -n -p finished_pid "${remaining_pids[@]}" || client_rc=$?
  if [[ -z "${finished_pid}" ]]; then
    exit_code=1
    printf '%s wait -n returned without a finished client PID\n' \
      "$(date -Is)" >"${run_root}/client-failure.txt"
    break
  fi
  next_pids=()
  for pid in "${remaining_pids[@]}"; do
    if [[ "${pid}" != "${finished_pid}" ]]; then
      next_pids+=("${pid}")
    fi
  done
  remaining_pids=("${next_pids[@]}")
  if [[ "${client_rc}" -ne 0 ]]; then
    exit_code=1
    printf '%s client_pid=%s exit_code=%s; stopping remaining shards\n' \
      "$(date -Is)" "${finished_pid}" "${client_rc}" >"${run_root}/client-failure.txt"
    for pid in "${remaining_pids[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    for pid in "${remaining_pids[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
    break
  fi
done
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
monitor_pid=
kill "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true
watchdog_pid=
snapshot_recordings

printf 'eval_id=%s\nexit_code=%s\n' "${eval_id}" "${exit_code}" >"${run_root}/runner-summary.txt"
exit "${exit_code}"
