#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks
xr1_python=/data/shared1/envs/xr1/bin/python
vla_eval="${repo_root}/.venv/bin/vla-eval"
benchmark_config=configs/benchmarks/vlabench/xr1_official_resume.yaml
manifest=/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812/resume/completed-55.json
run_root=/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_20260813
result_root=/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_20260813
shard_count=4
eval_id="xr1-full-resume-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${run_root}/server_logs" "${run_root}/shards" "${run_root}/snapshots" "${result_root}"
cd "${repo_root}"

exec 9>"${run_root}/runner.lock"
if ! flock -n 9; then
  echo "another XR-1 resume runner holds ${run_root}/runner.lock" >&2
  exit 1
fi
if find "${result_root}" -name 'recording-*.sqlite' -print -quit | grep -q .; then
  echo "refusing to mix a new run with existing recordings under ${result_root}" >&2
  exit 1
fi

manifest_count=$("${repo_root}/.venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["completed_episode_count"])' "${manifest}")
if [[ "${manifest_count}" -ne 55 ]]; then
  echo "expected a 55-episode resume manifest, found ${manifest_count}" >&2
  exit 1
fi
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
cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
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

for shard in $(seq 0 $((shard_count - 1))); do
  port=$((18000 + shard))
  "${xr1_python}" src/sim_benchmarks/model_servers/xiaomi_robotics_1.py \
    --config configs/model_servers/xiaomi_robotics_1/vlabench_gb10.yaml \
    --port "${port}" \
    >"${run_root}/server_logs/server${shard}.log" 2>&1 &
  server_pids+=("$!")
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/config" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pids[-1]}" 2>/dev/null; then
      echo "XR-1 server ${shard} exited during startup; see server log" >&2
      exit 1
    fi
    sleep 5
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "XR-1 server ${shard} did not become ready" >&2
    exit 1
  fi
done

for shard in $(seq 0 $((shard_count - 1))); do
  port=$((18000 + shard))
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
  for shard in $(seq 0 $((shard_count - 1))); do
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

exit_code=0
for pid in "${client_pids[@]}"; do
  if ! wait "${pid}"; then
    exit_code=1
  fi
done
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
monitor_pid=
snapshot_recordings

printf 'eval_id=%s\nexit_code=%s\n' "${eval_id}" "${exit_code}" >"${run_root}/runner-summary.txt"
exit "${exit_code}"
