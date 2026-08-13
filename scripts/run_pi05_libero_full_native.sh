#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks
pi05_python=/data/shared1/envs/pi05_v060/bin/python
libero_python=/data/shared1/envs/libero/bin/python
episodes_per_task=${PI05_EPISODES_PER_TASK:-50}
case "${episodes_per_task}" in
  10)
    run_name=pi05_reproduction
    benchmark_config=configs/benchmarks/libero/pi05_reproduction.yaml
    expected_episodes=100
    ;;
  50)
    run_name=pi05_full
    benchmark_config=configs/benchmarks/libero/pi05_full.yaml
    expected_episodes=500
    ;;
  *)
    echo "PI05_EPISODES_PER_TASK must be 10 or 50" >&2
    exit 2
    ;;
esac
run_root="/data/shared2/user/kampta/logs/sim_benchmarks/libero/${run_name}"
result_root="/data/shared1/cache/sim_benchmarks/libero/${run_name}"
server_log="${run_root}/server.log"
shard_count=10

mkdir -p "${run_root}/recordings" "${result_root}"
cd "${repo_root}"

export HF_HOME=/data/shared1/cache/huggingface
export HF_HUB_CACHE=/data/shared1/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/shared1/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/data/shared1/cache/torch/inductor/pi05_v060_cuda130
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PYTHONPATH="${repo_root}/src"
export PYTHONUNBUFFERED=1

server_started=$(date +%s)
"${pi05_python}" src/sim_benchmarks/model_servers/pi05.py \
  --port 18010 \
  --args.policy_type pi05 \
  --args.checkpoint /data/shared1/models/pi05_libero_finetuned \
  --args.image_keys='{"agentview":"observation.images.image","wrist":"observation.images.image2"}' \
  --args.state_key observation.state \
  --args.device cuda \
  --args.chunk_size 10 \
  >"${server_log}" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

server_ready=0
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:18010/config >/dev/null 2>&1; then
    server_ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "pi0.5 model server exited during startup; see ${server_log}" >&2
    exit 1
  fi
  sleep 5
done
if [[ "${server_ready}" -ne 1 ]]; then
  echo "pi0.5 model server did not become ready; see ${server_log}" >&2
  exit 1
fi
startup_seconds=$(($(date +%s) - server_started))

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
eval_id="pi05-libero-object-${expected_episodes}-$(date -u +%Y%m%dT%H%M%SZ)"
eval_started=$(date +%s)
pids=()
for shard in $(seq 0 $((shard_count - 1))); do
  shard_result="${result_root}/shard${shard}"
  shard_log="${run_root}/shard${shard}.log"
  mkdir -p "${shard_result}" "${run_root}/recordings/shard${shard}"
  "${libero_python}" -m vla_eval.cli.main run \
    --config "${benchmark_config}" \
    --server-url ws://127.0.0.1:18010 \
    --no-docker --yes \
    --output-dir "${shard_result}" \
    --eval-id "${eval_id}" \
    --shard-id "${shard}" --num-shards "${shard_count}" \
    >"${shard_log}" 2>&1 &
  pids+=("$!")
done

eval_rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    eval_rc=1
  fi
done
wall_seconds=$(($(date +%s) - eval_started))

episodes=0
successes=0
steps=0
for shard in $(seq 0 $((shard_count - 1))); do
  database="${result_root}/shard${shard}/recording-${eval_id}.sqlite"
  [[ "$(sqlite3 "${database}" "PRAGMA quick_check;")" == "ok" ]]
  read -r shard_episodes shard_successes shard_steps < <(
    sqlite3 -separator ' ' "${database}" \
      "SELECT COUNT(*), SUM(status = 'success'), SUM(steps) FROM episode_results;"
  )
  episodes=$((episodes + shard_episodes))
  successes=$((successes + shard_successes))
  steps=$((steps + shard_steps))
  sqlite3 "${database}" ".backup '${run_root}/recordings/shard${shard}/recording-${eval_id}.sqlite'"
done

{
  printf 'eval_id=%s\n' "${eval_id}"
  printf 'server_startup_seconds=%s\n' "${startup_seconds}"
  printf 'wall_seconds=%s\n' "${wall_seconds}"
  printf 'episodes=%s\n' "${episodes}"
  printf 'successes=%s\n' "${successes}"
  printf 'steps=%s\n' "${steps}"
  printf 'exit_code=%s\n' "${eval_rc}"
} | tee "${run_root}/summary.txt"

[[ "${episodes}" -eq "${expected_episodes}" ]]
exit "${eval_rc}"
