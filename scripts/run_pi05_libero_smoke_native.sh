#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks
pi05_python=/data/shared1/envs/pi05_v060/bin/python
libero_python=/data/shared1/envs/libero/bin/python
run_root=/data/shared2/user/kampta/logs/sim_benchmarks/libero/pi05_v060_native
smoke_log_root="${run_root}/smoke"
throughput_log_root="${run_root}/ten_shards"
smoke_result_root=/data/shared1/cache/sim_benchmarks/libero/pi05_v060_smoke
throughput_result_root=/data/shared1/cache/sim_benchmarks/libero/pi05_v060_ten_shards
compile_model=${PI05_COMPILE_MODEL:-false}
server_log="${run_root}/server.log"
eval_log="${smoke_log_root}/eval.log"

mkdir -p "${smoke_log_root}" "${throughput_log_root}" "${smoke_result_root}" "${throughput_result_root}"
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

"${pi05_python}" src/sim_benchmarks/model_servers/pi05.py \
  --port 18010 \
  --args.policy_type pi05 \
  --args.checkpoint /data/shared1/models/pi05_libero_finetuned \
  --args.image_keys='{"agentview":"observation.images.image","wrist":"observation.images.image2"}' \
  --args.state_key observation.state \
  --args.device cuda \
  --args.chunk_size 10 \
  --args.compile_model "${compile_model}" \
  >"${server_log}" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:18010/config >/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "pi0.5 model server exited during startup; see ${server_log}" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:18010/config >/dev/null

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
eval_id="pi05-native-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
"${libero_python}" -m vla_eval.cli.main run \
  --config configs/benchmarks/libero/pi05_smoke.yaml \
  --server-url ws://127.0.0.1:18010 \
  --no-docker --yes \
  --output-dir "${smoke_result_root}" \
  --eval-id "${eval_id}" \
  2>&1 | tee "${eval_log}"

cp -a "${smoke_result_root}/recording-${eval_id}.sqlite" "${smoke_log_root}/"
sqlite3 "${smoke_log_root}/recording-${eval_id}.sqlite" "PRAGMA quick_check;"

throughput_eval_id="pi05-native-ten-$(date -u +%Y%m%dT%H%M%SZ)"
start_seconds=$(date +%s)
pids=()
for shard in $(seq 0 9); do
  shard_result="${throughput_result_root}/shard${shard}"
  shard_log="${throughput_log_root}/shard${shard}.log"
  mkdir -p "${shard_result}" "${throughput_log_root}/recordings/shard${shard}"
  "${libero_python}" -m vla_eval.cli.main run \
    --config configs/benchmarks/libero/pi05_throughput.yaml \
    --server-url ws://127.0.0.1:18010 \
    --no-docker --yes \
    --output-dir "${shard_result}" \
    --eval-id "${throughput_eval_id}" \
    --shard-id "${shard}" --num-shards 10 \
    >"${shard_log}" 2>&1 &
  pids+=("$!")
done

throughput_rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    throughput_rc=1
  fi
done
wall_seconds=$(($(date +%s) - start_seconds))
printf 'eval_id=%s\nepisodes=10\nwall_seconds=%s\nexit_code=%s\n' \
  "${throughput_eval_id}" "${wall_seconds}" "${throughput_rc}" \
  | tee "${throughput_log_root}/summary.txt"

for shard in $(seq 0 9); do
  database="${throughput_result_root}/shard${shard}/recording-${throughput_eval_id}.sqlite"
  sqlite3 "${database}" "PRAGMA quick_check;"
  cp -a "${database}" "${throughput_log_root}/recordings/shard${shard}/"
done
exit "${throughput_rc}"
