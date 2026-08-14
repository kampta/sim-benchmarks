#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks
pi05_python=/data/shared1/envs/pi05_v060/bin/python
libero_python=/data/shared1/envs/libero/bin/python
run_root=/data/shared2/user/kampta/logs/sim_benchmarks/libero/pi05_fast_gate
result_root=/data/shared1/cache/sim_benchmarks/libero/pi05_fast_gate
compile_model=${PI05_COMPILE_MODEL:-true}
num_shards=${PI05_NUM_SHARDS:-10}

if ((num_shards < 1 || num_shards > 10)); then
  echo "PI05_NUM_SHARDS must be between 1 and 10" >&2
  exit 2
fi

mkdir -p "${run_root}" "${result_root}"
cd "${repo_root}"

export HF_HOME=/data/shared1/cache/huggingface
export HF_HUB_CACHE=/data/shared1/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/shared1/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/data/shared1/cache/torch/inductor/pi05_v060_cuda130_compact
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PYTHONPATH="${repo_root}/src"
export PYTHONUNBUFFERED=1

server_log="${run_root}/server.log"
"${pi05_python}" src/sim_benchmarks/model_servers/pi05.py \
  --port 18010 \
  --args.policy_type pi05 \
  --args.checkpoint /data/shared1/models/pi05_libero_finetuned \
  --args.image_keys='{"agentview":"observation.images.image","wrist":"observation.images.image2"}' \
  --args.state_key observation.state \
  --args.device cuda \
  --args.chunk_size 10 \
  --args.compile_model "${compile_model}" \
  --args.max_batch_size 1 \
  --args.compact_masked_cameras true \
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

# The smoke doubles as the one-time torch.compile warmup. It must succeed
# before the throughput gate begins.
smoke_eval_id="pi05-fast-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
smoke_dir="${result_root}/smoke"
mkdir -p "${smoke_dir}"
"${libero_python}" -m sim_benchmarks.fast_eval run \
  --config configs/benchmarks/libero/pi05_smoke_fast.yaml \
  --server-url ws://127.0.0.1:18010 \
  --no-docker --yes \
  --output-dir "${smoke_dir}" \
  --eval-id "${smoke_eval_id}" \
  2>&1 | tee "${run_root}/smoke.log"
sqlite3 "${smoke_dir}/recording-${smoke_eval_id}.sqlite" "PRAGMA quick_check;"
smoke_successes=$(
  sqlite3 "${smoke_dir}/recording-${smoke_eval_id}.sqlite" \
    "SELECT count(*) FROM episode_results WHERE status = 'success';"
)
if [[ "${smoke_successes}" != 1 ]]; then
  echo "pi0.5 smoke gate did not record exactly one successful episode" >&2
  exit 1
fi

gate_eval_id="pi05-fast-gate-$(date -u +%Y%m%dT%H%M%SZ)"
gate_root="${result_root}/gate-${gate_eval_id}"
mkdir -p "${gate_root}"
start_seconds=$(date +%s)
pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  shard_dir="${gate_root}/shard${shard}"
  mkdir -p "${shard_dir}"
  "${libero_python}" -m sim_benchmarks.fast_eval run \
    --config configs/benchmarks/libero/pi05_throughput_fast.yaml \
    --server-url ws://127.0.0.1:18010 \
    --no-docker --yes \
    --output-dir "${shard_dir}" \
    --eval-id "${gate_eval_id}" \
    --shard-id "${shard}" --num-shards "${num_shards}" \
    >"${run_root}/gate-shard${shard}.log" 2>&1 &
  pids+=("$!")
done

gate_rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    gate_rc=1
  fi
done
episode_count=0
success_count=0
for database in "${gate_root}"/shard*/recording-"${gate_eval_id}".sqlite; do
  sqlite3 "${database}" "PRAGMA quick_check;"
  episode_count=$((episode_count + $(sqlite3 "${database}" "SELECT count(*) FROM episode_results;")))
  success_count=$((success_count + $(sqlite3 "${database}" "SELECT count(*) FROM episode_results WHERE status = 'success';")))
done
if ((episode_count != 10 || success_count != 10)); then
  echo "pi0.5 gate expected 10/10 successful episodes; got ${success_count}/${episode_count}" >&2
  gate_rc=1
fi

wall_seconds=$(($(date +%s) - start_seconds))
printf 'eval_id=%s\nepisodes=%s\nsuccesses=%s\nnum_shards=%s\nwall_seconds=%s\nexit_code=%s\n' \
  "${gate_eval_id}" "${episode_count}" "${success_count}" "${num_shards}" "${wall_seconds}" "${gate_rc}" \
  | tee "${run_root}/summary.txt"
exit "${gate_rc}"
