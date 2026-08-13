#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks
original_root=/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812
resume_root=/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_20260813
result_root=/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_20260813
final_root="${resume_root}/final"
track_dir="${original_root}/provenance/tracks"
python_bin="${repo_root}/.venv/bin/python"
report_src="${resume_root}/provenance/src"
retry_server_pid=

cleanup() {
  if [[ -n "${retry_server_pid}" ]]; then
    kill "${retry_server_pid}" 2>/dev/null || true
    wait "${retry_server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "${final_root}"

# The runner owns the live SQLite databases. Wait for it to exit before making
# the immutable final snapshots. A crash is handled by the same coverage gate.
while tmux has-session -t xr1_resume 2>/dev/null; do
  date -Is >"${resume_root}/finalizer-status.txt"
  printf 'waiting for xr1_resume\n' >>"${resume_root}/finalizer-status.txt"
  sleep 300
done

if [[ ! -f "${resume_root}/eval_id.txt" ]]; then
  printf '%s missing eval_id.txt\n' "$(date -Is)" >"${resume_root}/finalizer-status.txt"
  exit 1
fi
eval_id=$(<"${resume_root}/eval_id.txt")

if ! (cd "${resume_root}/provenance" && sha256sum -c SHA256SUMS) \
  >"${final_root}/provenance-verification.log" 2>&1; then
  printf '%s frozen provenance checksum verification failed\n' \
    "$(date -Is)" >"${resume_root}/finalizer-status.txt"
  exit 1
fi

snapshot_dir="${final_root}/recordings"
mkdir -p "${snapshot_dir}"
original_databases=()
resume_databases=()
for shard in 0 1 2 3; do
  original="${original_root}/shard${shard}/recording-xr1-full-flash-20260812.sqlite"
  resumed="${result_root}/shard${shard}/recording-${eval_id}.sqlite"
  if [[ ! -f "${original}" || ! -f "${resumed}" ]]; then
    printf '%s missing shard database original=%s resumed=%s\n' \
      "$(date -Is)" "${original}" "${resumed}" >"${resume_root}/finalizer-status.txt"
    exit 1
  fi
  original_snapshot="${snapshot_dir}/original-shard${shard}.sqlite"
  resume_snapshot="${snapshot_dir}/resume-shard${shard}.sqlite"
  sqlite3 "${original}" ".backup '${original_snapshot}'"
  sqlite3 "${resumed}" ".backup '${resume_snapshot}'"
  original_databases+=("${original_snapshot}")
  resume_databases+=("${resume_snapshot}")
done
sha256sum "${snapshot_dir}"/*.sqlite >"${snapshot_dir}/SHA256SUMS"

old_count=0
new_count=0
for database in "${original_databases[@]}"; do
  count=$(sqlite3 "${database}" "SELECT COUNT(*) FROM episode_results;")
  old_count=$((old_count + count))
done
for database in "${resume_databases[@]}"; do
  count=$(sqlite3 "${database}" "SELECT COUNT(*) FROM episode_results;")
  new_count=$((new_count + count))
done
if [[ "${old_count}" -ne 55 || "${new_count}" -ne 2405 ]]; then
  printf '%s incomplete old=%s/55 resumed=%s/2405\n' \
    "$(date -Is)" "${old_count}" "${new_count}" >"${resume_root}/finalizer-status.txt"
  exit 1
fi

cd "${repo_root}"
if [[ ! -f "${report_src}/sim_benchmarks/reporting/vlabench.py" ]]; then
  printf '%s missing frozen reporting source: %s\n' \
    "$(date -Is)" "${report_src}" >"${resume_root}/finalizer-status.txt"
  exit 1
fi
recovery_dir="${final_root}/recovery"
PYTHONPATH="${report_src}" "${python_bin}" \
  "${resume_root}/provenance/scripts/prepare_vlabench_retries.py" \
  --db "${resume_databases[@]}" \
  --base-completed-manifest "${original_root}/resume/completed-55.json" \
  --track-dir "${track_dir}" \
  --output-dir "${recovery_dir}" \
  >"${recovery_dir}.log" 2>&1

retry_count=$("${python_bin}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["retry_identities"])' \
  "${recovery_dir}/preparation.json")
scored_resume_databases=()
for shard in 0 1 2 3; do
  scored_resume_databases+=("${recovery_dir}/clean/shard${shard}.sqlite")
done
retry_databases=()

if [[ "${retry_count}" -gt 0 ]]; then
  export HF_HOME=/data/shared1/cache/huggingface
  export HF_HUB_CACHE=/data/shared1/cache/huggingface/hub
  export HF_MODULES_CACHE=/data/shared1/cache/huggingface/kampta/modules
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TORCH_HOME=/data/shared1/cache/torch
  export TORCH_EXTENSIONS_DIR=/data/shared1/cache/torch/extensions/kampta/xr1
  export TORCHINDUCTOR_CACHE_DIR=/data/shared1/cache/torch/inductor/xr1_cuda130
  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  export PYTHONPATH="${report_src}"
  export PYTHONUNBUFFERED=1

  /data/shared1/envs/xr1/bin/python \
    "${report_src}/sim_benchmarks/model_servers/xiaomi_robotics_1.py" \
    --config "${resume_root}/provenance/configs/model_servers-xiaomi_robotics_1/vlabench_gb10.yaml" \
    --port 18000 \
    >"${recovery_dir}/retry-server.log" 2>&1 &
  retry_server_pid=$!
  retry_ready=0
  for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:18000/config >/dev/null 2>&1; then
      retry_ready=1
      break
    fi
    if ! kill -0 "${retry_server_pid}" 2>/dev/null; then
      printf '%s retry server exited during startup\n' "$(date -Is)" >"${resume_root}/finalizer-status.txt"
      exit 1
    fi
    sleep 5
  done
  if [[ "${retry_ready}" -ne 1 ]]; then
    printf '%s retry server did not become ready\n' "$(date -Is)" >"${resume_root}/finalizer-status.txt"
    exit 1
  fi

  retry_attempt=1
  while [[ "${retry_count}" -gt 0 && "${retry_attempt}" -le 3 ]]; do
    retry_eval_id="xr1-retry${retry_attempt}-$(date -u +%Y%m%dT%H%M%SZ)"
    retry_result_dir="/data/shared1/cache/sim_benchmarks/vlabench/xr1_retry_20260813/${retry_eval_id}"
    retry_attempt_dir="${recovery_dir}/attempt${retry_attempt}"
    mkdir -p "${retry_result_dir}" "${retry_attempt_dir}"

    retry_eval_rc=0
    "${repo_root}/.venv/bin/vla-eval" run \
      --config "${resume_root}/provenance/configs/benchmarks/vlabench/xr1_retry.yaml" \
      --server-url ws://127.0.0.1:18000 \
      --output-dir "${retry_result_dir}" \
      --eval-id "${retry_eval_id}" \
      --cpus 0-4 --yes \
      >"${retry_attempt_dir}/eval.log" 2>&1 || retry_eval_rc=$?
    printf 'exit_code=%s\n' "${retry_eval_rc}" >"${retry_attempt_dir}/eval-exit-code.txt"

    retry_live_database="${retry_result_dir}/recording-${retry_eval_id}.sqlite"
    retry_snapshot="${snapshot_dir}/retry-attempt${retry_attempt}-raw.sqlite"
    sqlite3 "${retry_live_database}" ".backup '${retry_snapshot}'"
    actual_retry_count=$(sqlite3 "${retry_snapshot}" "SELECT COUNT(*) FROM episode_results;")
    if [[ "${actual_retry_count}" -ne "${retry_count}" ]]; then
      printf '%s incomplete retry attempt=%s expected=%s observed=%s\n' \
        "$(date -Is)" "${retry_attempt}" "${retry_count}" "${actual_retry_count}" \
        >"${resume_root}/finalizer-status.txt"
      exit 1
    fi

    PYTHONPATH="${report_src}" "${python_bin}" \
      "${resume_root}/provenance/scripts/prepare_vlabench_retries.py" \
      --db "${retry_snapshot}" \
      --base-completed-manifest "${recovery_dir}/retry-manifest.json" \
      --track-dir "${track_dir}" \
      --output-dir "${retry_attempt_dir}" \
      >"${retry_attempt_dir}/preparation.log" 2>&1
    retry_databases+=("${retry_attempt_dir}/clean/shard0.sqlite")
    retry_count=$("${python_bin}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["retry_identities"])' \
      "${retry_attempt_dir}/preparation.json")
    cp "${retry_attempt_dir}/retry-manifest.json" "${recovery_dir}/retry-manifest.next.json"
    mv -f "${recovery_dir}/retry-manifest.next.json" "${recovery_dir}/retry-manifest.json"
    retry_attempt=$((retry_attempt + 1))
  done
  kill "${retry_server_pid}" 2>/dev/null || true
  wait "${retry_server_pid}" 2>/dev/null || true
  retry_server_pid=
  if [[ "${retry_count}" -gt 0 ]]; then
    printf '%s retries exhausted remaining=%s\n' \
      "$(date -Is)" "${retry_count}" >"${resume_root}/finalizer-status.txt"
    exit 1
  fi
fi
sha256sum "${snapshot_dir}"/*.sqlite >"${snapshot_dir}/SHA256SUMS"

PYTHONPATH="${report_src}" "${python_bin}" -m sim_benchmarks.reporting.vlabench \
  --db "${original_databases[@]}" "${scored_resume_databases[@]}" "${retry_databases[@]}" \
  --track-dir "${track_dir}" \
  --output "${final_root}/report.json" \
  --markdown-output "${final_root}/comparison.md" \
  >"${final_root}/report.log" 2>&1

"${python_bin}" - "${final_root}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["num_episodes_total"] == 2460
assert report["coverage_validation"]["status"] == "complete"
assert report["episode_identity_validation"]["status"] == "exact_complete"
assert report["episode_identity_validation"]["observed_identities"] == 2460
assert sum(entry["episode_results"] for entry in report["recording_validation"]) == 2460
PY

sha256sum "${final_root}/report.json" "${final_root}/comparison.md" >"${final_root}/SHA256SUMS"
printf '%s complete episodes=2460 report=%s\n' \
  "$(date -Is)" "${final_root}/report.json" >"${resume_root}/finalizer-status.txt"
