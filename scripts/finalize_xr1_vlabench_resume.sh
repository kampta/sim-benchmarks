#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks
original_root=${XR1_ORIGINAL_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812}
resume_root=${XR1_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_20260813}
result_root=${XR1_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_20260813}
base_completed_manifest=${XR1_COMPLETED_MANIFEST:-${original_root}/resume/completed-55.json}
prior_clean_dirs=${XR1_PRIOR_CLEAN_DIRS:-${XR1_PRIOR_CLEAN_DIR:-}}
session_name=${XR1_SESSION_NAME:-xr1_resume}
shard_count=${XR1_SHARD_COUNT:-4}
base_port=${XR1_BASE_PORT:-18000}
retry_result_root=${XR1_RETRY_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/$(basename "${resume_root}")_retry}
final_root="${resume_root}/final"
track_dir="${original_root}/provenance/tracks"
python_bin="${repo_root}/.venv/bin/python"
report_src=${XR1_REPORT_SRC:-${resume_root}/provenance/src}
retry_config=${XR1_RETRY_CONFIG:-${resume_root}/provenance/configs/benchmarks/vlabench/xr1_retry.yaml}
retry_server_pid=

if [[ -z "${prior_clean_dirs}" && -f "${resume_root}/base/prior-clean-dirs.txt" ]]; then
  prior_clean_dirs=$(paste -sd: "${resume_root}/base/prior-clean-dirs.txt")
fi

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
while tmux has-session -t "${session_name}" 2>/dev/null; do
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
prior_databases=()
resume_databases=()
for shard in $(seq 0 $((shard_count - 1))); do
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
  prior_databases+=("${original_snapshot}")
  resume_databases+=("${resume_snapshot}")
done
if [[ -n "${prior_clean_dirs}" ]]; then
  IFS=: read -r -a clean_dirs <<<"${prior_clean_dirs}"
  for clean_index in "${!clean_dirs[@]}"; do
    prior_clean_dir=${clean_dirs[clean_index]}
    for shard in 0 1 2 3; do
      prior_clean="${prior_clean_dir}/shard${shard}.sqlite"
      if [[ ! -f "${prior_clean}" ]]; then
        printf '%s missing prior clean database=%s\n' \
          "$(date -Is)" "${prior_clean}" >"${resume_root}/finalizer-status.txt"
        exit 1
      fi
      prior_snapshot="${snapshot_dir}/prior-clean${clean_index}-shard${shard}.sqlite"
      sqlite3 "${prior_clean}" ".backup '${prior_snapshot}'"
      prior_databases+=("${prior_snapshot}")
    done
  done
fi
sha256sum "${snapshot_dir}"/*.sqlite >"${snapshot_dir}/SHA256SUMS"

base_count=$(PYTHONPATH="${report_src}" "${python_bin}" - "${base_completed_manifest}" <<'PY'
import sys
from sim_benchmarks.benchmarks.vlabench_xr1 import load_completed_episode_identities

print(len(load_completed_episode_identities(sys.argv[1])))
PY
)
expected_resume_count=$((2460 - base_count))
prior_count=0
new_count=0
for database in "${prior_databases[@]}"; do
  count=$(sqlite3 "${database}" "SELECT COUNT(*) FROM episode_results;")
  prior_count=$((prior_count + count))
done
for database in "${resume_databases[@]}"; do
  count=$(sqlite3 "${database}" "SELECT COUNT(*) FROM episode_results;")
  new_count=$((new_count + count))
done
if [[ "${prior_count}" -ne "${base_count}" || "${new_count}" -ne "${expected_resume_count}" ]]; then
  printf '%s incomplete prior=%s/%s resumed=%s/%s\n' \
    "$(date -Is)" "${prior_count}" "${base_count}" "${new_count}" "${expected_resume_count}" \
    >"${resume_root}/finalizer-status.txt"
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
  --base-completed-manifest "${base_completed_manifest}" \
  --track-dir "${track_dir}" \
  --output-dir "${recovery_dir}" \
  >"${recovery_dir}.log" 2>&1

retry_count=$("${python_bin}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["retry_identities"])' \
  "${recovery_dir}/preparation.json")
scored_resume_databases=()
for shard in $(seq 0 $((shard_count - 1))); do
  scored_resume_databases+=("${recovery_dir}/clean/shard${shard}.sqlite")
done
retry_databases=()
runtime_error_args=()

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
  export XR1_REPORT_SRC="${report_src}"
  export XR1_RETRY_MANIFEST="${recovery_dir}/retry-manifest.json"
  export XR1_RETRY_RESULT_ROOT="${retry_result_root}"

  /data/shared1/envs/xr1/bin/python \
    "${report_src}/sim_benchmarks/model_servers/xiaomi_robotics_1.py" \
    --config "${resume_root}/provenance/configs/model_servers-xiaomi_robotics_1/vlabench_gb10.yaml" \
    --port "${base_port}" \
    >"${recovery_dir}/retry-server.log" 2>&1 &
  retry_server_pid=$!
  retry_ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${base_port}/config" >/dev/null 2>&1; then
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
    retry_result_dir="${retry_result_root}/${retry_eval_id}"
    retry_attempt_dir="${recovery_dir}/attempt${retry_attempt}"
    mkdir -p "${retry_result_dir}" "${retry_attempt_dir}"

    retry_eval_rc=0
    "${repo_root}/.venv/bin/vla-eval" run \
      --config "${retry_config}" \
      --server-url "ws://127.0.0.1:${base_port}" \
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
    unresolved_errors="${retry_attempt_dir}/failed-attempts.json"
    if [[ ! -f "${unresolved_errors}" ]]; then
      printf '%s retries exhausted but unresolved error records are missing\n' \
        "$(date -Is)" >"${resume_root}/finalizer-status.txt"
      exit 1
    fi
    cp "${unresolved_errors}" "${recovery_dir}/unresolved-runtime-errors.json"
    runtime_error_args=(--runtime-errors-json "${recovery_dir}/unresolved-runtime-errors.json")
    printf '%s retries exhausted; preserving %s Xiaomi-compatible runtime exclusion(s)\n' \
      "$(date -Is)" "${retry_count}" >"${resume_root}/finalizer-status.txt"
  fi
fi
sha256sum "${snapshot_dir}"/*.sqlite >"${snapshot_dir}/SHA256SUMS"

PYTHONPATH="${report_src}" "${python_bin}" -m sim_benchmarks.reporting.vlabench \
  --db "${prior_databases[@]}" "${scored_resume_databases[@]}" "${retry_databases[@]}" \
  --track-dir "${track_dir}" \
  "${runtime_error_args[@]}" \
  --output "${final_root}/report.json" \
  --markdown-output "${final_root}/comparison.md" \
  >"${final_root}/report.log" 2>&1

"${python_bin}" - "${final_root}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["num_attempted_episodes"] == 2460
assert report["coverage_validation"]["status"] in {"complete", "complete_with_runtime_exclusions"}
assert report["episode_identity_validation"]["status"] in {
    "exact_complete",
    "exact_complete_with_runtime_exclusions",
}
assert report["episode_identity_validation"]["attempted_identities"] == 2460
excluded = len(report.get("runtime_errors", []))
assert report["num_scored_episodes"] + excluded == 2460
assert sum(entry["episode_results"] for entry in report["recording_validation"]) == report["num_scored_episodes"]
PY

sha256sum "${final_root}/report.json" "${final_root}/comparison.md" >"${final_root}/SHA256SUMS"
printf '%s complete attempted_episodes=2460 report=%s\n' \
  "$(date -Is)" "${final_root}/report.json" >"${resume_root}/finalizer-status.txt"
