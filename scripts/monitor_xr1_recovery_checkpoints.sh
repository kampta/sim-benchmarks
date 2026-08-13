#!/usr/bin/env bash
set -euo pipefail

repo_root=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks
resume_root=/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_20260813
original_root=/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812
result_root=/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_20260813
checkpoint_root="${resume_root}/recovery-checkpoints"
monitor_root="${resume_root}/recovery-checkpoint-monitor"
python_bin="${repo_root}/.venv/bin/python"
report_src="${resume_root}/provenance/src"
checkpoint_script="${monitor_root}/checkpoint_vlabench_progress.py"
interval_seconds=${XR1_CHECKPOINT_INTERVAL_SECONDS:-21600}

if [[ "${interval_seconds}" -lt 300 ]]; then
  echo "XR1_CHECKPOINT_INTERVAL_SECONDS must be at least 300" >&2
  exit 2
fi
if [[ ! -f "${resume_root}/eval_id.txt" ]]; then
  echo "missing ${resume_root}/eval_id.txt" >&2
  exit 1
fi
eval_id=$(<"${resume_root}/eval_id.txt")

mkdir -p "${checkpoint_root}" "${monitor_root}"
cp "${repo_root}/scripts/checkpoint_vlabench_progress.py" "${checkpoint_script}.next"
mv -f "${checkpoint_script}.next" "${checkpoint_script}"
sha256sum "${checkpoint_script}" >"${monitor_root}/checkpoint-script.SHA256"

checkpoint_once() {
  local timestamp output_dir log_path latest_next
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  output_dir="${checkpoint_root}/${timestamp}"
  while [[ -e "${output_dir}" ]]; do
    sleep 1
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    output_dir="${checkpoint_root}/${timestamp}"
  done
  log_path="${output_dir}.log"
  PYTHONPATH="${report_src}" "${python_bin}" "${checkpoint_script}" \
    --db "${result_root}/shard0/recording-${eval_id}.sqlite" \
         "${result_root}/shard1/recording-${eval_id}.sqlite" \
         "${result_root}/shard2/recording-${eval_id}.sqlite" \
         "${result_root}/shard3/recording-${eval_id}.sqlite" \
    --track-dir "${original_root}/provenance/tracks" \
    --base-completed-manifest "${original_root}/resume/completed-55.json" \
    --output-dir "${output_dir}" >"${log_path}" 2>&1
  (cd "${output_dir}" && sha256sum -c SHA256SUMS) >"${output_dir}/verification.log"
  latest_next="${checkpoint_root}/latest.next"
  printf '%s\n' "${output_dir}" >"${latest_next}"
  mv -f "${latest_next}" "${checkpoint_root}/latest"
  printf '%s checkpoint=%s\n' "$(date -Is)" "${output_dir}" >"${monitor_root}/status.txt"
}

checkpoint_once
while tmux has-session -t xr1_resume 2>/dev/null; do
  remaining=${interval_seconds}
  while [[ "${remaining}" -gt 0 ]] && tmux has-session -t xr1_resume 2>/dev/null; do
    slice=300
    if [[ "${remaining}" -lt "${slice}" ]]; then
      slice=${remaining}
    fi
    sleep "${slice}"
    remaining=$((remaining - slice))
  done
  if tmux has-session -t xr1_resume 2>/dev/null; then
    checkpoint_once
  fi
done

# Capture the final closed databases even when the main finalizer later fails
# its coverage gate because a shard exited early.
checkpoint_once
printf '%s runner-exited final-checkpoint=%s\n' \
  "$(date -Is)" "$(<"${checkpoint_root}/latest")" >"${monitor_root}/status.txt"
