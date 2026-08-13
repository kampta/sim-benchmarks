#!/usr/bin/env bash
set -euo pipefail

repo_root=${XR1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
resume_root=${XR1_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_20260813}
original_root=${XR1_ORIGINAL_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812}
result_root=${XR1_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_20260813}
base_completed_manifest=${XR1_COMPLETED_MANIFEST:-${original_root}/resume/completed-55.json}
session_name=${XR1_SESSION_NAME:-xr1_resume}
local_shard_count=${XR1_LOCAL_SHARD_COUNT:-4}
shard_offset=${XR1_SHARD_OFFSET:-0}
remote_host=${XR1_REMOTE_HOST:-}
checkpoint_root="${resume_root}/recovery-checkpoints"
monitor_root="${resume_root}/recovery-checkpoint-monitor"
checkpoint_mirror_root=${XR1_CHECKPOINT_MIRROR_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/$(basename "${resume_root}")_recovery_mirror}
python_bin="${repo_root}/.venv/bin/python"
report_src="${resume_root}/provenance/src"
checkpoint_script="${monitor_root}/checkpoint_vlabench_progress.py"
interval_seconds=${XR1_CHECKPOINT_INTERVAL_SECONDS:-21600}

if [[ -z "${remote_host}" && -f "${resume_root}/base/remote-host.txt" ]]; then
  remote_host=$(<"${resume_root}/base/remote-host.txt")
fi

runner_active() {
  local remote_state
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    return 0
  fi
  if [[ -n "${remote_host}" ]]; then
    if ! remote_state=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${remote_host}" \
      "if tmux has-session -t '${session_name}' 2>/dev/null; then printf active; else printf inactive; fi"); then
      return 0
    fi
    [[ "${remote_state}" == active ]] && return 0
  fi
  return 1
}

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
  local timestamp output_dir log_path latest_next eval_id local_shard shard mirror_dir mirror_next
  local -a databases=()
  eval_id=$(<"${resume_root}/eval_id.txt")
  for local_shard in $(seq 0 $((local_shard_count - 1))); do
    shard=$((shard_offset + local_shard))
    databases+=("${result_root}/shard${shard}/recording-${eval_id}.sqlite")
  done
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  output_dir="${checkpoint_root}/${timestamp}"
  while [[ -e "${output_dir}" ]]; do
    sleep 1
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    output_dir="${checkpoint_root}/${timestamp}"
  done
  log_path="${output_dir}.log"
  PYTHONPATH="${report_src}" "${python_bin}" "${checkpoint_script}" \
    --db "${databases[@]}" \
    --track-dir "${original_root}/provenance/tracks" \
    --base-completed-manifest "${base_completed_manifest}" \
    --output-dir "${output_dir}" >"${log_path}" 2>&1
  (cd "${output_dir}" && sha256sum -c SHA256SUMS) >"${output_dir}/verification.log"
  mirror_dir="${checkpoint_mirror_root}/${timestamp}"
  mirror_next="${mirror_dir}.next"
  if [[ -e "${mirror_dir}" || -e "${mirror_next}" ]]; then
    echo "checkpoint mirror target already exists: ${mirror_dir}" >&2
    return 1
  fi
  mkdir -p "${checkpoint_mirror_root}" "${mirror_next}"
  rsync -a --checksum "${output_dir}/" "${mirror_next}/"
  (cd "${mirror_next}" && sha256sum -c SHA256SUMS) >"${mirror_next}/verification.log"
  mv "${mirror_next}" "${mirror_dir}"
  latest_next="${checkpoint_root}/latest.next"
  printf '%s\n' "${output_dir}" >"${latest_next}"
  mv -f "${latest_next}" "${checkpoint_root}/latest"
  printf '%s checkpoint=%s mirror=%s\n' \
    "$(date -Is)" "${output_dir}" "${mirror_dir}" >"${monitor_root}/status.txt"
}

checkpoint_once
while runner_active; do
  remaining=${interval_seconds}
  while [[ "${remaining}" -gt 0 ]] && runner_active; do
    slice=300
    if [[ "${remaining}" -lt "${slice}" ]]; then
      slice=${remaining}
    fi
    sleep "${slice}"
    remaining=$((remaining - slice))
  done
  if runner_active; then
    checkpoint_once
  fi
done

# Capture the final closed databases even when the main finalizer later fails
# its coverage gate because a shard exited early.
checkpoint_once
printf '%s runner-exited final-checkpoint=%s\n' \
  "$(date -Is)" "$(<"${checkpoint_root}/latest")" >"${monitor_root}/status.txt"
