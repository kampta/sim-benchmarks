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
python_bin=${XR1_HARNESS_PYTHON:-${repo_root}/.venv/bin/python}
audit_script="${resume_root}/provenance/scripts/audit_vlabench_partial.py"
report_src="${resume_root}/provenance/src"

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

audit_once() {
  local eval_id local_shard shard
  local -a databases=()
  eval_id=$(<"${resume_root}/eval_id.txt")
  for local_shard in $(seq 0 $((local_shard_count - 1))); do
    shard=$((shard_offset + local_shard))
    databases+=("${result_root}/shard${shard}/recording-${eval_id}.sqlite")
  done
  PYTHONPATH="${report_src}" "${python_bin}" "${audit_script}" \
    --db "${databases[@]}" \
    --track-dir "${original_root}/provenance/tracks" \
    --base-completed-manifest "${base_completed_manifest}" \
    --output "${resume_root}/partial-audit.json" \
    >"${resume_root}/partial-audit.log.next"
  mv -f "${resume_root}/partial-audit.log.next" "${resume_root}/partial-audit.log"
}

while runner_active; do
  audit_once
  sleep 600
done
audit_once
