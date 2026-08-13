#!/usr/bin/env bash
set -euo pipefail

resume_root=${XR1_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_20260813}
original_root=${XR1_ORIGINAL_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812}
result_root=${XR1_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_20260813}
base_completed_manifest=${XR1_COMPLETED_MANIFEST:-${original_root}/resume/completed-55.json}
session_name=${XR1_SESSION_NAME:-xr1_resume}
python_bin=/data/shared2/user/kampta/code/sim_benchmarks/.slide-worktrees/sim_benchmarks/.venv/bin/python
audit_script="${resume_root}/provenance/scripts/audit_vlabench_partial.py"
report_src="${resume_root}/provenance/src"

audit_once() {
  local eval_id
  eval_id=$(<"${resume_root}/eval_id.txt")
  PYTHONPATH="${report_src}" "${python_bin}" "${audit_script}" \
    --db "${result_root}/shard0/recording-${eval_id}.sqlite" \
         "${result_root}/shard1/recording-${eval_id}.sqlite" \
         "${result_root}/shard2/recording-${eval_id}.sqlite" \
         "${result_root}/shard3/recording-${eval_id}.sqlite" \
    --track-dir "${original_root}/provenance/tracks" \
    --base-completed-manifest "${base_completed_manifest}" \
    --output "${resume_root}/partial-audit.json" \
    >"${resume_root}/partial-audit.log.next"
  mv -f "${resume_root}/partial-audit.log.next" "${resume_root}/partial-audit.log"
}

while tmux has-session -t "${session_name}" 2>/dev/null; do
  audit_once
  sleep 600
done
audit_once
