#!/usr/bin/env bash
set -euo pipefail

# Transactionally replace the active four-shard Spark1 continuation with an
# eight-shard continuation split across Spark1 and Spark2. All paths default to
# the August 2026 XR-1 reproduction, but are overrideable for recovery.
repo_root=${XR1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
remote_host=${XR1_REMOTE_HOST:-spark2}
original_root=${XR1_ORIGINAL_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_official_flash_20260812}
current_run_root=${XR1_CURRENT_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_g4_two_server_20260813}
current_result_root=${XR1_CURRENT_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/xr1_resume_g4_two_server_20260813}
current_base_manifest=${XR1_CURRENT_BASE_MANIFEST:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_resume_g2_20260813/recovery-checkpoints/perf-audit-20260813T215143Z/completed-manifest.json}
remote_repo_root=${XR1_REMOTE_REPO_ROOT:-/data/shared2/user/kampta/code/sim_benchmarks}
label=${XR1_DUAL_LABEL:-xr1_resume_g5_dual_20260813}
new_run_root=${XR1_DUAL_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/${label}}
new_result_root=${XR1_DUAL_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/${label}}
session_name=${XR1_SESSION_NAME:-xr1_resume}
minimum_remote_available_kib=${XR1_REMOTE_MIN_AVAILABLE_KIB:-52428800}
image=sim-benchmarks/vlabench:cf588fe
track_dir=${original_root}/provenance/tracks
cutover_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
cutover_root=${current_run_root}/dual-spark-cutover/${cutover_timestamp}

mkdir -p "${cutover_root}"
exec 9>"${current_run_root}/dual-spark-cutover.lock"
if ! flock -n 9; then
  echo "another dual-Spark cutover holds ${current_run_root}/dual-spark-cutover.lock" >&2
  exit 1
fi
exec > >(tee -a "${cutover_root}/cutover.log") 2>&1

remote_readiness=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${remote_host}" '
  set -euo pipefail
  container=$(docker ps --filter name=^/minimax-h3-fl2va$ -q)
  available_kib=$(awk "/MemAvailable/ {print \$2}" /proc/meminfo)
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed "/^[[:space:]]*$/d" | paste -sd, -)
  printf "container=%s available_kib=%s compute_pids=%s\n" \
    "${container:+running}" "${available_kib}" "${compute_pids}"
')
printf '%s remote_readiness %s\n' "$(date -Is)" "${remote_readiness}"
remote_container=$(sed -n 's/.*container=\([^ ]*\).*/\1/p' <<<"${remote_readiness}")
remote_available_kib=$(sed -n 's/.*available_kib=\([0-9]*\).*/\1/p' <<<"${remote_readiness}")
remote_compute_pids=$(sed -n 's/.*compute_pids=\([^ ]*\).*/\1/p' <<<"${remote_readiness}")
if [[ -n "${remote_container}" || -n "${remote_compute_pids}" \
  || -z "${remote_available_kib}" || "${remote_available_kib}" -lt "${minimum_remote_available_kib}" ]]; then
  echo "Spark2 is not exclusively available; leaving the active run untouched" >&2
  exit 3
fi

if ! tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "missing active local tmux session ${session_name}; refusing cutover" >&2
  exit 1
fi
if ssh "${remote_host}" "tmux has-session -t '${session_name}' 2>/dev/null"; then
  echo "remote tmux session ${session_name} already exists; refusing cutover" >&2
  exit 1
fi
if [[ ! -f "${current_run_root}/eval_id.txt" || ! -f "${current_base_manifest}" ]]; then
  echo "current evaluation metadata or base manifest is missing" >&2
  exit 1
fi
current_eval_id=$(<"${current_run_root}/eval_id.txt")
for shard in 0 1 2 3; do
  current_database="${current_result_root}/shard${shard}/recording-${current_eval_id}.sqlite"
  [[ -f "${current_database}" ]] || { echo "missing current database ${current_database}" >&2; exit 1; }
done

local_head=$(git -C "${repo_root}" rev-parse HEAD)
remote_head=$(ssh "${remote_host}" "git -C '${remote_repo_root}' rev-parse HEAD")
local_image=$(docker image inspect "${image}" --format '{{.Id}}')
remote_image=$(ssh "${remote_host}" "docker image inspect '${image}' --format '{{.Id}}")
if [[ "${local_head}" != "${remote_head}" || "${local_image}" != "${remote_image}" ]]; then
  echo "Spark hosts do not have identical code and simulator image revisions" >&2
  exit 1
fi
printf 'git_head=%s\nimage_id=%s\n' "${local_head}" "${local_image}" >"${cutover_root}/revisions.txt"

if find "${new_result_root}" -name 'recording-*.sqlite' -print -quit 2>/dev/null | grep -q .; then
  echo "new result root already contains recordings: ${new_result_root}" >&2
  exit 1
fi

# Stop only XR-1-owned sessions. The runner trap closes its clients, servers,
# and Docker containers before the immutable checkpoint is taken.
for monitor_session in xr1_audit xr1_checkpoint xr1_finalize; do
  tmux kill-session -t "${monitor_session}" 2>/dev/null || true
done
tmux kill-session -t "${session_name}"
for _ in $(seq 1 120); do
  if ! pgrep -f "${current_eval_id}" >/dev/null \
    && ! pgrep -f 'src/sim_benchmarks/model_servers/xiaomi_robotics_1.py.*--port 1800[01]' >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f "${current_eval_id}" >/dev/null \
  || pgrep -f 'src/sim_benchmarks/model_servers/xiaomi_robotics_1.py.*--port 1800[01]' >/dev/null; then
  echo "current XR-1 processes did not close cleanly" >&2
  exit 1
fi

checkpoint_dir=${cutover_root}/checkpoint
PYTHONPATH="${current_run_root}/provenance/src" "${repo_root}/.venv/bin/python" \
  "${repo_root}/scripts/checkpoint_vlabench_progress.py" \
  --db "${current_result_root}/shard0/recording-${current_eval_id}.sqlite" \
       "${current_result_root}/shard1/recording-${current_eval_id}.sqlite" \
       "${current_result_root}/shard2/recording-${current_eval_id}.sqlite" \
       "${current_result_root}/shard3/recording-${current_eval_id}.sqlite" \
  --track-dir "${track_dir}" \
  --base-completed-manifest "${current_base_manifest}" \
  --output-dir "${checkpoint_dir}" >"${cutover_root}/checkpoint.log"
(cd "${checkpoint_dir}" && sha256sum -c SHA256SUMS) >"${cutover_root}/checkpoint-verification.log"
completed_manifest=${checkpoint_dir}/completed-manifest.json
completed_count=$("${repo_root}/.venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["completed_episode_count"])' \
  "${completed_manifest}")
printf '%s checkpoint_complete valid_identities=%s\n' "$(date -Is)" "${completed_count}"

freeze_provenance() {
  local root=$1
  mkdir -p "${root}/provenance/src" \
    "${root}/provenance/scripts" \
    "${root}/provenance/configs/benchmarks/vlabench" \
    "${root}/provenance/configs/model_servers-xiaomi_robotics_1"
  cp -a "${repo_root}/src/sim_benchmarks" "${root}/provenance/src/"
  cp "${repo_root}/scripts/audit_vlabench_partial.py" \
     "${repo_root}/scripts/checkpoint_vlabench_progress.py" \
     "${repo_root}/scripts/prepare_vlabench_retries.py" \
     "${repo_root}/scripts/finalize_xr1_vlabench_resume.sh" \
     "${root}/provenance/scripts/"
  cp "${repo_root}/configs/benchmarks/vlabench/xr1_retry.yaml" \
     "${root}/provenance/configs/benchmarks/vlabench/"
  cp "${repo_root}/configs/model_servers/xiaomi_robotics_1/vlabench_gb10.yaml" \
     "${root}/provenance/configs/model_servers-xiaomi_robotics_1/"
  (cd "${root}/provenance" \
    && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}

mkdir -p "${new_run_root}/base" "${new_result_root}"
cp "${completed_manifest}" "${new_run_root}/base/completed-manifest.json"
freeze_provenance "${new_run_root}"
ssh "${remote_host}" "mkdir -p '${new_run_root}/base' '${new_result_root}'"
rsync -a --checksum "${new_run_root}/base/" "${remote_host}:${new_run_root}/base/"
rsync -a --checksum "${new_run_root}/provenance/" "${remote_host}:${new_run_root}/provenance/"

distributed_eval_id=xr1-full-resume-g5-dual-${cutover_timestamp}
common_env="XR1_COMPLETED_MANIFEST=${new_run_root}/base/completed-manifest.json XR1_RUN_ROOT=${new_run_root} XR1_RESULT_ROOT=${new_result_root} XR1_EVAL_ID=${distributed_eval_id} XR1_BASE_PORT=18000 XR1_SHARD_COUNT=8 XR1_LOCAL_SHARD_COUNT=4 XR1_SERVER_COUNT=2"
tmux new-session -d -s "${session_name}" \
  "cd '${repo_root}' && env ${common_env} XR1_SHARD_OFFSET=0 bash scripts/run_xr1_vlabench_resume.sh >'${new_run_root}/supervisor.log' 2>&1"
ssh "${remote_host}" "tmux new-session -d -s '${session_name}' \"cd '${remote_repo_root}' && env ${common_env} XR1_SHARD_OFFSET=4 bash scripts/run_xr1_vlabench_resume.sh >'${new_run_root}/supervisor.log' 2>&1\""

for _ in $(seq 1 240); do
  local_ready=0
  remote_ready=0
  [[ -f "${new_run_root}/eval_id.txt" ]] && local_ready=1
  ssh "${remote_host}" "test -f '${new_run_root}/eval_id.txt'" && remote_ready=1 || true
  database_count=$(find "${new_result_root}" -name 'recording-*.sqlite' | wc -l)
  if [[ "${local_ready}" -eq 1 && "${remote_ready}" -eq 1 && "${database_count}" -eq 8 ]]; then
    break
  fi
  if ! tmux has-session -t "${session_name}" 2>/dev/null \
    || ! ssh "${remote_host}" "tmux has-session -t '${session_name}' 2>/dev/null"; then
    echo "a distributed XR-1 runner exited during startup" >&2
    exit 1
  fi
  sleep 5
done
database_count=$(find "${new_result_root}" -name 'recording-*.sqlite' | wc -l)
if [[ "${database_count}" -ne 8 ]]; then
  echo "distributed runners did not create eight recording databases" >&2
  exit 1
fi

tmux new-session -d -s xr1_audit \
  "cd '${repo_root}' && env ${common_env} XR1_SHARD_OFFSET=0 XR1_SESSION_NAME=${session_name} bash scripts/monitor_xr1_vlabench_partial.sh >'${new_run_root}/partial-audit-monitor.log' 2>&1"
tmux new-session -d -s xr1_checkpoint \
  "cd '${repo_root}' && env ${common_env} XR1_SHARD_OFFSET=0 XR1_SESSION_NAME=${session_name} bash scripts/monitor_xr1_recovery_checkpoints.sh >>'${new_run_root}/recovery-checkpoint-monitor.log' 2>&1"
ssh "${remote_host}" "tmux new-session -d -s xr1_audit \"cd '${remote_repo_root}' && env ${common_env} XR1_SHARD_OFFSET=4 XR1_SESSION_NAME=${session_name} bash scripts/monitor_xr1_vlabench_partial.sh >'${new_run_root}/partial-audit-monitor.log' 2>&1\""
ssh "${remote_host}" "tmux new-session -d -s xr1_checkpoint \"cd '${remote_repo_root}' && env ${common_env} XR1_SHARD_OFFSET=4 XR1_SESSION_NAME=${session_name} bash scripts/monitor_xr1_recovery_checkpoints.sh >>'${new_run_root}/recovery-checkpoint-monitor.log' 2>&1\""

printf '%s distributed_started base_valid=%s result_root=%s\n' \
  "$(date -Is)" "${completed_count}" "${new_result_root}" >"${cutover_root}/status.txt"
cat "${cutover_root}/status.txt"
