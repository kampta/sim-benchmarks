#!/usr/bin/env bash
set -euo pipefail

repo_root=${XR1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
bundle=${XR1_MIGRATION_BUNDLE:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_migration_bundle_20260813T234213Z}
env_prefix=${XR1_ENV_PREFIX:-/data/shared1/envs/xr1-rtx-x86_64}
xr1_python=${XR1_PYTHON:-${env_prefix}/bin/python}
harness_python=${XR1_HARNESS_PYTHON:-${xr1_python}}
vla_eval=${XR1_VLA_EVAL:-${env_prefix}/bin/vla-eval}
image=${XR1_VLABENCH_IMAGE:-sim-benchmarks/vlabench:cf588fe-cuda128-amd64}
gpu_ids=${XR1_GPU_IDS:-0}
shard_count=${XR1_SHARD_COUNT:-}
server_count=${XR1_SERVER_COUNT:-}
cpu_spec=${XR1_CPUS:-0-31}
base_port=${XR1_BASE_PORT:-18000}
session_name=${XR1_SESSION_NAME:-xr1_resume_rtx}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
label=${XR1_RUN_LABEL:-xr1_resume_rtx_${timestamp}}
run_root=${XR1_RUN_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/${label}}
result_root=${XR1_RESULT_ROOT:-/data/shared1/cache/sim_benchmarks/vlabench/${label}}
gate_file=${XR1_THROUGHPUT_GATE_FILE:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_rtx_throughput_gate_$(hostname -s).json}

cd "${repo_root}"
if [[ "$(uname -m)" != x86_64 ]]; then
  echo "XR-1 RTX continuation requires x86_64; found $(uname -m)" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "refusing to launch the official continuation from a dirty worktree" >&2
  exit 2
fi
for required in "${bundle}/BUNDLE_SHA256SUMS" "${gate_file}" "${xr1_python}" "${vla_eval}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required RTX continuation input: ${required}" >&2
    exit 2
  fi
done
for executable in tmux sqlite3 docker nvidia-smi; do
  command -v "${executable}" >/dev/null || { echo "missing executable: ${executable}" >&2; exit 2; }
done
(cd "${bundle}" && sha256sum -c BUNDLE_SHA256SUMS) >/dev/null
image_id=$(docker image inspect --format '{{.Id}}' "${image}")
image_arch=$(docker image inspect --format '{{.Architecture}}' "${image}")
image_vlabench_revision=$(docker image inspect \
  --format '{{index .Config.Labels "org.opencontainers.image.revision.vlabench"}}' "${image}")
if [[ "${image_arch}" != amd64 \
  || "${image_vlabench_revision}" != cf588fe60c0c7282174fe979f5913170cfe69017 ]]; then
  echo "RTX simulator image failed architecture/revision validation" >&2
  exit 2
fi

"${harness_python}" - "${gate_file}" "$(hostname)" "$(git rev-parse HEAD)" "${image}" "${image_id}" <<'PY'
import json
import hashlib
import socket
import sys
from pathlib import Path

gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert gate["schema_version"] == 1
assert gate["material_speedup"] is True
assert gate["host"] in {sys.argv[2], socket.gethostname()}
assert gate["git_commit"] == sys.argv[3]
assert gate["image"] == sys.argv[4]
assert gate["image_id"] == sys.argv[5]
assert gate["episodes"] == 3
assert gate["steps"] > 0
assert gate["speedup_over_gb10"] >= gate["minimum_required_speedup"]
database = Path(gate["recording_database"])
assert database.is_file(), database
assert hashlib.sha256(database.read_bytes()).hexdigest() == gate["recording_database_sha256"]
PY

IFS=, read -r -a gpu_array <<<"${gpu_ids}"
if [[ "${#gpu_array[@]}" -eq 0 ]]; then
  echo "XR1_GPU_IDS must name at least one GPU" >&2
  exit 2
fi
for gpu_id in "${gpu_array[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "XR1_GPU_IDS must be a comma-separated list of integer GPU IDs" >&2
    exit 2
  fi
  nvidia-smi --id="${gpu_id}" >/dev/null
done
server_count=${server_count:-${#gpu_array[@]}}
shard_count=${shard_count:-${server_count}}
for value_name in server_count shard_count; do
  value=${!value_name}
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer" >&2
    exit 2
  fi
done
if [[ "${server_count}" -gt "${shard_count}" ]]; then
  echo "XR1_SERVER_COUNT cannot exceed XR1_SHARD_COUNT" >&2
  exit 2
fi
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 2
fi
for reserved_session in xr1_audit_rtx xr1_checkpoint_rtx xr1_finalize_rtx; do
  if tmux has-session -t "${reserved_session}" 2>/dev/null; then
    echo "tmux session already exists: ${reserved_session}" >&2
    exit 2
  fi
done
if [[ -e "${run_root}" ]] || [[ -e "${result_root}" ]]; then
  echo "refusing to reuse run/result roots: ${run_root} ${result_root}" >&2
  exit 2
fi

mkdir -p "${run_root}/base" "${run_root}/provenance/src" \
  "${run_root}/provenance/scripts" \
  "${run_root}/provenance/configs/benchmarks/vlabench" \
  "${run_root}/provenance/configs/model_servers-xiaomi_robotics_1" \
  "${run_root}/provenance/environments" "${run_root}/provenance/docker" \
  "${result_root}"
cp "${bundle}/base/completed-manifest.json" "${run_root}/base/completed-manifest.json"
cp "${bundle}/prior-clean-dirs.relative.txt" "${run_root}/base/prior-clean-dirs.relative.txt"
prior_clean_dirs=
: >"${run_root}/base/prior-clean-dirs.txt"
while IFS= read -r relative_dir; do
  [[ -n "${relative_dir}" ]] || continue
  absolute_dir=${bundle}/${relative_dir}
  [[ -d "${absolute_dir}" ]] || { echo "missing prior clean directory: ${absolute_dir}" >&2; exit 2; }
  printf '%s\n' "${absolute_dir}" >>"${run_root}/base/prior-clean-dirs.txt"
  if [[ -z "${prior_clean_dirs}" ]]; then
    prior_clean_dirs=${absolute_dir}
  else
    prior_clean_dirs=${prior_clean_dirs}:${absolute_dir}
  fi
done <"${run_root}/base/prior-clean-dirs.relative.txt"
printf '%s\n' "${shard_count}" >"${run_root}/base/shard-count.txt"

cp -a src/sim_benchmarks "${run_root}/provenance/src/"
cp scripts/audit_vlabench_partial.py scripts/checkpoint_vlabench_progress.py \
  scripts/prepare_vlabench_retries.py scripts/finalize_xr1_vlabench_resume.sh \
  "${run_root}/provenance/scripts/"
cp configs/benchmarks/vlabench/xr1_official_resume.yaml \
  configs/benchmarks/vlabench/xr1_retry.yaml \
  "${run_root}/provenance/configs/benchmarks/vlabench/"
cp configs/model_servers/xiaomi_robotics_1/vlabench_rtx.yaml \
  "${run_root}/provenance/configs/model_servers-xiaomi_robotics_1/"
cp environments/xr1-rtx-x86_64.yaml "${run_root}/provenance/environments/"
cp docker/Dockerfile.vlabench "${run_root}/provenance/docker/"
cp "${gate_file}" "${run_root}/provenance/throughput-gate.json"
git rev-parse HEAD >"${run_root}/provenance/git-commit.txt"
git status --porcelain=v1 >"${run_root}/provenance/git-status.txt"
docker image inspect "${image}" >"${run_root}/provenance/docker-image-inspect.json"
nvidia-smi -q >"${run_root}/provenance/nvidia-smi-q.txt"
"${harness_python}" -m pip freeze >"${run_root}/provenance/pip-freeze.txt"
(
  cd "${run_root}/provenance"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)

eval_id=xr1-full-resume-rtx-${timestamp}
common_env=(
  "XR1_REPO_ROOT=${repo_root}"
  "XR1_PYTHON=${xr1_python}"
  "XR1_HARNESS_PYTHON=${harness_python}"
  "XR1_VLA_EVAL=${vla_eval}"
  "XR1_COMPLETED_MANIFEST=${run_root}/base/completed-manifest.json"
  "XR1_TRACK_DIR=${bundle}/original/provenance/tracks"
  "XR1_RUN_ROOT=${run_root}"
  "XR1_RESULT_ROOT=${result_root}"
  "XR1_EVAL_ID=${eval_id}"
  "XR1_BASE_PORT=${base_port}"
  "XR1_SHARD_COUNT=${shard_count}"
  "XR1_LOCAL_SHARD_COUNT=${shard_count}"
  "XR1_SHARD_OFFSET=0"
  "XR1_SERVER_COUNT=${server_count}"
  "XR1_SERVER_GPU_IDS=${gpu_ids}"
  "XR1_SIM_GPU_IDS=${gpu_ids}"
  "XR1_CPUS=${cpu_spec}"
  "XR1_CACHE_NAMESPACE=xr1-rtx-x86_64"
  "XR1_VLABENCH_IMAGE=${image}"
  "XR1_BENCHMARK_CONFIG=configs/benchmarks/vlabench/xr1_official_resume.yaml"
  "XR1_MODEL_CONFIG=configs/model_servers/xiaomi_robotics_1/vlabench_rtx.yaml"
)

quote_command() {
  local command=$1 item
  shift
  for item in "$@"; do
    printf -v command '%s %q' "${command}" "${item}"
  done
  printf '%s' "${command}"
}
runner_command=$(quote_command "cd $(printf %q "${repo_root}") && env" "${common_env[@]}" \
  bash scripts/run_xr1_vlabench_resume.sh)
runner_command+=" >$(printf %q "${run_root}/supervisor.log") 2>&1"
tmux new-session -d -s "${session_name}" "${runner_command}"

for _ in $(seq 1 240); do
  database_count=$(find "${result_root}" -name 'recording-*.sqlite' | wc -l)
  if [[ -f "${run_root}/eval_id.txt" && "${database_count}" -eq "${shard_count}" ]]; then
    break
  fi
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "RTX runner exited during startup; see ${run_root}/supervisor.log" >&2
    exit 1
  fi
  sleep 5
done
database_count=$(find "${result_root}" -name 'recording-*.sqlite' | wc -l)
if [[ "${database_count}" -ne "${shard_count}" ]]; then
  echo "runner did not create ${shard_count} recording databases" >&2
  exit 1
fi

monitor_env=("${common_env[@]}" "XR1_ORIGINAL_ROOT=${bundle}/original" \
  "XR1_PRIOR_CLEAN_DIRS=${prior_clean_dirs}" "XR1_SESSION_NAME=${session_name}")
audit_command=$(quote_command "cd $(printf %q "${repo_root}") && env" "${monitor_env[@]}" \
  bash scripts/monitor_xr1_vlabench_partial.sh)
audit_command+=" >$(printf %q "${run_root}/partial-audit-monitor.log") 2>&1"
checkpoint_command=$(quote_command "cd $(printf %q "${repo_root}") && env" "${monitor_env[@]}" \
  bash scripts/monitor_xr1_recovery_checkpoints.sh)
checkpoint_command+=" >$(printf %q "${run_root}/recovery-checkpoint-monitor.log") 2>&1"

frozen_model_config=${run_root}/provenance/configs/model_servers-xiaomi_robotics_1/vlabench_rtx.yaml
finalizer_env=("${monitor_env[@]}" "XR1_MODEL_CONFIG=${frozen_model_config}" \
  "XR1_REPORT_SRC=${run_root}/provenance/src" \
  "XR1_RETRY_CONFIG=${run_root}/provenance/configs/benchmarks/vlabench/xr1_retry.yaml")
finalizer_command=$(quote_command "cd $(printf %q "${repo_root}") && env" "${finalizer_env[@]}" \
  bash "${run_root}/provenance/scripts/finalize_xr1_vlabench_resume.sh")
finalizer_command+=" >$(printf %q "${run_root}/finalizer.log") 2>&1"

tmux new-session -d -s xr1_audit_rtx "${audit_command}"
tmux new-session -d -s xr1_checkpoint_rtx "${checkpoint_command}"
tmux new-session -d -s xr1_finalize_rtx "${finalizer_command}"
printf 'started=1\nbase_valid=150\npending=2310\nshards=%s\nservers=%s\ngpus=%s\nrun_root=%s\nresult_root=%s\n' \
  "${shard_count}" "${server_count}" "${gpu_ids}" "${run_root}" "${result_root}" \
  | tee "${run_root}/launch-status.txt"
