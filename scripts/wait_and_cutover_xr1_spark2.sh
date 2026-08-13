#!/usr/bin/env bash
set -euo pipefail

repo_root=${XR1_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
remote_host=${XR1_REMOTE_HOST:-spark2}
session_name=${XR1_SESSION_NAME:-xr1_resume}
status_root=${XR1_SPARK2_WATCH_ROOT:-/data/shared2/user/kampta/logs/sim_benchmarks/vlabench/xr1_spark2_readiness}
minimum_available_kib=${XR1_REMOTE_MIN_AVAILABLE_KIB:-52428800}
stable_passes_required=${XR1_REMOTE_STABLE_PASSES:-3}
interval_seconds=${XR1_REMOTE_WATCH_INTERVAL_SECONDS:-300}
stable_passes=0

if [[ "${stable_passes_required}" -lt 1 || "${interval_seconds}" -lt 60 ]]; then
  echo "stable passes must be positive and watch interval must be at least 60 seconds" >&2
  exit 2
fi
mkdir -p "${status_root}"

while tmux has-session -t "${session_name}" 2>/dev/null; do
  readiness=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${remote_host}" '
    set -euo pipefail
    container=$(docker ps --filter name=^/minimax-h3-fl2va$ -q)
    available_kib=$(awk "/MemAvailable/ {print \$2}" /proc/meminfo)
    compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed "/^[[:space:]]*$/d" | paste -sd, -)
    printf "container=%s available_kib=%s compute_pids=%s" \
      "${container:+running}" "${available_kib}" "${compute_pids}"
  ' 2>&1) || readiness="ssh_error=${readiness:-unknown}"

  container=$(sed -n 's/.*container=\([^ ]*\).*/\1/p' <<<"${readiness}")
  available_kib=$(sed -n 's/.*available_kib=\([0-9]*\).*/\1/p' <<<"${readiness}")
  compute_pids=$(sed -n 's/.*compute_pids=\([^ ]*\).*/\1/p' <<<"${readiness}")
  if [[ -z "${container}" && -z "${compute_pids}" && -n "${available_kib}" \
    && "${available_kib}" -ge "${minimum_available_kib}" ]]; then
    stable_passes=$((stable_passes + 1))
  else
    stable_passes=0
  fi
  printf '%s %s stable_passes=%s/%s\n' \
    "$(date -Is)" "${readiness}" "${stable_passes}" "${stable_passes_required}" \
    >"${status_root}/status.txt"

  if [[ "${stable_passes}" -ge "${stable_passes_required}" ]]; then
    printf '%s invoking transactional dual-Spark cutover\n' "$(date -Is)" \
      >>"${status_root}/watch.log"
    cutover_rc=0
    "${repo_root}/scripts/cutover_xr1_vlabench_dual_spark.sh" \
      >>"${status_root}/cutover.log" 2>&1 || cutover_rc=$?
    if [[ "${cutover_rc}" -eq 0 ]]; then
      printf '%s cutover complete\n' "$(date -Is)" >"${status_root}/status.txt"
      exit 0
    fi
    printf '%s cutover failed exit_code=%s\n' "$(date -Is)" "${cutover_rc}" \
      >>"${status_root}/watch.log"
    [[ "${cutover_rc}" -eq 3 ]] || exit "${cutover_rc}"
    stable_passes=0
  fi

  remaining=${interval_seconds}
  while [[ "${remaining}" -gt 0 ]] && tmux has-session -t "${session_name}" 2>/dev/null; do
    slice=60
    [[ "${remaining}" -lt "${slice}" ]] && slice=${remaining}
    sleep "${slice}"
    remaining=$((remaining - slice))
  done
done

printf '%s local XR-1 session ended before Spark2 became available\n' "$(date -Is)" \
  >"${status_root}/status.txt"
