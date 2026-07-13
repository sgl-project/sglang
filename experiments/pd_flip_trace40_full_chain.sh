#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-preflight}"
ENV_FILE="${ENV_FILE:-/home/tiancij/trace40-full-chain.env}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing ENV_FILE: ${ENV_FILE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${ADMIN_API_KEY:?ADMIN_API_KEY is required}"
: "${IMAGE:?IMAGE is required}"
: "${SGLANG_REPO:?SGLANG_REPO is required}"
: "${TRACE_PATH:?TRACE_PATH is required}"
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"

case "${ADMIN_API_KEY}" in
  replace-with-*|changeme|CHANGE_ME) echo "unsafe ADMIN_API_KEY" >&2; exit 2 ;;
esac

HOSTS=(192.168.0.42 192.168.0.40 192.168.0.39 192.168.0.41)
NAMES=(node0 node1 node2 node3)
WORKER_CONTAINERS=(
  tiancij-pd-node0
  tiancij-pd-node1
  tiancij-pd-node2
  tiancij-pd-node3
)
NODE_URLS=(
  http://192.168.0.42:30000
  http://192.168.0.40:30000
  http://192.168.0.39:30000
  http://192.168.0.41:30000
)
ROUTER_URL="${ROUTER_URL:-http://192.168.0.42:8000}"
FIRST_MIGRATION_RATIO="${PD_FLIP_FIRST_MIGRATION_RATIO:-0.5}"
OBSERVATION_SECONDS="${PD_FLIP_OBSERVATION_SECONDS:-10}"
SLO_THRESHOLD="${PD_FLIP_SLO_THRESHOLD:-0.99}"
MIN_PREFILL_SAMPLES="${PD_FLIP_MIN_PREFILL_SLO_SAMPLES:-10}"
MIN_DECODE_SAMPLES="${PD_FLIP_MIN_DECODE_SLO_SAMPLES:-10}"
MONITOR_ITERATIONS="${PD_FLIP_MONITOR_ITERATIONS:-120}"
MONITOR_POLL_INTERVAL="${PD_FLIP_MONITOR_POLL_INTERVAL:-0.25}"
SOURCE_NAME="${PD_FLIP_SOURCE_NAME:-node2}"
MIGRATION_TARGET_NAME="${PD_FLIP_MIGRATION_TARGET_NAME:-node3}"
TRACE_WAVE_SIZE="${TRACE_WAVE_SIZE:-10}"
TRACE_WAVE_GAP_SECONDS="${TRACE_WAVE_GAP_SECONDS:-6}"
TRACE_INTRA_WAVE_INTERVAL_SECONDS="${TRACE_INTRA_WAVE_INTERVAL_SECONDS:-0.15}"
TRACE_TTFT_SLO_OVERRIDE_SECONDS="${TRACE_TTFT_SLO_OVERRIDE_SECONDS:-0}"
SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10)
CODE_HASH_FILES=(
  python/sglang/srt/disaggregation/decode.py
  python/sglang/srt/disaggregation/decode_hicache_mixin.py
  python/sglang/srt/managers/scheduler.py
  python/sglang/srt/managers/scheduler_components/invariant_checker.py
  python/sglang/srt/mem_cache/radix_cache.py
  scripts/playground/disaggregation/pd_flip_controller.py
  scripts/playground/disaggregation/pd_flip_trace_slo.py
  scripts/playground/disaggregation/pd_flip_migration_measure.py
  scripts/playground/disaggregation/pd_flip_prepare_trace.py
  scripts/playground/disaggregation/pd_flip_trace_replay.py
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

redact() {
  sed "s/${ADMIN_API_KEY//\\//\\\\/}/[REDACTED]/g"
}

remote() {
  local host="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] ssh %q ' "${host}"
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  ssh "${SSH_OPTIONS[@]}" "root@${host}" "$@"
}

remote_with_admin_key() {
  local host="$1"
  local command="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] ssh %q [admin-key-via-stdin] %q\n' "${host}" "${command}"
    return 0
  fi
  printf '%s\n' "${ADMIN_API_KEY}" |
    ssh "${SSH_OPTIONS[@]}" "root@${host}" "IFS= read -r ADMIN_KEY; ${command}"
}

if [[ -z "${RUN_ID:-}" ]]; then
  if [[ "${ACTION}" == "run" || "${ACTION}" == "preflight" ]]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-trace40-full-chain"
  else
    RUN_ID="$(remote "${HOSTS[0]}" "cat '${ARTIFACT_ROOT}/trace40-current-run'")"
  fi
fi
RUN_DIR="${ARTIFACT_ROOT}/${RUN_ID}"
SAFE_RUN_ID="${RUN_ID//[^a-zA-Z0-9_.-]/-}"
MEASURE_CONTAINER="tiancij-pd-${SAFE_RUN_ID}-measure"
WORKLOAD_CONTAINER="tiancij-pd-${SAFE_RUN_ID}-workload"
CONTROLLER_CONTAINER="tiancij-pd-${SAFE_RUN_ID}-controller"
LEDGER="${RUN_DIR}/workload/trace_slo_ledger.jsonl"
EFFECTIVE_TRACE="${RUN_DIR}/trace/trace40_scheduled.jsonl"

timeline_event() {
  local event="$1"
  local status="${2:-ok}"
  local epoch_ns mono_ns
  epoch_ns="$(date +%s%N)"
  mono_ns="$(awk '{printf "%.0f", $1 * 1000000000}' /proc/uptime)"
  printf '{"run_id":"%s","event":"%s","status":"%s","epoch_ns":%s,"mono_ns":%s}\n' \
    "${RUN_ID}" "${event}" "${status}" "${epoch_ns}" "${mono_ns}" \
    >> "${RUN_DIR}/runner_timeline.jsonl"
}

node_is_configured() {
  local candidate="$1"
  local node_name
  for node_name in "${NAMES[@]}"; do
    [[ "${candidate}" == "${node_name}" ]] && return 0
  done
  return 1
}

validate_execution_layout() {
  if ! node_is_configured "${SOURCE_NAME}"; then
    echo "unknown PD_FLIP_SOURCE_NAME: ${SOURCE_NAME}" >&2
    return 2
  fi
  if ! node_is_configured "${MIGRATION_TARGET_NAME}"; then
    echo "unknown PD_FLIP_MIGRATION_TARGET_NAME: ${MIGRATION_TARGET_NAME}" >&2
    return 2
  fi
  if [[ "${SOURCE_NAME}" == "${MIGRATION_TARGET_NAME}" ]]; then
    echo "PD_FLIP_SOURCE_NAME and PD_FLIP_MIGRATION_TARGET_NAME must differ" >&2
    return 2
  fi
  case "${ARTIFACT_ROOT}" in
    /home/tiancij|/home/tiancij/*) ;;
    *) echo "ARTIFACT_ROOT must be under /home/tiancij for container visibility" >&2; return 2 ;;
  esac
  if [[ "${DRY_RUN}" != "1" ]]; then
    hostname -I | tr ' ' '\n' | grep -Fxq "${HOSTS[0]}" || {
      echo "run this script on host0 (${HOSTS[0]})" >&2
      return 2
    }
  fi
}

preflight() {
  log "preflight run_id=${RUN_ID}"
  validate_execution_layout
  for i in "${!HOSTS[@]}"; do
    remote "${HOSTS[$i]}" "test -d '${SGLANG_REPO}' && docker image inspect '${IMAGE}' >/dev/null && docker inspect '${WORKER_CONTAINERS[$i]}' >/dev/null && (chronyc tracking 2>/dev/null || timedatectl status 2>/dev/null || true)"
  done
  remote "${HOSTS[0]}" "test -s '${TRACE_PATH}' && docker inspect tiancij-pd-router >/dev/null && docker inspect tiancij-pd-store >/dev/null && docker inspect tiancij-pd-master >/dev/null && docker inspect tiancij-pd-metadata >/dev/null"
  remote "${HOSTS[0]}" "python3 -c \"import json; rows=[json.loads(line) for line in open('${TRACE_PATH}', encoding='utf-8') if line.strip()]; assert len(rows) == 40; assert [row.get('prompt_kind') for row in rows] == ['long','short'] * 20; assert sum(row.get('prompt_chars') == 10000 for row in rows) == 20; assert sum(row.get('prompt_chars') == 1000 for row in rows) == 20; assert all(float(row.get('ttft_slo_s', 0)) > 0 and float(row.get('tpot_slo_s', 0)) > 0 for row in rows)\""
  if [[ "${DRY_RUN}" != "1" ]]; then
    for i in "${!HOSTS[@]}"; do
      remote "${HOSTS[$i]}" "docker inspect '${WORKER_CONTAINERS[$i]}' --format '{{json .Config.Cmd}}' | python3 -c \"import base64,json,sys; cmd=' '.join(json.load(sys.stdin)); parts=cmd.split(); pos=parts.index('echo') if 'echo' in parts else -1; decoded=base64.b64decode(parts[pos + 1]).decode('utf-8') if pos >= 0 and len(parts) > pos + 1 else cmd; assert '--enable-pd-flip-hicache-stitch' in decoded\""
      remote "${HOSTS[$i]}" "docker inspect '${WORKER_CONTAINERS[$i]}' --format '{{range .Mounts}}{{println .Source .Destination}}{{end}}' | grep -Fq '${SGLANG_REPO} /sgl-workspace/sglang'"
    done
    local hash_file_args reference_hash node_hash
    printf -v hash_file_args " '%s'" "${CODE_HASH_FILES[@]}"
    reference_hash="$(remote "${HOSTS[0]}" "cd '${SGLANG_REPO}' && sha256sum${hash_file_args} | sha256sum | cut -d' ' -f1")"
    for i in "${!HOSTS[@]}"; do
      node_hash="$(remote "${HOSTS[$i]}" "cd '${SGLANG_REPO}' && sha256sum${hash_file_args} | sha256sum | cut -d' ' -f1")"
      [[ "${node_hash}" == "${reference_hash}" ]] || {
        echo "critical code hash mismatch on ${NAMES[$i]}" >&2
        return 2
      }
    done
  fi
  log "preflight complete"
}

validate_admin_access() {
  for i in "${!HOSTS[@]}"; do
    remote_with_admin_key "${HOSTS[$i]}" "test -n \"\$ADMIN_KEY\"; curl -fsS -H \"Authorization: Bearer \$ADMIN_KEY\" 'http://127.0.0.1:30000/pd_flip/runtime_role/status' >/dev/null"
  done
}

wait_http() {
  local host="$1"
  local url="$2"
  remote "${host}" "deadline=\$((SECONDS+1800)); until curl -fsS '${url}' >/dev/null; do (( SECONDS < deadline )) || exit 1; sleep 5; done"
}

capture_clocks() {
  remote "${HOSTS[0]}" "mkdir -p '${RUN_DIR}/clock'"
  for i in "${!HOSTS[@]}"; do
    remote "${HOSTS[$i]}" "(hostname; date +%s%N; date --iso-8601=ns; chronyc tracking 2>/dev/null || timedatectl status 2>/dev/null || true)"       > "${RUN_DIR}/clock/${NAMES[$i]}.txt"
  done
}

prepare_scheduled_trace() {
  # prepare-trace-in-container: host Python on the ECS nodes is too old.
  remote "${HOSTS[0]}" "mkdir -p '${RUN_DIR}/trace'; docker run --rm -v '${SGLANG_REPO}:/sgl-workspace/sglang' -v /home/tiancij:/home/tiancij '${IMAGE}' python3 /sgl-workspace/sglang/scripts/playground/disaggregation/pd_flip_prepare_trace.py --source '${TRACE_PATH}' --output '${EFFECTIVE_TRACE}' --manifest '${RUN_DIR}/trace/schedule.json' --wave-size '${TRACE_WAVE_SIZE}' --wave-gap-seconds '${TRACE_WAVE_GAP_SECONDS}' --intra-wave-interval-seconds '${TRACE_INTRA_WAVE_INTERVAL_SECONDS}' --ttft-slo-override-seconds '${TRACE_TTFT_SLO_OVERRIDE_SECONDS}'; sha256sum '${TRACE_PATH}' > '${RUN_DIR}/trace/source.sha256'; sha256sum '${EFFECTIVE_TRACE}' > '${RUN_DIR}/trace/effective.sha256'"
}

start_shared_stack() {
  remote "${HOSTS[0]}" "docker start tiancij-pd-metadata tiancij-pd-master tiancij-pd-store >/dev/null"
  remote "${HOSTS[0]}" "docker start tiancij-pd-node0 >/dev/null"
  remote "${HOSTS[1]}" "docker start tiancij-pd-node1 >/dev/null"
  remote "${HOSTS[2]}" "docker start tiancij-pd-node2 >/dev/null"
  remote "${HOSTS[3]}" "docker start tiancij-pd-node3 >/dev/null"
  for i in "${!HOSTS[@]}"; do
    wait_http "${HOSTS[$i]}" "http://127.0.0.1:30000/health"
  done
  # Router classifies workers only during discovery. Restart it after every
  # worker is healthy so a slow prefill startup cannot be cached as an empty
  # model/Plain worker for the whole experiment.
  remote "${HOSTS[0]}" "docker restart tiancij-pd-router >/dev/null"
  wait_http "${HOSTS[0]}" "http://127.0.0.1:8000/v1/models"
}

save_initial_status() {
  remote "${HOSTS[0]}" "mkdir -p '${RUN_DIR}/status'; KEY=\$(docker inspect tiancij-pd-node0 --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^ADMIN_API_KEY=//p'); for item in 'node0 192.168.0.42' 'node1 192.168.0.40' 'node2 192.168.0.39' 'node3 192.168.0.41'; do set -- \$item; curl -fsS -H \"Authorization: Bearer \$KEY\" \"http://\$2:30000/pd_flip/runtime_role/status\" > '${RUN_DIR}/status/'\$1'-before.json'; curl -fsS -H \"Authorization: Bearer \$KEY\" \"http://\$2:30000/pd_flip/migration/status\" > '${RUN_DIR}/status/'\$1'-migration-before.json'; done"
}

start_measurement() {
  remote "${HOSTS[0]}" "! docker inspect '${MEASURE_CONTAINER}' >/dev/null 2>&1; KEY=\$(docker inspect tiancij-pd-node0 --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^ADMIN_API_KEY=//p'); test -n \"\$KEY\"; docker run -d --name '${MEASURE_CONTAINER}' --label 'pd-flip.run-id=${RUN_ID}' --network host -e ADMIN_API_KEY=\"\$KEY\" -e PD_FLIP_ROUTER_ADMIN_API_KEY=\"\$KEY\" -v '${SGLANG_REPO}:/sgl-workspace/sglang' -v /home/tiancij:/home/tiancij '${IMAGE}' bash -lc \"cd /sgl-workspace/sglang && PYTHONPATH=python python3 scripts/playground/disaggregation/pd_flip_migration_measure.py sample --router-url http://127.0.0.1:8000 --node name=node0,worker_url=http://192.168.0.42:30000 --node name=node1,worker_url=http://192.168.0.40:30000 --node name=node2,worker_url=http://192.168.0.39:30000 --node name=node3,worker_url=http://192.168.0.41:30000 --api-key-env ADMIN_API_KEY --router-api-key-env PD_FLIP_ROUTER_ADMIN_API_KEY --interval-seconds 0.05 --duration-seconds 900 --output-events '${RUN_DIR}/metrics/events.jsonl'\""
}

start_workload() {
  remote "${HOSTS[0]}" "! docker inspect '${WORKLOAD_CONTAINER}' >/dev/null 2>&1; docker run -d --name '${WORKLOAD_CONTAINER}' --label 'pd-flip.run-id=${RUN_ID}' --network host -v '${SGLANG_REPO}:/sgl-workspace/sglang' -v /home/tiancij:/home/tiancij '${IMAGE}' bash -lc \"cd /sgl-workspace/sglang && PYTHONPATH=python python3 scripts/playground/disaggregation/pd_flip_trace_replay.py replay --trace-jsonl '${EFFECTIVE_TRACE}' --router-url http://127.0.0.1:8000 --mode state_machine --output-dir '${RUN_DIR}/workload' --ledger-path '${LEDGER}' --timeout-seconds 900 --max-workers 40\""
  remote "${HOSTS[0]}" "deadline=\$((SECONDS+180)); until test -s '${LEDGER}'; do (( SECONDS < deadline )) || exit 1; sleep 0.1; done"
}

start_controller() {
  # The target_hicache_restore and fallback phases remain enabled and are timed
  # independently by Worker request_measurements and the 50 ms sidecar.
  remote "${HOSTS[0]}" "! docker inspect '${CONTROLLER_CONTAINER}' >/dev/null 2>&1; KEY=\$(docker inspect tiancij-pd-node0 --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^ADMIN_API_KEY=//p'); test -n \"\$KEY\"; docker run -d --name '${CONTROLLER_CONTAINER}' --label 'pd-flip.run-id=${RUN_ID}' --network host -e ADMIN_API_KEY=\"\$KEY\" -v '${SGLANG_REPO}:/sgl-workspace/sglang' -v /home/tiancij:/home/tiancij '${IMAGE}' bash -lc \"cd /sgl-workspace/sglang && PYTHONPATH=python python3 scripts/playground/disaggregation/pd_flip_controller.py --router-url http://127.0.0.1:8000 --node name=node0,worker_url=http://192.168.0.42:30000,router_worker_id=http://192.168.0.42:30000,bootstrap_port=8998 --node name=node1,worker_url=http://192.168.0.40:30000,router_worker_id=http://192.168.0.40:30000,bootstrap_port=8998 --node name=node2,worker_url=http://192.168.0.39:30000,router_worker_id=http://192.168.0.39:30000,bootstrap_port=8998 --node name=node3,worker_url=http://192.168.0.41:30000,router_worker_id=http://192.168.0.41:30000,bootstrap_port=8998 --api-key-env ADMIN_API_KEY --first-migration-ratio '${FIRST_MIGRATION_RATIO}' --observation-seconds '${OBSERVATION_SECONDS}' --slo-threshold '${SLO_THRESHOLD}' --min-prefill-slo-samples '${MIN_PREFILL_SAMPLES}' --min-decode-slo-samples '${MIN_DECODE_SAMPLES}' --session-journal-path '${RUN_DIR}/controller/pd_flip_session.json' --session-id-prefix '${RUN_ID}' monitor-progressive --trace-slo-ledger '${LEDGER}' --source-name '${SOURCE_NAME}' --migration-target-name '${MIGRATION_TARGET_NAME}' --iterations '${MONITOR_ITERATIONS}' --poll-interval '${MONITOR_POLL_INTERVAL}'\""
}

collect() {
  remote "${HOSTS[0]}" "mkdir -p '${RUN_DIR}/logs' '${RUN_DIR}/controller' '${RUN_DIR}/report'; docker logs '${WORKLOAD_CONTAINER}' > '${RUN_DIR}/workload/container.log' 2>&1 || true; docker logs '${CONTROLLER_CONTAINER}' > '${RUN_DIR}/controller/controller.log' 2>&1 || true; docker logs '${MEASURE_CONTAINER}' > '${RUN_DIR}/metrics/container.log' 2>&1 || true; docker logs tiancij-pd-router > '${RUN_DIR}/logs/router.log' 2>&1 || true; docker logs tiancij-pd-store > '${RUN_DIR}/logs/mooncake-store.log' 2>&1 || true"
  for i in "${!HOSTS[@]}"; do
    remote "${HOSTS[$i]}" "docker logs '${WORKER_CONTAINERS[$i]}' 2>&1"       > "${RUN_DIR}/logs/${NAMES[$i]}.log" || true
  done
  remote "${HOSTS[0]}" "KEY=\$(docker inspect tiancij-pd-node0 --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^ADMIN_API_KEY=//p'); for item in 'node0 192.168.0.42' 'node1 192.168.0.40' 'node2 192.168.0.39' 'node3 192.168.0.41'; do set -- \$item; curl -fsS -H \"Authorization: Bearer \$KEY\" \"http://\$2:30000/pd_flip/runtime_role/status\" > '${RUN_DIR}/status/'\$1'-after.json' || true; done; docker run --rm --network host -v '${SGLANG_REPO}:/sgl-workspace/sglang' -v /home/tiancij:/home/tiancij '${IMAGE}' bash -lc \"cd /sgl-workspace/sglang && PYTHONPATH=python python3 scripts/playground/disaggregation/pd_flip_migration_measure.py summarize --events-jsonl '${RUN_DIR}/metrics/events.jsonl' --controller-log '${RUN_DIR}/controller/controller.log' --request-metrics-jsonl '${RUN_DIR}/workload/state_machine/request_metrics.jsonl' --errors-jsonl '${RUN_DIR}/workload/state_machine/errors.jsonl' --output-dir '${RUN_DIR}/report'\" || true"
  log "artifacts: ${RUN_DIR}"
}

validate_artifacts() {
  remote "${HOSTS[0]}" "python3 -c \"import json; ledger=[json.loads(line) for line in open('${LEDGER}', encoding='utf-8') if line.strip()]; final=[row for row in ledger if row.get('status') != 'running']; metrics=[json.loads(line) for line in open('${RUN_DIR}/workload/state_machine/request_metrics.jsonl', encoding='utf-8') if line.strip()]; assert len(final) == 40, len(final); assert len({row['request_id'] for row in final}) == 40; assert len(metrics) == 40, len(metrics); json.dump({'ledger_records': len(ledger), 'final_ledger_records': len(final), 'request_metric_records': len(metrics), 'valid': True}, open('${RUN_DIR}/report/raw_validation.json', 'w', encoding='utf-8'), indent=2, sort_keys=True)\""
}

archive() {
  remote "${HOSTS[0]}" "tar -C '${ARTIFACT_ROOT}' -czf '/home/tiancij/${SAFE_RUN_ID}.tar.gz' '${RUN_ID}'"
}

run_experiment() {
  validate_execution_layout
  remote "${HOSTS[0]}" "test ! -e '${RUN_DIR}'; mkdir -p '${RUN_DIR}/workload' '${RUN_DIR}/controller' '${RUN_DIR}/metrics' '${RUN_DIR}/logs' '${RUN_DIR}/status'; printf '%s\n' '${RUN_ID}' > '${ARTIFACT_ROOT}/trace40-current-run'; (git -C '${SGLANG_REPO}' rev-parse HEAD 2>/dev/null || printf '%s\n' git-unavailable) > '${RUN_DIR}/git-commit.txt'"
  timeline_event runner_started
  trap 'handle_run_error $?' ERR
  timeline_event preflight_started
  preflight
  timeline_event preflight_finished
  prepare_scheduled_trace
  timeline_event scheduled_trace_ready
  capture_clocks
  timeline_event clocks_captured
  start_shared_stack
  timeline_event shared_stack_ready
  validate_admin_access
  timeline_event admin_access_validated
  save_initial_status
  timeline_event initial_status_saved
  start_measurement
  timeline_event measurement_started
  start_workload
  timeline_event workload_ledger_ready
  start_controller
  timeline_event controller_started
  controller_exit="$(remote "${HOSTS[0]}" "docker wait '${CONTROLLER_CONTAINER}'")"
  printf '%s\n' "${controller_exit}" > "${RUN_DIR}/controller/exit.txt"
  timeline_event controller_finished "exit_${controller_exit}"
  workload_exit="$(remote "${HOSTS[0]}" "docker wait '${WORKLOAD_CONTAINER}'")"
  printf '%s\n' "${workload_exit}" > "${RUN_DIR}/workload/exit.txt"
  timeline_event workload_finished "exit_${workload_exit}"
  remote "${HOSTS[0]}" "docker stop '${MEASURE_CONTAINER}' >/dev/null 2>&1 || true"
  timeline_event measurement_stopped
  collect
  validate_artifacts
  timeline_event artifacts_validated
  timeline_event collection_finished
  timeline_event archive_started
  archive
  timeline_event archive_finished
  archive
  trap - ERR
  [[ "${controller_exit}" == "0" && "${workload_exit}" == "0" ]]
}

handle_run_error() {
  local exit_code="$1"
  trap - ERR
  set +e
  timeline_event runner_failed "exit_${exit_code}"
  stop_run_owned
  timeline_event run_owned_stopped_after_failure
  collect
  timeline_event failure_collection_finished
  timeline_event archive_started_after_failure
  archive
  timeline_event archive_finished_after_failure
  archive
  exit "${exit_code}"
}

status() {
  for i in "${!HOSTS[@]}"; do
    remote "${HOSTS[$i]}" "docker ps -a --filter name='${WORKER_CONTAINERS[$i]}' --format '{{.Names}} {{.Status}}'"
  done
  remote "${HOSTS[0]}" "docker ps -a --filter name='tiancij-pd-${SAFE_RUN_ID}-' --format '{{.Names}} {{.Status}}'"
}

stop_run_owned() {
  remote "${HOSTS[0]}" "ids=\$(docker ps -aq --filter 'label=pd-flip.run-id=${RUN_ID}'); if test -n \"\$ids\"; then docker stop \$ids >/dev/null 2>&1 || true; fi"
  log "stopped only run-owned containers for ${RUN_ID}"
}

case "${ACTION}" in
  preflight) preflight ;;
  run) run_experiment ;;
  status) status ;;
  collect) collect ;;
  stop) stop_run_owned ;;
  *) echo "usage: $0 preflight|run|status|collect|stop" >&2; exit 2 ;;
esac
