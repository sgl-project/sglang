#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="${1:-/root/sglang/experiments/pd_flip_waiting_queue_$(date +%Y%m%d_%H%M%S)}"
REPO="${REPO:-/root/sglang}"
DOCKER_DIR="${REPO}/scripts/playground/disaggregation/pd_flip_docker"
BASE_ENV="${DOCKER_DIR}/env.local"
WAIT_ENV="${DOCKER_DIR}/env.waiting_queue"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
PD_FLIP_MIGRATION_MAX_REQS="${PD_FLIP_MIGRATION_MAX_REQS:-}"
REQUESTS="${REQUESTS:-40}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-0.5}"
SHORT_CHARS="${SHORT_CHARS:-1000}"
LONG_CHARS="${LONG_CHARS:-10000}"
SHORT_COUNT="${SHORT_COUNT:-20}"
LONG_COUNT="${LONG_COUNT:-20}"
FORCE_DELAY_SECONDS="${FORCE_DELAY_SECONDS:-8}"
POST_UNPIN_DELAY_SECONDS="${POST_UNPIN_DELAY_SECONDS:-0}"
SAMPLER_DURATION_SECONDS="${SAMPLER_DURATION_SECONDS:-420}"
REPLAY_TIMEOUT_SECONDS="${REPLAY_TIMEOUT_SECONDS:-900}"
EXTRA_SGLANG_ARGS_SUFFIX="${EXTRA_SGLANG_ARGS_SUFFIX:-}"
EXTRA_DOCKER_ARGS_SUFFIX="${EXTRA_DOCKER_ARGS_SUFFIX:-}"
PD_FLIP_OBSERVE_QUIESCE_SECONDS="${PD_FLIP_OBSERVE_QUIESCE_SECONDS:-15}"
PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS="${PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS:-2}"
SKIP_CLUSTER_START="${SKIP_CLUSTER_START:-0}"
PIN_NODE3_DRAIN_FOR_WAITING="${PIN_NODE3_DRAIN_FOR_WAITING:-1}"
PIN_DRAIN_FOR_WAITING="${PIN_DRAIN_FOR_WAITING:-${PIN_NODE3_DRAIN_FOR_WAITING}}"
RUN_NAME="${RUN_NAME:-01_waiting_queue_two_phase}"
MODE="${MODE:-waiting_queue_state_machine}"
TRACE_DIR_NAME="${TRACE_DIR_NAME:-trace_waiting_queue}"
FORCE_SOURCE_NAME="${FORCE_SOURCE_NAME:-node2}"
MIGRATION_TARGET_NAME="${MIGRATION_TARGET_NAME:-}"
INITIAL_ROLE_NODE0="${INITIAL_ROLE_NODE0:-prefill}"
INITIAL_ROLE_NODE1="${INITIAL_ROLE_NODE1:-prefill}"
INITIAL_ROLE_NODE2="${INITIAL_ROLE_NODE2:-decode}"
INITIAL_ROLE_NODE3="${INITIAL_ROLE_NODE3:-decode}"

mkdir -p "${SUITE_DIR}"
exec > >(tee -a "${SUITE_DIR}/waiting_queue_runner.log") 2>&1

source "${BASE_ENV}"

cat "${BASE_ENV}" > "${WAIT_ENV}"
cat >> "${WAIT_ENV}" <<ENV

EXTRA_SGLANG_ARGS='--trust-remote-code --enable-metrics --served-model-name ${MODEL_ID} --max-running-requests ${MAX_RUNNING_REQUESTS} ${EXTRA_SGLANG_ARGS_SUFFIX}'
EXTRA_DOCKER_ARGS='${EXTRA_DOCKER_ARGS} ${EXTRA_DOCKER_ARGS_SUFFIX}'
ENV

TRACE="${REPO}/scripts/playground/disaggregation/pd_flip_trace_replay.py"
MEASURE="${REPO}/scripts/playground/disaggregation/pd_flip_migration_measure.py"
CTRL="${REPO}/scripts/playground/disaggregation/pd_flip_controller.py"
DIAGRAM="${REPO}/experiments/make_pd_state_machine_latency_diagram.py"
ROUTER_URL="http://127.0.0.1:${ROUTER_PORT}"

NODE_NAMES=(node0 node1 node2 node3)
NODE_HOSTS=(192.168.0.42 192.168.0.40 192.168.0.39 192.168.0.41)
NODE_URLS=("${NODE0}" "${NODE1}" "${NODE2}" "${NODE3}")
NODE_SESSIONS=(pd-node0 pd-node1 pd-node2 pd-node3)
INITIAL_ROLES=("${INITIAL_ROLE_NODE0}" "${INITIAL_ROLE_NODE1}" "${INITIAL_ROLE_NODE2}" "${INITIAL_ROLE_NODE3}")
PIN_DRAIN_WORKER_IDS="${PIN_DRAIN_WORKER_IDS:-${NODE3}}"
if [[ -z "${UNPIN_BEFORE_FLIP_WORKER_IDS+x}" ]]; then
  if [[ -n "${MIGRATION_TARGET_NAME}" ]]; then
    UNPIN_BEFORE_FLIP_WORKER_IDS=""
  else
    UNPIN_BEFORE_FLIP_WORKER_IDS="${PIN_DRAIN_WORKER_IDS}"
  fi
fi

NODE_ARGS=(
  --node "name=node0,worker_url=${NODE0},router_worker_id=${NODE0},bootstrap_port=${BOOTSTRAP_PORT}"
  --node "name=node1,worker_url=${NODE1},router_worker_id=${NODE1},bootstrap_port=${BOOTSTRAP_PORT}"
  --node "name=node2,worker_url=${NODE2},router_worker_id=${NODE2},bootstrap_port=${BOOTSTRAP_PORT}"
  --node "name=node3,worker_url=${NODE3},router_worker_id=${NODE3},bootstrap_port=${BOOTSTRAP_PORT}"
)

MEASURE_NODE_ARGS=(
  --node "name=node0,worker_url=${NODE0}"
  --node "name=node1,worker_url=${NODE1}"
  --node "name=node2,worker_url=${NODE2}"
  --node "name=node3,worker_url=${NODE3}"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

ssh_node() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" "$@"
}

distribute_wait_env() {
  log "distributing ${WAIT_ENV} to all nodes"
  for host in "${NODE_HOSTS[@]}"; do
    ssh_node "${host}" "mkdir -p '${DOCKER_DIR}'"
    scp -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
      "${WAIT_ENV}" "${host}:${WAIT_ENV}"
  done
}

wait_workers_ready() {
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    local ok=1
    for url in "${NODE_URLS[@]}"; do
      if ! curl -fsS "${url}/server_info" >/dev/null 2>&1; then
        ok=0
        break
      fi
    done
    if (( ok == 1 )); then
      log "workers ready"
      return 0
    fi
    sleep 10
  done
  log "workers not ready before timeout"
  return 1
}

wait_router_ready() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -fsS "${ROUTER_URL}/pd_flip/router/workers" >/dev/null 2>&1; then
      log "router ready"
      return 0
    fi
    sleep 5
  done
  log "router not ready before timeout"
  return 1
}

start_workers() {
  local label="$1"
  log "starting workers label=${label} max_running=${MAX_RUNNING_REQUESTS} migration_max_reqs=${PD_FLIP_MIGRATION_MAX_REQS:-unbounded}"
  for i in "${!NODE_HOSTS[@]}"; do
    local host="${NODE_HOSTS[$i]}"
    local role="${INITIAL_ROLES[$i]}"
    local session="${NODE_SESSIONS[$i]}"
    ssh_node "${host}" "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t '${session}' 2>/dev/null || true; fi; if [ -f '${DOCKER_DIR}/${session}.pid' ]; then kill \$(cat '${DOCKER_DIR}/${session}.pid') 2>/dev/null || true; rm -f '${DOCKER_DIR}/${session}.pid'; fi; fuser -k ${PORT}/tcp 2>/dev/null || true; cd '${DOCKER_DIR}'; rm -f 'worker.${label}.log'; if command -v tmux >/dev/null 2>&1; then tmux new -d -s '${session}' 'cd ${DOCKER_DIR}; ENV_FILE=${WAIT_ENV} ENABLE_PD_FLIP_STATE_MACHINE=1 ENABLE_PD_RUNTIME_ROLE_SWITCH=1 ./run_worker.sh ${role} 0.0.0.0 2>&1 | tee worker.${label}.log'; else nohup bash -lc 'cd ${DOCKER_DIR}; ENV_FILE=${WAIT_ENV} ENABLE_PD_FLIP_STATE_MACHINE=1 ENABLE_PD_RUNTIME_ROLE_SWITCH=1 ./run_worker.sh ${role} 0.0.0.0' > 'worker.${label}.log' 2>&1 < /dev/null & echo \$! > '${session}.pid'; fi"
  done
  wait_workers_ready
}

start_router() {
  local label="$1"
  log "starting router label=${label}"
  ssh_node 192.168.0.42 "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t pd-router 2>/dev/null || true; fi; if [ -f '${DOCKER_DIR}/pd-router.pid' ]; then kill \$(cat '${DOCKER_DIR}/pd-router.pid') 2>/dev/null || true; rm -f '${DOCKER_DIR}/pd-router.pid'; fi; fuser -k ${ROUTER_PORT}/tcp 2>/dev/null || true; cd '${DOCKER_DIR}'; rm -f 'router.${label}.log'; if command -v tmux >/dev/null 2>&1; then tmux new -d -s pd-router 'cd ${DOCKER_DIR}; ENV_FILE=${WAIT_ENV} ./run_router.sh 2>&1 | tee router.${label}.log'; else nohup bash -lc 'cd ${DOCKER_DIR}; ENV_FILE=${WAIT_ENV} ./run_router.sh' > 'router.${label}.log' 2>&1 < /dev/null & echo \$! > pd-router.pid; fi"
  wait_router_ready
}

controller() {
  docker run --rm --network host \
    -e PD_FLIP_MIGRATION_MAX_REQS="${PD_FLIP_MIGRATION_MAX_REQS}" \
    -e PD_FLIP_OBSERVE_QUIESCE_SECONDS="${PD_FLIP_OBSERVE_QUIESCE_SECONDS}" \
    -e PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS="${PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS}" \
    -v /root/sglang:/root/sglang \
    "${IMAGE}" \
    python3 "${CTRL}" \
    --router-url "${ROUTER_URL}" \
    --timeout-seconds 20 \
    "${NODE_ARGS[@]}" \
    "$@"
}

save_roles() {
  local out="$1"
  curl -fsS "${ROUTER_URL}/pd_flip/router/workers" > "${out}" || true
}

set_router_drain() {
  local worker_id="$1"
  local draining="$2"
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d "{\"worker_id\":\"${worker_id}\",\"draining\":${draining}}" \
    "${ROUTER_URL}/pd_flip/router/worker/drain" >/dev/null
}

summarize_run() {
  local run_dir="$1"
  local mode="$2"
  mkdir -p "${run_dir}/migration_link"
  python3 "${MEASURE}" summarize \
    --events-jsonl "${run_dir}/migration_events.jsonl" \
    --output-dir "${run_dir}/migration_link" \
    --controller-log "${run_dir}/controller.log" \
    --request-metrics-jsonl "${run_dir}/${mode}/request_metrics.jsonl" \
    --errors-jsonl "${run_dir}/${mode}/errors.jsonl" \
    > "${run_dir}/migration_link_summary.log" 2>&1 || true
  python3 "${TRACE}" summarize --output-dir "${run_dir}" --modes "${mode}" > "${run_dir}/trace_summary.log" 2>&1 || true
}

write_manifest() {
  local migration_max_reqs_json="null"
  if [[ -n "${PD_FLIP_MIGRATION_MAX_REQS}" ]]; then
    migration_max_reqs_json="${PD_FLIP_MIGRATION_MAX_REQS}"
  fi
  cat > "${SUITE_DIR}/suite_manifest.json" <<JSON
{
  "suite_dir": "${SUITE_DIR}",
  "repo": "${REPO}",
  "router_url": "${ROUTER_URL}",
  "model_id": "${MODEL_ID}",
  "max_running_requests": ${MAX_RUNNING_REQUESTS},
  "migration_max_reqs": ${migration_max_reqs_json},
  "requests": ${REQUESTS},
  "interval_seconds": ${INTERVAL_SECONDS},
  "short_chars": ${SHORT_CHARS},
  "long_chars": ${LONG_CHARS},
  "short_count": ${SHORT_COUNT},
  "long_count": ${LONG_COUNT},
  "force_delay_seconds": ${FORCE_DELAY_SECONDS},
  "post_unpin_delay_seconds": ${POST_UNPIN_DELAY_SECONDS},
  "observation_quiesce_seconds": ${PD_FLIP_OBSERVE_QUIESCE_SECONDS},
  "post_migration_idle_timeout_seconds": ${PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS},
  "force_source_name": "${FORCE_SOURCE_NAME}",
  "migration_target_name": "${MIGRATION_TARGET_NAME}",
  "initial_roles": {
    "node0": "${INITIAL_ROLE_NODE0}",
    "node1": "${INITIAL_ROLE_NODE1}",
    "node2": "${INITIAL_ROLE_NODE2}",
    "node3": "${INITIAL_ROLE_NODE3}"
  },
  "pin_drain_for_waiting": ${PIN_DRAIN_FOR_WAITING},
  "pin_drain_worker_ids": "${PIN_DRAIN_WORKER_IDS}",
  "unpin_before_flip_worker_ids": "${UNPIN_BEFORE_FLIP_WORKER_IDS}",
  "nodes": {
    "node0": "${NODE0}",
    "node1": "${NODE1}",
    "node2": "${NODE2}",
    "node3": "${NODE3}"
  }
}
JSON
}

RUN_DIR="${SUITE_DIR}/${RUN_NAME}"
TRACE_DIR="${SUITE_DIR}/${TRACE_DIR_NAME}"

write_manifest
if [[ "${SKIP_CLUSTER_START}" == "1" ]]; then
  log "skipping cluster start; reusing existing workers/router"
  wait_router_ready
else
  distribute_wait_env
  start_workers "waiting_queue_full_link"
  start_router "waiting_queue_full_link"
fi
save_roles "${SUITE_DIR}/roles_initial.json"

mkdir -p "${TRACE_DIR}"
python3 "${TRACE}" generate \
  --output-dir "${TRACE_DIR}" \
  --model "${MODEL_ID}" \
  --num-requests "${REQUESTS}" \
  --interval-seconds "${INTERVAL_SECONDS}" \
  --short-chars "${SHORT_CHARS}" \
  --long-chars "${LONG_CHARS}" \
  --short-count "${SHORT_COUNT}" \
  --long-count "${LONG_COUNT}" \
  --seed 20260708 \
  --stream true \
  > "${TRACE_DIR}/trace_generate.log" 2>&1

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}/${MODE}"
cp "${TRACE_DIR}/trace_requests.jsonl" "${RUN_DIR}/trace_requests.jsonl"
cp "${TRACE_DIR}/trace_requests.csv" "${RUN_DIR}/trace_requests.csv"
save_roles "${RUN_DIR}/roles_before.json"

if [[ "${PIN_DRAIN_FOR_WAITING}" == "1" ]]; then
  log "temporarily draining workers so ${FORCE_SOURCE_NAME} accumulates scheduler waiting_queue: ${PIN_DRAIN_WORKER_IDS}"
  for worker_id in ${PIN_DRAIN_WORKER_IDS}; do
    set_router_drain "${worker_id}" true
  done
  save_roles "${RUN_DIR}/roles_after_pin_drain.json"
fi

log "starting sampler"
python3 "${MEASURE}" sample \
  --router-url "${ROUTER_URL}" \
  "${MEASURE_NODE_ARGS[@]}" \
  --output-events "${RUN_DIR}/migration_events.jsonl" \
  --interval-seconds 0.1 \
  --duration-seconds "${SAMPLER_DURATION_SECONDS}" \
  > "${RUN_DIR}/migration_sampler.log" 2>&1 &
SAMPLER_PID=$!
echo "${SAMPLER_PID}" > "${RUN_DIR}/migration_sampler.pid"

log "starting replay"
python3 "${TRACE}" replay \
  --trace-jsonl "${RUN_DIR}/trace_requests.jsonl" \
  --router-url "${ROUTER_URL}" \
  --mode "${MODE}" \
  --output-dir "${RUN_DIR}" \
  --ledger-path "${RUN_DIR}/${MODE}/trace_slo_ledger.jsonl" \
  --timeout-seconds "${REPLAY_TIMEOUT_SECONDS}" \
  --max-workers 256 \
  > "${RUN_DIR}/replay.log" 2>&1 &
REPLAY_PID=$!
echo "${REPLAY_PID}" > "${RUN_DIR}/replay.pid"

sleep "${FORCE_DELAY_SECONDS}"
if [[ "${PIN_DRAIN_FOR_WAITING}" == "1" ]]; then
  if [[ -n "${UNPIN_BEFORE_FLIP_WORKER_IDS}" ]]; then
    log "undraining selected workers before migration target selection: ${UNPIN_BEFORE_FLIP_WORKER_IDS}"
    for worker_id in ${UNPIN_BEFORE_FLIP_WORKER_IDS}; do
      set_router_drain "${worker_id}" false
    done
  else
    log "no pinned workers selected for undrain before migration"
  fi
  save_roles "${RUN_DIR}/roles_before_flip.json"
fi
if [[ "${POST_UNPIN_DELAY_SECONDS}" != "0" ]]; then
  log "waiting ${POST_UNPIN_DELAY_SECONDS}s before forcing flip"
  sleep "${POST_UNPIN_DELAY_SECONDS}"
fi
log "forcing two-phase D->P for ${FORCE_SOURCE_NAME}"
CTRL_TARGET_ARGS=()
if [[ -n "${MIGRATION_TARGET_NAME}" ]]; then
  CTRL_TARGET_ARGS=(--migration-target-name "${MIGRATION_TARGET_NAME}")
fi
controller execute-two-phase --direction d_to_p --source-name "${FORCE_SOURCE_NAME}" "${CTRL_TARGET_ARGS[@]}" > "${RUN_DIR}/controller.log" 2>&1 || true

set +e
wait "${REPLAY_PID}"
REPLAY_RC=$?
set -e
echo "${REPLAY_RC}" > "${RUN_DIR}/replay.exit"

kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
save_roles "${RUN_DIR}/roles_after.json"
summarize_run "${RUN_DIR}" "${MODE}"

python3 "${DIAGRAM}" "${SUITE_DIR}" "${RUN_NAME}" > "${SUITE_DIR}/diagram.log" 2>&1 || true

python3 - <<'PY' "${SUITE_DIR}" "${RUN_NAME}" "${MODE}"
import csv
import json
import sys
from pathlib import Path

suite = Path(sys.argv[1])
run_name = sys.argv[2]
mode = sys.argv[3]
summary_path = suite / run_name / mode / "summary.json"
rows = []
if summary_path.exists():
    with summary_path.open("r", encoding="utf-8") as f:
        item = json.load(f)
    item["experiment"] = run_name
    rows.append(item)
fields = [
    "experiment",
    "mode",
    "request_count",
    "completed_count",
    "error_count",
    "ttft_attainment",
    "tpot_avg_attainment",
    "tpot_p95_attainment",
    "tpot_interval_attainment",
    "all_attainment",
    "run_elapsed_s",
]
with (suite / "suite_slo_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
with (suite / "suite_slo_summary.json").open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, sort_keys=True)
PY

tar -C "$(dirname "${SUITE_DIR}")" -czf "${SUITE_DIR}.tar.gz" "$(basename "${SUITE_DIR}")"
log "waiting queue full-link experiment complete package=${SUITE_DIR}.tar.gz replay_rc=${REPLAY_RC}"
