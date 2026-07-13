#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="${1:-/root/sglang/experiments/pd_flip_suite_$(date +%Y%m%d_%H%M%S)}"
REPO="${REPO:-/root/sglang}"
DOCKER_DIR="${REPO}/scripts/playground/disaggregation/pd_flip_docker"
ENV_FILE="${DOCKER_DIR}/env.local"
START_AT="${START_AT:-1}"

mkdir -p "${SUITE_DIR}"
exec > >(tee -a "${SUITE_DIR}/suite_runner.log") 2>&1

source "${ENV_FILE}"

TRACE="${REPO}/scripts/playground/disaggregation/pd_flip_trace_replay.py"
MEASURE="${REPO}/scripts/playground/disaggregation/pd_flip_migration_measure.py"
CTRL="${REPO}/scripts/playground/disaggregation/pd_flip_controller.py"
ROUTER_URL="http://127.0.0.1:${ROUTER_PORT}"

NODE_NAMES=(node0 node1 node2 node3)
NODE_HOSTS=(192.168.0.42 192.168.0.40 192.168.0.39 192.168.0.41)
NODE_URLS=("${NODE0}" "${NODE1}" "${NODE2}" "${NODE3}")
NODE_SESSIONS=(pd-node0 pd-node1 pd-node2 pd-node3)
INITIAL_ROLES=(prefill prefill decode decode)

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

controller() {
  docker run --rm --network host \
    -v /root/sglang:/root/sglang \
    "${IMAGE}" \
    python3 "${CTRL}" \
    --router-url "${ROUTER_URL}" \
    --timeout-seconds 20 \
    "${NODE_ARGS[@]}" \
    "$@"
}

controller_timeout() {
  local seconds="$1"
  shift
  timeout "${seconds}" docker run --rm --network host \
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

roles_are_initial() {
  curl -fsS "${ROUTER_URL}/pd_flip/router/workers" 2>/dev/null | python3 -c '
import json, sys
body = json.load(sys.stdin)
roles = {w.get("url"): w.get("role") for w in body.get("workers", [])}
expected = {
    "'"${NODE0}"'": "prefill",
    "'"${NODE1}"'": "prefill",
    "'"${NODE2}"'": "decode",
    "'"${NODE3}"'": "decode",
}
sys.exit(0 if roles == expected else 1)
' 2>/dev/null
}

node2_prefill_idle() {
  curl -fsS "${ROUTER_URL}/pd_flip/router/workers" 2>/dev/null | python3 -c '
import json, sys
body = json.load(sys.stdin)
for worker in body.get("workers", []):
    if worker.get("url") == "'"${NODE2}"'":
        role = worker.get("role")
        active = int(worker.get("active_load") or 0)
        draining = bool(worker.get("draining"))
        if role == "decode":
            sys.exit(2)
        if role == "prefill" and active == 0 and not draining:
            sys.exit(0)
        sys.exit(1)
sys.exit(1)
' 2>/dev/null
}

wait_workers_ready() {
  local label="$1"
  local deadline=$((SECONDS + 2400))
  local ok=0
  while (( SECONDS < deadline )); do
    ok=1
    for url in "${NODE_URLS[@]}"; do
      if ! curl -fsS "${url}/server_info" >/dev/null 2>&1; then
        ok=0
        break
      fi
    done
    if (( ok == 1 )); then
      log "workers ready (${label})"
      return 0
    fi
    sleep 10
  done
  log "workers not ready before timeout (${label})"
  return 1
}

wait_router_ready() {
  local label="$1"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -fsS "${ROUTER_URL}/pd_flip/router/workers" >/dev/null 2>&1; then
      log "router ready (${label})"
      return 0
    fi
    sleep 5
  done
  log "router not ready before timeout (${label})"
  return 1
}

start_workers() {
  local label="$1"
  local enable_state_machine="$2"
  log "starting workers label=${label} enable_state_machine=${enable_state_machine}"
  for i in "${!NODE_HOSTS[@]}"; do
    local host="${NODE_HOSTS[$i]}"
    local role="${INITIAL_ROLES[$i]}"
    local session="${NODE_SESSIONS[$i]}"
    ssh_node "${host}" "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t '${session}' 2>/dev/null || true; fi; if [ -f '${DOCKER_DIR}/${session}.pid' ]; then kill \$(cat '${DOCKER_DIR}/${session}.pid') 2>/dev/null || true; rm -f '${DOCKER_DIR}/${session}.pid'; fi; fuser -k ${PORT}/tcp 2>/dev/null || true; cd '${DOCKER_DIR}'; rm -f 'worker.${label}.log'; if command -v tmux >/dev/null 2>&1; then tmux new -d -s '${session}' 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ENABLE_PD_FLIP_STATE_MACHINE=${enable_state_machine} ENABLE_PD_RUNTIME_ROLE_SWITCH=${enable_state_machine} ./run_worker.sh ${role} 0.0.0.0 2>&1 | tee worker.${label}.log'; else nohup bash -lc 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ENABLE_PD_FLIP_STATE_MACHINE=${enable_state_machine} ENABLE_PD_RUNTIME_ROLE_SWITCH=${enable_state_machine} ./run_worker.sh ${role} 0.0.0.0' > 'worker.${label}.log' 2>&1 < /dev/null & echo \$! > '${session}.pid'; fi"
  done
  wait_workers_ready "${label}"
}

start_router() {
  local label="$1"
  log "starting router label=${label}"
  ssh_node 192.168.0.42 "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t pd-router 2>/dev/null || true; fi; if [ -f '${DOCKER_DIR}/pd-router.pid' ]; then kill \$(cat '${DOCKER_DIR}/pd-router.pid') 2>/dev/null || true; rm -f '${DOCKER_DIR}/pd-router.pid'; fi; fuser -k ${ROUTER_PORT}/tcp 2>/dev/null || true; cd '${DOCKER_DIR}'; rm -f 'router.${label}.log'; if command -v tmux >/dev/null 2>&1; then tmux new -d -s pd-router 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ./run_router.sh 2>&1 | tee router.${label}.log'; else nohup bash -lc 'cd ${DOCKER_DIR}; ENV_FILE=${ENV_FILE} ./run_router.sh' > 'router.${label}.log' 2>&1 < /dev/null & echo \$! > pd-router.pid; fi"
  wait_router_ready "${label}"
}

reset_state_machine_cluster() {
  local label="$1"
  log "resetting state-machine cluster label=${label}"
  start_workers "${label}" 1
  start_router "${label}"
}

generate_trace() {
  local out_dir="$1"
  local num_requests="$2"
  local interval_seconds="$3"
  local seed="$4"
  local stream="$5"
  mkdir -p "${out_dir}"
  python3 "${TRACE}" generate \
    --output-dir "${out_dir}" \
    --model "${MODEL_ID}" \
    --num-requests "${num_requests}" \
    --interval-seconds "${interval_seconds}" \
    --seed "${seed}" \
    --stream "${stream}" \
    > "${out_dir}/trace_generate.log" 2>&1
}

summarize_run() {
  local run_dir="$1"
  local mode="$2"
  local events="${run_dir}/migration_events.jsonl"
  mkdir -p "${run_dir}/migration_link"
  local args=(summarize --events-jsonl "${events}" --output-dir "${run_dir}/migration_link")
  if [[ -s "${run_dir}/controller.log" ]]; then
    args+=(--controller-log "${run_dir}/controller.log")
  fi
  if [[ -s "${run_dir}/${mode}/request_metrics.jsonl" ]]; then
    args+=(--request-metrics-jsonl "${run_dir}/${mode}/request_metrics.jsonl")
  fi
  if [[ -s "${run_dir}/${mode}/errors.jsonl" ]]; then
    args+=(--errors-jsonl "${run_dir}/${mode}/errors.jsonl")
  fi
  python3 "${MEASURE}" "${args[@]}" > "${run_dir}/migration_link_summary.log" 2>&1 || true
  python3 "${TRACE}" summarize --output-dir "${run_dir}" --modes "${mode}" > "${run_dir}/trace_summary.log" 2>&1 || true
}

restore_node2_decode() {
  local run_dir="$1"
  log "restoring node2 to decode"
  if roles_are_initial; then
    log "node2 already decode"
    save_roles "${run_dir}/roles_after_restore.json"
    return 0
  fi
  set +e
  node2_prefill_idle
  local node2_state=$?
  set -e
  if [[ "${node2_state}" == "2" ]]; then
    log "node2 reports decode but cluster roles are not initial; resetting"
    reset_state_machine_cluster "reset_after_$(basename "${run_dir}")"
  elif [[ "${node2_state}" == "0" ]]; then
    controller_timeout 150s execute --direction p_to_d --source-name node2 > "${run_dir}/restore_p_to_d.log" 2>&1 || true
    if ! roles_are_initial; then
      log "p_to_d restore did not return initial roles; resetting"
      reset_state_machine_cluster "reset_after_$(basename "${run_dir}")"
    fi
  else
    log "node2 is not idle for p_to_d restore; resetting"
    reset_state_machine_cluster "reset_after_$(basename "${run_dir}")"
  fi
  sleep 2
  save_roles "${run_dir}/roles_after_restore.json"
}

run_replay_run() {
  local name="$1"
  local mode="$2"
  local trace_dir="$3"
  local timeout_seconds="$4"
  local sampler_duration="$5"
  local force_kind="$6"
  local force_delay="$7"
  local run_dir="${SUITE_DIR}/${name}"

  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"
  cp "${trace_dir}/trace_requests.jsonl" "${run_dir}/trace_requests.jsonl"
  cp "${trace_dir}/trace_requests.csv" "${run_dir}/trace_requests.csv"
  save_roles "${run_dir}/roles_before.json"
  log "experiment start name=${name} mode=${mode} force=${force_kind} delay=${force_delay}"

  python3 "${MEASURE}" sample \
    --router-url "${ROUTER_URL}" \
    "${MEASURE_NODE_ARGS[@]}" \
    --output-events "${run_dir}/migration_events.jsonl" \
    --interval-seconds 0.2 \
    --duration-seconds "${sampler_duration}" \
    > "${run_dir}/migration_sampler.log" 2>&1 &
  local sampler_pid=$!
  echo "${sampler_pid}" > "${run_dir}/migration_sampler.pid"

  mkdir -p "${run_dir}/${mode}"
  python3 "${TRACE}" replay \
    --trace-jsonl "${run_dir}/trace_requests.jsonl" \
    --router-url "${ROUTER_URL}" \
    --mode "${mode}" \
    --output-dir "${run_dir}" \
    --ledger-path "${run_dir}/${mode}/trace_slo_ledger.jsonl" \
    --timeout-seconds "${timeout_seconds}" \
    --max-workers 256 \
    > "${run_dir}/replay.log" 2>&1 &
  local replay_pid=$!
  echo "${replay_pid}" > "${run_dir}/replay.pid"

  if [[ "${force_kind}" == "two_phase" ]]; then
    sleep "${force_delay}"
    log "forcing two-phase D->P for ${name}"
    controller execute-two-phase --direction d_to_p --source-name node2 > "${run_dir}/controller.log" 2>&1 || true
  elif [[ "${force_kind}" == "abort" ]]; then
    sleep "${force_delay}"
    log "forcing monitor abort branch for ${name}"
    controller monitor \
      --ttft-slo 5 \
      --tpot-slo 0.02 \
      --window-seconds 30 \
      --enter-threshold 1.1 \
      --exit-threshold 0.0 \
      --commit-threshold 1.1 \
      --iterations 1 \
      --poll-interval 1 \
      --trace-slo-ledger "${run_dir}/${mode}/trace_slo_ledger.jsonl" \
      > "${run_dir}/controller.log" 2>&1 || true
  else
    :
  fi

  set +e
  wait "${replay_pid}"
  local replay_rc=$?
  set -e
  echo "${replay_rc}" > "${run_dir}/replay.exit"

  kill "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
  save_roles "${run_dir}/roles_after.json"
  summarize_run "${run_dir}" "${mode}"

  if [[ "${force_kind}" == "two_phase" ]]; then
    restore_node2_decode "${run_dir}"
  fi
log "experiment done name=${name} replay_rc=${replay_rc}"
}

write_suite_manifest() {
  cat > "${SUITE_DIR}/suite_manifest.json" <<JSON
{
  "suite_dir": "${SUITE_DIR}",
  "repo": "${REPO}",
  "router_url": "${ROUTER_URL}",
  "model_id": "${MODEL_ID}",
  "nodes": {
    "node0": "${NODE0}",
    "node1": "${NODE1}",
    "node2": "${NODE2}",
    "node3": "${NODE3}"
  }
}
JSON
}

write_suite_manifest
save_roles "${SUITE_DIR}/roles_initial.json"
wait_router_ready "initial"

TRACE_ROOT="${SUITE_DIR}/traces"
generate_trace "${TRACE_ROOT}/stream_true_40_seed301" 40 1 301 true
generate_trace "${TRACE_ROOT}/stream_false_40_seed302" 40 1 302 false
generate_trace "${TRACE_ROOT}/stream_false_sweep_80_seed401" 80 0.5 401 false
generate_trace "${TRACE_ROOT}/stream_true_abort_50_seed501" 50 0.5 501 true

if (( START_AT <= 1 )); then
  run_replay_run "01_stream_true_no_migration" "stream_true_no_migration" "${TRACE_ROOT}/stream_true_40_seed301" 180 260 none 0
else
  log "skip 01_stream_true_no_migration START_AT=${START_AT}"
fi
if (( START_AT <= 2 )); then
  run_replay_run "02_stream_true_two_phase" "stream_true_two_phase" "${TRACE_ROOT}/stream_true_40_seed301" 180 260 two_phase 12
else
  log "skip 02_stream_true_two_phase START_AT=${START_AT}"
fi
if (( START_AT <= 3 )); then
  run_replay_run "03_stream_false_two_phase" "stream_false_two_phase" "${TRACE_ROOT}/stream_false_40_seed302" 180 260 two_phase 12
else
  log "skip 03_stream_false_two_phase START_AT=${START_AT}"
fi

if (( START_AT <= 4 )); then
  run_replay_run "04_sweep_force_delay_6s" "sweep_delay_6s" "${TRACE_ROOT}/stream_false_sweep_80_seed401" 180 300 two_phase 6
else
  log "skip 04_sweep_force_delay_6s START_AT=${START_AT}"
fi
if (( START_AT <= 5 )); then
  run_replay_run "05_sweep_force_delay_12s" "sweep_delay_12s" "${TRACE_ROOT}/stream_false_sweep_80_seed401" 180 300 two_phase 12
else
  log "skip 05_sweep_force_delay_12s START_AT=${START_AT}"
fi
if (( START_AT <= 6 )); then
  run_replay_run "06_sweep_force_delay_24s" "sweep_delay_24s" "${TRACE_ROOT}/stream_false_sweep_80_seed401" 180 300 two_phase 24
else
  log "skip 06_sweep_force_delay_24s START_AT=${START_AT}"
fi
if (( START_AT <= 7 )); then
  run_replay_run "07_sweep_force_delay_36s" "sweep_delay_36s" "${TRACE_ROOT}/stream_false_sweep_80_seed401" 180 300 two_phase 36
else
  log "skip 07_sweep_force_delay_36s START_AT=${START_AT}"
fi

if (( START_AT <= 8 )); then
  run_replay_run "08_abort_branch" "abort_branch" "${TRACE_ROOT}/stream_true_abort_50_seed501" 180 260 abort 8
else
  log "skip 08_abort_branch START_AT=${START_AT}"
fi

generate_trace "${TRACE_ROOT}/stream_true_200_seed20260707" 200 1 20260707 true

if (( START_AT <= 9 )); then
  start_workers "baseline_no_state_machine" 0
  start_router "baseline_no_state_machine"
  run_replay_run "09_ab_200_baseline_no_state_machine" "baseline" "${TRACE_ROOT}/stream_true_200_seed20260707" 240 520 none 0
else
  log "skip 09_ab_200_baseline_no_state_machine START_AT=${START_AT}"
fi

if (( START_AT <= 10 )); then
  start_workers "state_machine_enabled" 1
  start_router "state_machine_enabled"
  run_replay_run "10_ab_200_state_machine_two_phase" "state_machine" "${TRACE_ROOT}/stream_true_200_seed20260707" 240 560 two_phase 60
else
  log "skip 10_ab_200_state_machine_two_phase START_AT=${START_AT}"
fi

python3 "${TRACE}" summarize --output-dir "${SUITE_DIR}/09_ab_200_baseline_no_state_machine" --modes baseline > "${SUITE_DIR}/ab_baseline_summary.log" 2>&1 || true
python3 "${TRACE}" summarize --output-dir "${SUITE_DIR}/10_ab_200_state_machine_two_phase" --modes state_machine > "${SUITE_DIR}/ab_state_summary.log" 2>&1 || true

python3 - <<'PY' "${SUITE_DIR}"
import csv
import json
import sys
from pathlib import Path

suite = Path(sys.argv[1])
rows = []
for summary_path in sorted(suite.glob("*/**/summary.json")):
    if summary_path.parent.name not in {"baseline", "state_machine"} and not summary_path.parent.name.startswith(("stream_", "sweep_", "abort_")):
        continue
    with summary_path.open("r", encoding="utf-8") as f:
        item = json.load(f)
    item["experiment"] = summary_path.parent.parent.name
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
    for row in rows:
        writer.writerow(row)
with (suite / "suite_slo_summary.json").open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, sort_keys=True)
PY

tar -C "$(dirname "${SUITE_DIR}")" -czf "${SUITE_DIR}.tar.gz" "$(basename "${SUITE_DIR}")"
log "suite complete package=${SUITE_DIR}.tar.gz"
