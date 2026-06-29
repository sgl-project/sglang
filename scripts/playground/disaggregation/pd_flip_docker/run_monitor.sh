#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
source "${ENV_FILE:-${SCRIPT_DIR}/env.example}"

exec python3 "${REPO_ROOT}/scripts/playground/disaggregation/pd_flip_controller.py" \
  --router-url "http://${ROUTER_HOST}:${ROUTER_PORT}" \
  --node "name=node0,worker_url=${NODE0},router_worker_id=${NODE0},bootstrap_port=${BOOTSTRAP_PORT}" \
  --node "name=node1,worker_url=${NODE1},router_worker_id=${NODE1},bootstrap_port=${BOOTSTRAP_PORT}" \
  --node "name=node2,worker_url=${NODE2},router_worker_id=${NODE2},bootstrap_port=${BOOTSTRAP_PORT}" \
  --node "name=node3,worker_url=${NODE3},router_worker_id=${NODE3},bootstrap_port=${BOOTSTRAP_PORT}" \
  monitor \
  --ttft-slo "${TTFT_SLO_SECONDS}" \
  --tpot-slo "${TPOT_SLO_SECONDS}" \
  --window-seconds "${PD_FLIP_WINDOW_SECONDS}" \
  --enter-threshold "${PD_FLIP_ENTER_THRESHOLD}" \
  --exit-threshold "${PD_FLIP_EXIT_THRESHOLD}" \
  --commit-threshold "${PD_FLIP_COMMIT_THRESHOLD}" \
  --iterations "${PD_FLIP_MONITOR_ITERATIONS}" \
  --poll-interval "${PD_FLIP_MONITOR_POLL_INTERVAL}"
