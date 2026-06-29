#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
source "${ENV_FILE:-${SCRIPT_DIR}/env.example}"

ACTION="${1:-dry-run}"
DIRECTION="${DIRECTION:-d_to_p}"
SOURCE_NAME="${SOURCE_NAME:-}"

base_args=(
  --router-url "http://${ROUTER_HOST}:${ROUTER_PORT}"
  --node "name=node0,worker_url=${NODE0},router_worker_id=${NODE0},bootstrap_port=${BOOTSTRAP_PORT}"
  --node "name=node1,worker_url=${NODE1},router_worker_id=${NODE1},bootstrap_port=${BOOTSTRAP_PORT}"
  --node "name=node2,worker_url=${NODE2},router_worker_id=${NODE2},bootstrap_port=${BOOTSTRAP_PORT}"
  --node "name=node3,worker_url=${NODE3},router_worker_id=${NODE3},bootstrap_port=${BOOTSTRAP_PORT}"
)

case "${ACTION}" in
  metrics)
    exec python3 "${REPO_ROOT}/scripts/playground/disaggregation/pd_flip_controller.py" \
      "${base_args[@]}" metrics
    ;;
  dry-run)
    dry_run_args=(dry-run --direction "${DIRECTION}")
    if [[ -n "${SOURCE_NAME}" ]]; then
      dry_run_args+=(--source-name "${SOURCE_NAME}")
    fi
    exec python3 "${REPO_ROOT}/scripts/playground/disaggregation/pd_flip_controller.py" \
      "${base_args[@]}" "${dry_run_args[@]}"
    ;;
  execute)
    execute_args=(execute --direction "${DIRECTION}")
    if [[ -n "${SOURCE_NAME}" ]]; then
      execute_args+=(--source-name "${SOURCE_NAME}")
    fi
    exec python3 "${REPO_ROOT}/scripts/playground/disaggregation/pd_flip_controller.py" \
      "${base_args[@]}" "${execute_args[@]}"
    ;;
  monitor)
    exec "${SCRIPT_DIR}/run_monitor.sh"
    ;;
  *)
    echo "usage: run_controller.sh [metrics|dry-run|execute|monitor]" >&2
    echo "       DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run" >&2
    echo "       DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute" >&2
    echo "       ./run_controller.sh monitor" >&2
    exit 2
    ;;
esac
