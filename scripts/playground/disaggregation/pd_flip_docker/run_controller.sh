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
  --first-migration-ratio "${PD_FLIP_FIRST_MIGRATION_RATIO:-0.5}"
  --observation-seconds "${PD_FLIP_OBSERVATION_SECONDS:-10}"
  --slo-threshold "${PD_FLIP_SLO_THRESHOLD:-0.9}"
  --min-prefill-slo-samples "${PD_FLIP_MIN_PREFILL_SLO_SAMPLES:-20}"
  --min-decode-slo-samples "${PD_FLIP_MIN_DECODE_SLO_SAMPLES:-20}"
  --session-journal-path "${PD_FLIP_ARTIFACT_DIR:-/sgl-workspace/sglang/pd-flip-artifacts/four-node-progressive}/pd_flip_session.json"
)

if [[ -n "${ADMIN_API_KEY:-}" ]]; then
  base_args+=(--api-key "${ADMIN_API_KEY}")
fi

run_controller() {
  if [[ "${PD_FLIP_CONTROLLER_USE_DOCKER:-1}" == "1" ]]; then
    # shellcheck disable=SC2206
    extra_docker_args=(${EXTRA_DOCKER_ARGS:-})
    exec docker run --rm \
      --network host \
      --env-file "${ENV_FILE:-${SCRIPT_DIR}/env.example}" \
      "${extra_docker_args[@]}" \
      -v "${SGLANG_REPO}:/sgl-workspace/sglang" \
      "${IMAGE}" \
      python3 /sgl-workspace/sglang/scripts/playground/disaggregation/pd_flip_controller.py \
      "$@"
  fi

  exec python3 "${REPO_ROOT}/scripts/playground/disaggregation/pd_flip_controller.py" "$@"
}

case "${ACTION}" in
  metrics)
    run_controller "${base_args[@]}" metrics
    ;;
  dry-run)
    dry_run_args=(dry-run --direction "${DIRECTION}")
    if [[ -n "${SOURCE_NAME}" ]]; then
      dry_run_args+=(--source-name "${SOURCE_NAME}")
    fi
    run_controller "${base_args[@]}" "${dry_run_args[@]}"
    ;;
  execute)
    execute_args=(execute --direction "${DIRECTION}")
    if [[ -n "${SOURCE_NAME}" ]]; then
      execute_args+=(--source-name "${SOURCE_NAME}")
    fi
    run_controller "${base_args[@]}" "${execute_args[@]}"
    ;;
  monitor)
    # run_monitor.sh delegates here so monitor uses the same Docker Python.
    run_controller "${base_args[@]}" \
      monitor \
      --ttft-slo "${TTFT_SLO_SECONDS}" \
      --tpot-slo "${TPOT_SLO_SECONDS}" \
      --window-seconds "${PD_FLIP_WINDOW_SECONDS}" \
      --enter-threshold "${PD_FLIP_ENTER_THRESHOLD}" \
      --exit-threshold "${PD_FLIP_EXIT_THRESHOLD}" \
      --commit-threshold "${PD_FLIP_COMMIT_THRESHOLD}" \
      --iterations "${PD_FLIP_MONITOR_ITERATIONS}" \
      --poll-interval "${PD_FLIP_MONITOR_POLL_INTERVAL}"
    ;;
  *)
    echo "usage: run_controller.sh [metrics|dry-run|execute|monitor]" >&2
    echo "       DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run" >&2
    echo "       DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute" >&2
    echo "       ./run_controller.sh monitor" >&2
    exit 2
    ;;
esac
