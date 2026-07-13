#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

# Values explicitly exported on the command line are per-run controls and take
# precedence over the shared ENV_FILE. Preserve both set and set-to-empty values.
override_names=(
  ADMIN_API_KEY DIRECTION SOURCE_NAME MIGRATION_TARGET_NAME
  PD_FLIP_CONTROLLER_USE_DOCKER PD_FLIP_TRACE_SLO_LEDGER
  PD_FLIP_FIRST_MIGRATION_RATIO PD_FLIP_OBSERVATION_SECONDS PD_FLIP_SLO_THRESHOLD
  PD_FLIP_MIN_PREFILL_SLO_SAMPLES PD_FLIP_MIN_DECODE_SLO_SAMPLES
  PD_FLIP_ARTIFACT_DIR PD_FLIP_SESSION_JOURNAL_PATH PD_FLIP_SESSION_ID_PREFIX
  PD_FLIP_MONITOR_ITERATIONS PD_FLIP_MONITOR_POLL_INTERVAL
  TTFT_SLO_SECONDS TPOT_SLO_SECONDS PD_FLIP_WINDOW_SECONDS
  PD_FLIP_ENTER_THRESHOLD PD_FLIP_EXIT_THRESHOLD PD_FLIP_COMMIT_THRESHOLD
)
declare -A command_overrides=()
for name in "${override_names[@]}"; do
  if [[ -v "${name}" ]]; then
    command_overrides["${name}"]="${!name}"
  fi
done
source "${ENV_FILE:-${SCRIPT_DIR}/env.example}"
for name in "${override_names[@]}"; do
  if [[ -v "command_overrides[${name}]" ]]; then
    printf -v "${name}" '%s' "${command_overrides[${name}]}"
  fi
done

case "${ADMIN_API_KEY:-}" in
  ""|replace-with-*|changeme|CHANGE_ME)
    echo "ADMIN_API_KEY must be set to a non-placeholder secret" >&2
    exit 2
    ;;
esac

ACTION="${1:-dry-run}"
DIRECTION="${DIRECTION:-d_to_p}"
SOURCE_NAME="${SOURCE_NAME:-}"
MIGRATION_TARGET_NAME="${MIGRATION_TARGET_NAME:-}"

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
  --session-journal-path "${PD_FLIP_SESSION_JOURNAL_PATH:-${PD_FLIP_ARTIFACT_DIR:-/sgl-workspace/sglang/pd-flip-artifacts/four-node-progressive}/pd_flip_session.json}"
)

if [[ -n "${PD_FLIP_SESSION_ID_PREFIX:-}" ]]; then
  base_args+=(--session-id-prefix "${PD_FLIP_SESSION_ID_PREFIX}")
fi

base_args+=(--api-key "${ADMIN_API_KEY}")

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
  monitor-progressive)
    if [[ -z "${PD_FLIP_TRACE_SLO_LEDGER:-}" ]]; then
      echo "PD_FLIP_TRACE_SLO_LEDGER is required" >&2
      exit 2
    fi
    progressive_args=(
      monitor-progressive
      --trace-slo-ledger "${PD_FLIP_TRACE_SLO_LEDGER}"
      --iterations "${PD_FLIP_MONITOR_ITERATIONS}"
      --poll-interval "${PD_FLIP_MONITOR_POLL_INTERVAL}"
    )
    if [[ -n "${SOURCE_NAME}" ]]; then
      progressive_args+=(--source-name "${SOURCE_NAME}")
    fi
    if [[ -n "${MIGRATION_TARGET_NAME}" ]]; then
      progressive_args+=(--migration-target-name "${MIGRATION_TARGET_NAME}")
    fi
    run_controller "${base_args[@]}" "${progressive_args[@]}"
    ;;
  *)
    echo "usage: run_controller.sh [metrics|dry-run|execute|monitor|monitor-progressive]" >&2
    echo "       DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh dry-run" >&2
    echo "       DIRECTION=d_to_p SOURCE_NAME=node2 ./run_controller.sh execute" >&2
    echo "       ./run_controller.sh monitor" >&2
    echo "       PD_FLIP_TRACE_SLO_LEDGER=/path/ledger.jsonl SOURCE_NAME=node2 MIGRATION_TARGET_NAME=node3 ./run_controller.sh monitor-progressive" >&2
    exit 2
    ;;
esac
