#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/sglang}"
DOCKER_DIR="${REPO}/scripts/playground/disaggregation/pd_flip_docker"
BASE_ENV="${DOCKER_DIR}/env.local"
SUITE_DIR="${1:-${REPO}/experiments/pd_flip_1p3d_to_2p2d_$(date +%Y%m%d_%H%M%S)}"

source "${BASE_ENV}"
mkdir -p "${SUITE_DIR}"

cat > "${SUITE_DIR}/reschedule_stitching_manifest.json" <<JSON
{
  "requested_transition": "1P3D -> 2P2D",
  "initial_roles": {
    "node0": "prefill",
    "node1": "decode",
    "node2": "decode",
    "node3": "decode"
  },
  "forced_source": "node2",
  "expected_final_roles": {
    "node0": "prefill",
    "node1": "decode",
    "node2": "prefill",
    "node3": "decode"
  },
  "request_reschedule_strategy": "drain node1/node3 before force_delay so node2 accumulates waiting_queue; before flip undrain node1 for user traffic and keep node3 draining as the explicit KV migration target",
  "current_kv_path": "decode-source committed KV is migrated to the selected target decode node",
  "prefill_decode_kv_stitching": {
    "requested": true,
    "exercised_in_this_runner": false,
    "reason": "router only handles request routing/drain/role refresh today; target-side concatenation of prefill KV and decode KV needs worker/controller support beyond the existing decode-source migration path"
  }
}
JSON

export RUN_NAME="${RUN_NAME:-01_1p3d_to_2p2d_two_phase}"
export MODE="${MODE:-one_p_three_d_state_machine}"
export TRACE_DIR_NAME="${TRACE_DIR_NAME:-trace_1p3d_to_2p2d}"
export INITIAL_ROLE_NODE0="${INITIAL_ROLE_NODE0:-prefill}"
export INITIAL_ROLE_NODE1="${INITIAL_ROLE_NODE1:-decode}"
export INITIAL_ROLE_NODE2="${INITIAL_ROLE_NODE2:-decode}"
export INITIAL_ROLE_NODE3="${INITIAL_ROLE_NODE3:-decode}"
export FORCE_SOURCE_NAME="${FORCE_SOURCE_NAME:-node2}"
export MIGRATION_TARGET_NAME="${MIGRATION_TARGET_NAME:-node3}"
export PIN_DRAIN_FOR_WAITING="${PIN_DRAIN_FOR_WAITING:-1}"
export PIN_DRAIN_WORKER_IDS="${PIN_DRAIN_WORKER_IDS:-${NODE1} ${NODE3}}"
export UNPIN_BEFORE_FLIP_WORKER_IDS="${UNPIN_BEFORE_FLIP_WORKER_IDS:-${NODE1}}"

exec bash "${REPO}/experiments/pd_flip_waiting_queue_full_link_runner.sh" "${SUITE_DIR}"
