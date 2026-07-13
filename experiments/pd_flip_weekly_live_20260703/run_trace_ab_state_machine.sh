#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/sglang/scripts/playground/disaggregation/pd_flip_docker"
OUT="${SCRIPT_DIR}/artifacts/trace_ab_state_machine_20260703_1p2d"
ENV_FILE_PATH="${SCRIPT_DIR}/env.local"

rm -rf "${OUT}"
mkdir -p "${OUT}"

cd "${SCRIPT_DIR}"

docker run --rm \
  --network host \
  -v "${SCRIPT_DIR}:/work" \
  -w /work \
  sglang-pd-switch:tianciJ \
  python3 live_trace_ab.py \
    --router-url http://127.0.0.1:18001 \
    --model deepseek_v3.1_terminus \
    --out-dir /work/artifacts/trace_ab_state_machine_20260703_1p2d/client \
    --requests 100 \
    --concurrency 16 \
    --max-tokens 128 \
    --ttft-slo 8 \
    --tpot-slo 0.02 \
  >"${OUT}/client.raw" 2>&1 &
CLIENT_PID=$!

sleep 5

set +e
env \
  ENV_FILE="${ENV_FILE_PATH}" \
  PD_FLIP_NODE_NAMES="node0 node2 node3" \
  PD_FLIP_TTFT_SLO_SECONDS=0.001 \
  PD_FLIP_TPOT_SLO_SECONDS=0.02 \
  PD_FLIP_WINDOW_SECONDS_OVERRIDE=30 \
  PD_FLIP_MONITOR_ITERATIONS_OVERRIDE=60 \
  PD_FLIP_MONITOR_POLL_INTERVAL_OVERRIDE=1 \
  ./run_controller.sh monitor >"${OUT}/monitor.raw" 2>&1
echo $? >"${OUT}/monitor.exit"

wait "${CLIENT_PID}"
echo $? >"${OUT}/client.exit"

env \
  ENV_FILE="${ENV_FILE_PATH}" \
  PD_FLIP_NODE_NAMES="node0 node2 node3" \
  DIRECTION=p_to_d \
  SOURCE_NAME=node2 \
  ./run_controller.sh execute >"${OUT}/restore.raw" 2>&1
echo $? >"${OUT}/restore.exit"
set -e

cat "${OUT}/client.raw"
echo "==== monitor.exit ===="
cat "${OUT}/monitor.exit"
echo "==== restore.exit ===="
cat "${OUT}/restore.exit"
