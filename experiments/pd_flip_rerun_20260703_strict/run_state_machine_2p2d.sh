#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/sglang/scripts/playground/disaggregation/pd_flip_docker"
ROOT_OUT="${SCRIPT_DIR}/artifacts/pd_flip_rerun_20260703_strict"
OUT="${ROOT_OUT}/state_machine_2p2d_cold"
ENV_FILE_PATH="${SCRIPT_DIR}/env.local"

rm -rf "${OUT}"
mkdir -p "${OUT}"

cd "${SCRIPT_DIR}"

date -Is >"${OUT}/started_at.txt"
curl -m 5 -fsS http://127.0.0.1:18001/pd_flip/router/workers >"${OUT}/workers_before.json" 2>"${OUT}/workers_before.err" || true

docker run --rm \
  --network host \
  -v "${SCRIPT_DIR}:/work" \
  -w /work \
  sglang-pd-switch:tianciJ \
  python3 live_trace_ab.py \
    --router-url http://127.0.0.1:18001 \
    --model deepseek_v3.1_terminus \
    --out-dir /work/artifacts/pd_flip_rerun_20260703_strict/state_machine_2p2d_cold/client \
    --requests 100 \
    --concurrency 16 \
    --max-tokens 128 \
    --ttft-slo 8 \
    --tpot-slo 0.02 \
  >"${OUT}/client.raw" 2>&1 &
CLIENT_PID=$!

sleep 5
date -Is >"${OUT}/monitor_started_at.txt"

set +e
env \
  ENV_FILE="${ENV_FILE_PATH}" \
  PD_FLIP_NODE_NAMES="node0 node1 node2 node3" \
  PD_FLIP_TTFT_SLO_SECONDS=0.001 \
  PD_FLIP_TPOT_SLO_SECONDS=0.02 \
  PD_FLIP_WINDOW_SECONDS_OVERRIDE=30 \
  PD_FLIP_MONITOR_ITERATIONS_OVERRIDE=60 \
  PD_FLIP_MONITOR_POLL_INTERVAL_OVERRIDE=1 \
  ./run_controller.sh monitor >"${OUT}/monitor.raw" 2>&1
echo $? >"${OUT}/monitor.exit"

wait "${CLIENT_PID}"
echo $? >"${OUT}/client.exit"
set -e

date -Is >"${OUT}/client_finished_at.txt"
curl -m 5 -fsS http://127.0.0.1:18001/pd_flip/router/workers >"${OUT}/workers_after_monitor.json" 2>"${OUT}/workers_after_monitor.err" || true

RESTORE_SOURCE="$(
  python3 -c 'import json,sys
path=sys.argv[1]
ip_to_name={"192.168.0.39":"node2","192.168.0.41":"node3"}
try:
    data=json.load(open(path, encoding="utf-8"))
except Exception:
    sys.exit(0)
for w in data.get("workers", []):
    url=w.get("url") or w.get("worker_id") or ""
    role=w.get("role")
    for ip,name in ip_to_name.items():
        if ip in url and role == "prefill":
            print(name)
            sys.exit(0)
' "${OUT}/workers_after_monitor.json"
)"

if [[ -n "${RESTORE_SOURCE}" ]]; then
  echo "${RESTORE_SOURCE}" >"${OUT}/restore_source.txt"
  set +e
  env \
    ENV_FILE="${ENV_FILE_PATH}" \
    PD_FLIP_NODE_NAMES="node0 node1 node2 node3" \
    DIRECTION=p_to_d \
    SOURCE_NAME="${RESTORE_SOURCE}" \
    ./run_controller.sh execute >"${OUT}/restore.raw" 2>&1
  echo $? >"${OUT}/restore.exit"
  set -e
else
  echo "" >"${OUT}/restore_source.txt"
  echo "no decode node was observed as prefill after monitor; restore skipped" >"${OUT}/restore.raw"
  echo 0 >"${OUT}/restore.exit"
fi

curl -m 5 -fsS http://127.0.0.1:18001/pd_flip/router/workers >"${OUT}/workers_after_restore.json" 2>"${OUT}/workers_after_restore.err" || true

cat "${OUT}/client.raw"
echo "==== monitor.exit ===="
cat "${OUT}/monitor.exit"
echo "==== restore.source ===="
cat "${OUT}/restore_source.txt"
echo "==== restore.exit ===="
cat "${OUT}/restore.exit"
