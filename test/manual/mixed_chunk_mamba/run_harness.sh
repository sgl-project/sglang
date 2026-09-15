#!/usr/bin/env bash
# One statistical harness run. Usage: run_harness.sh <label> <A|B|C> <eager|graph> [prompts] [repeats] [concurrency]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=server_lib.sh
source "$HERE/server_lib.sh"
LABEL=$1; ARM=$2; GRAPH=$3; N=${4:-200}; R=${5:-3}; C=${6:-32}
MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL_PATH:?set MODEL_PATH to the local model directory}}
TAG="harness-$LABEL-$ARM-$GRAPH"
start_server "$LABEL" "$ARM" "$GRAPH" "$TAG" 2048
python3 "$HERE/shared_prefix_harness.py" --url "http://127.0.0.1:$PORT" --model "$MODEL_NAME" --log "$WS/logs/$TAG.log" \
  --tag "$TAG" --out "$WS/results/$TAG.json" --prompts "$N" --repeats "$R" --concurrency "$C" 2>&1 | tee "$WS/results/$TAG.summary.txt"
stop_server
