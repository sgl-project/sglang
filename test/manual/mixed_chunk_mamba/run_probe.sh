#!/usr/bin/env bash
# Lifecycle probe, both phases. Usage: run_probe.sh <label> <A|B|C> <eager|graph> [decoders] [prefix_facts]
# The mixed phase runs on chunked_prefill_size 2048 with K decoders (prefill budget 2048-K);
# the reference phase runs alone on 2048-K (arms with mixed chunk on) or 2048 (arm B, whose
# budget is never reduced), so the checkpoint depths of the two phases coincide.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=server_lib.sh
source "$HERE/server_lib.sh"
LABEL=$1; ARM=$2; GRAPH=$3; DEC=${4:-8}; FACTS=${5:-380}
MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL_PATH:?set MODEL_PATH to the local model directory}}
TAG="probe-$LABEL-$ARM-$GRAPH"
if [ "$ARM" = B ]; then REFCHUNK=2048; else REFCHUNK=$((2048 - DEC)); fi
URL="http://127.0.0.1:$PORT"

start_server "$LABEL" "$ARM" "$GRAPH" "$TAG-mixed" 2048
python3 "$HERE/lifecycle_probe.py" --phase mixed --url "$URL" --model "$MODEL_NAME" \
  --log "$WS/logs/$TAG-mixed.log" --tag "$TAG-mixed" --out "$WS/results/$TAG-mixed.json" \
  --decoders "$DEC" --prefix-facts "$FACTS" 2>&1 | tee "$WS/results/$TAG-mixed.summary.txt"
stop_server

start_server "$LABEL" "$ARM" "$GRAPH" "$TAG-ref" "$REFCHUNK"
python3 "$HERE/lifecycle_probe.py" --phase ref --url "$URL" --model "$MODEL_NAME" \
  --log "$WS/logs/$TAG-ref.log" --tag "$TAG-ref" --out "$WS/results/$TAG-ref.json" \
  --decoders "$DEC" --prefix-facts "$FACTS" 2>&1 | tee "$WS/results/$TAG-ref.summary.txt"
stop_server

# Exit 2 (incomplete) is a valid outcome to report, so do not let set -e swallow the output.
python3 "$HERE/compare_probe.py" --mixed "$WS/results/$TAG-mixed.json" --ref "$WS/results/$TAG-ref.json" \
  --out "$WS/results/$TAG-compare.json" 2>&1 | tee "$WS/results/$TAG-compare.txt" || true
