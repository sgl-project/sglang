#!/usr/bin/env bash
# c=1 ISL-ablation ladder (GLM-5.2-NVFP4, 4xGB300, TP4).
#
# Replays the canonical c=1 / TP4 step across an ISL ladder — 80K, 128K, 256K,
# 512K, 1M mean input tokens — holding everything else fixed. The ONLY knob
# that changes per rung is the dataset's first-turn budget and the server's
# --context-length. Unlike the main sweep, the server must restart per rung
# (context length is a boot-time setting), so this driver launches the given
# server script itself.
#
# Usage:
#   ./run_isl_client.sh v0515                       # v0.5.15 series
#   DAY0_SGLANG=../sglang-day0 ./run_isl_client.sh day0   # day-0 series
#
# Results: results/<series>/<rung>/isl_<rung>/parallel_1_number_4/benchmark_summary.json
# Runtime: ~2-3 h per series (dataset builds for the long rungs dominate the
# first run; they are cached under datasets/ afterwards).
set -uo pipefail

SERIES=${1:?usage: run_isl_client.sh <v0515|day0>}
case "$SERIES" in
    v0515) SERVER=server_v0515_ctx.sh ;;
    day0)  SERVER=server_day0_ctx.sh ;;
    *) echo "series must be v0515 or day0" >&2; exit 1 ;;
esac

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PORT:=8002}"
MODEL=nvidia/GLM-5.2-NVFP4

# Mean ISL over the 13-turn conversation = first_turn + 6*(753+220).
RUNGS=(80k 128k 256k 512k 1m)
declare -A FIRST=( [80k]=74160 [128k]=122162 [256k]=250162 [512k]=506162 [1m]=994162 )
declare -A CTX=(   [80k]=90000 [128k]=138000 [256k]=266000 [512k]=522000 [1m]=1010000 )

# evalscope present?
python3 -c "import evalscope.perf.plugin.datasets.swe_smith" 2>/dev/null \
    || { echo "evalscope missing — see ../evalscope-deps/README.md" >&2; exit 1; }

# ---- Phase 1: build all per-rung datasets up front (CPU/network only) ----
mkdir -p "$DIR/datasets"
for tag in "${RUNGS[@]}"; do
    DS="$DIR/datasets/openhand-isl-${tag}.json"
    [ -s "$DS" ] && { echo "dataset $tag exists, skip"; continue; }
    echo "=== building dataset $tag (first_turn=${FIRST[$tag]}) ==="
    python3 "$DIR/../build_openhands_padded_dataset.py" \
        --model "$MODEL" \
        --pad-source openscience \
        --first-turn-length "${FIRST[$tag]}" \
        --subsequent-turn-length 753 \
        --num-turns 13 \
        --number 4 \
        --output-path "$DS"
done
echo "all datasets ready"

wait_port_free() {
    for _ in $(seq 1 90); do
        python3 -c "import socket; s=socket.socket(); s.bind(('localhost', $PORT)); s.close()" 2>/dev/null && return 0
        sleep 2
    done
    return 1
}

# ---- Phase 2: one server boot + one c=1 evalscope step per rung ----
for tag in "${RUNGS[@]}"; do
    OUT="$DIR/results/$SERIES/$tag"
    SUMM="$OUT/isl_${tag}/parallel_1_number_4/benchmark_summary.json"
    [ -s "$SUMM" ] && { echo "rung $tag already done, skip"; continue; }
    mkdir -p "$OUT"
    echo "=== rung $tag: ctx=${CTX[$tag]} ==="
    CONTEXT_LEN="${CTX[$tag]}" PORT=$PORT setsid "$DIR/$SERVER" > "$OUT/server.log" 2>&1 &
    SPID=$!
    ready=0
    start=$SECONDS
    while (( SECONDS - start < 3600 )); do
        curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
        kill -0 "$SPID" 2>/dev/null || break
        sleep 5
    done
    if (( ! ready )); then
        echo "rung $tag: server failed to become ready — see $OUT/server.log" >&2
        kill -TERM -"$SPID" 2>/dev/null; sleep 8; kill -KILL -"$SPID" 2>/dev/null
        wait_port_free; continue
    fi
    evalscope perf \
        --model "$MODEL" \
        --url "http://localhost:$PORT/v1/chat/completions" \
        --api openai \
        --dataset swe_smith \
        --dataset-path "$DIR/datasets/openhand-isl-${tag}.json" \
        --max-tokens 220 \
        --multi-turn \
        --number 4 \
        --parallel 1 \
        --extra-args '{"ignore_eos": true}' \
        --name "isl_${tag}" \
        --outputs-dir "$OUT" \
        --no-timestamp > "$OUT/client.log" 2>&1 \
        || echo "rung $tag: client failed — see $OUT/client.log" >&2
    kill -TERM -"$SPID" 2>/dev/null; sleep 8; kill -KILL -"$SPID" 2>/dev/null
    wait_port_free
    [ -s "$SUMM" ] && echo "rung $tag OK" || echo "rung $tag MISSING summary" >&2
done
echo "=== ladder complete: results/$SERIES/ ==="
