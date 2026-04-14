#!/usr/bin/env bash
# Benchmark serving throughput for Qwen3-30B-A3B under different precision configs.
#
# Usage:
#   bash bench_serving.sh --precision <bf16|int4|heter> [options]
#
# Options:
#   --precision    bf16 | int4 | heter                    (required)
#   --strategy     tp | ep | partial                      (heter only, default: tp)
#   --rr           request rate, integer or "inf"         (default: inf)
#   --num-prompts  number of prompts                      (default: 500)
#   --heter-config path to heter precision config JSON    (heter only, has default)
#   --partial-config path to partial/int4-only expert map (required for --strategy partial)
#   --tp           tensor parallelism degree              (optional)
#   --ep           expert parallelism degree              (optional)
#
# Examples:
#   bash bench_serving.sh --precision bf16
#   bash bench_serving.sh --precision int4 --rr 16
#   bash bench_serving.sh --precision heter --strategy tp --rr 8
#   bash bench_serving.sh --precision heter --strategy partial \
#       --partial-config /path/to/int4_only_experts.json
#   bash bench_serving.sh --precision bf16 --tp 2 --rr 4 --num-prompts 1000

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
INT4_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"
DEFAULT_HETER_CONFIG="$SCRIPT_DIR/../partial_bf16/heter_config.json"

PORT=30000
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"

PRECISION=""
STRATEGY=""
REQUEST_RATE="inf"
NUM_PROMPTS=500
HETER_CONFIG=""
PARTIAL_CONFIG=""
TP=""
EP=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --precision)      PRECISION="$2";      shift 2 ;;
        --strategy)       STRATEGY="$2";       shift 2 ;;
        --rr)             REQUEST_RATE="$2";   shift 2 ;;
        --num-prompts)    NUM_PROMPTS="$2";    shift 2 ;;
        --heter-config)   HETER_CONFIG="$2";   shift 2 ;;
        --partial-config) PARTIAL_CONFIG="$2"; shift 2 ;;
        --tp)             TP="$2";             shift 2 ;;
        --ep)             EP="$2";             shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$PRECISION" ]]; then
    echo "ERROR: --precision is required (bf16 | int4 | heter)"
    exit 1
fi

if [[ "$PRECISION" != "bf16" && "$PRECISION" != "int4" && "$PRECISION" != "heter" ]]; then
    echo "ERROR: --precision must be one of: bf16, int4, heter"
    exit 1
fi

if [[ "$PRECISION" == "heter" ]]; then
    # Default strategy
    if [[ -z "$STRATEGY" ]]; then
        STRATEGY="tp"
    fi
    if [[ "$STRATEGY" != "tp" && "$STRATEGY" != "ep" && "$STRATEGY" != "partial" ]]; then
        echo "ERROR: --strategy must be one of: tp, ep, partial"
        exit 1
    fi
    # Default heter config
    if [[ -z "$HETER_CONFIG" ]]; then
        HETER_CONFIG="$DEFAULT_HETER_CONFIG"
    fi
    # Partial requires partial-config
    if [[ "$STRATEGY" == "partial" && -z "$PARTIAL_CONFIG" ]]; then
        echo "ERROR: --partial-config is required when --strategy partial"
        exit 1
    fi
fi

if [[ -n "$TP" && -n "$EP" ]]; then
    echo "ERROR: --tp and --ep are mutually exclusive"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build tag for output file
# ---------------------------------------------------------------------------
TAG="$PRECISION"
if [[ "$PRECISION" == "heter" ]]; then
    TAG="${TAG}_${STRATEGY}"
fi
if [[ -n "$TP" ]]; then TAG="${TAG}_tp${TP}"; fi
if [[ -n "$EP" ]]; then TAG="${TAG}_ep${EP}"; fi

OUT_DIR="$SCRIPT_DIR/results"
mkdir -p "$OUT_DIR"
OUTPUT_FILE="$OUT_DIR/${TAG}_rr${REQUEST_RATE}_n${NUM_PROMPTS}.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_for_server() {
    echo "Waiting for server at $BASE_URL ..."
    local max_wait=600
    local elapsed=0
    while ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            exit 1
        fi
    done
    echo "Server ready (${elapsed}s)"
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Killing server process tree (pid=$SERVER_PID) ..."
        # Kill the entire process tree rooted at SERVER_PID
        pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        # Force kill any survivors
        pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
    fi
}
trap kill_server EXIT

# ---------------------------------------------------------------------------
# Build server args
# ---------------------------------------------------------------------------
build_server_args() {
    local model="$1"
    local args=(
        --model-path "$model"
        --host "$HOST" --port "$PORT"
        --trust-remote-code
    )

    if [[ -n "$TP" ]]; then
        args+=(--tp-size "$TP")
    fi

    if [[ -n "$EP" ]]; then
        args+=(--ep-size "$EP")
    fi

    # Heter-specific args
    if [[ "$PRECISION" == "heter" ]]; then
        args+=(--heter-precision-config "$HETER_CONFIG")
    fi

    echo "${args[@]}"
}

# ---------------------------------------------------------------------------
# Launch server
# ---------------------------------------------------------------------------
launch_server() {
    local model
    case "$PRECISION" in
        bf16)  model="$BF16_MODEL" ;;
        int4)  model="$INT4_MODEL" ;;
        heter) model="$BF16_MODEL" ;;
    esac

    local server_args
    server_args=$(build_server_args "$model")

    echo "Launching server: python3 -m sglang.launch_server $server_args"
    python3 -m sglang.launch_server $server_args &
    SERVER_PID=$!
    wait_for_server
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Precision:    $PRECISION"
if [[ "$PRECISION" == "heter" ]]; then
echo "  Strategy:     $STRATEGY"
echo "  Heter config: $HETER_CONFIG"
if [[ -n "$PARTIAL_CONFIG" ]]; then
echo "  Partial cfg:  $PARTIAL_CONFIG"
fi
fi
if [[ -n "$TP" ]]; then echo "  TP size:      $TP"; fi
if [[ -n "$EP" ]]; then echo "  EP size:      $EP"; fi
echo "  Request rate: $REQUEST_RATE"
echo "  Num prompts:  $NUM_PROMPTS"
echo "  Output:       $OUTPUT_FILE"
echo "============================================================"

launch_server

echo ""
echo "Running bench_serving ..."
python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --dataset-name sharegpt \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-file "$OUTPUT_FILE"

kill_server

echo ""
echo "Done. Results: $OUTPUT_FILE"
