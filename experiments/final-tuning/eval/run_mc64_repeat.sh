#!/bin/bash
# Re-run mc=64 only for both baseline and treatment to check stability.
# Both servers with PCG enabled (fair).
set -o pipefail
cd "$(dirname "$0")/../../.."

EVAL_DIR=experiments/final-tuning/eval
LOG_DIR=$EVAL_DIR/logs_mc64rerun
RESULT_DIR=$EVAL_DIR/results_mc64rerun
mkdir -p "$LOG_DIR" "$RESULT_DIR"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
SHAREGPT="/data/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"

BASELINE_CFG=$EVAL_DIR/configs/baseline_int4_only.json
TREATMENT_CFG=$EVAL_DIR/configs/treatment_mc64.json
NUM_PROMPTS=${NUM_PROMPTS:-300}
MC=64

eval "$(conda shell.bash hook)"
conda activate sglang

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/driver.log" >&2; }

launch() {
    local label=$1 cfg=$2 port=$3 gpus=$4
    log "launching $label on GPU $gpus port $port (PCG enabled)"
    CUDA_VISIBLE_DEVICES=$gpus python -m sglang.launch_server \
        --model-path "$BF16_MODEL" --tp 1 \
        --host 127.0.0.1 --port "$port" --trust-remote-code \
        --heter-precision-config "$cfg" \
        --mem-fraction-static 0.83 \
        > "$LOG_DIR/server_${label}.log" 2>&1 &
    echo $!
}

wait_ready() {
    local port=$1 pid=$2 elapsed=0
    while ! curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: server pid $pid died (port $port)"; return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge 1800 ]; then log "TIMEOUT"; return 1; fi
    done
    log "  ready (port $port) after ${elapsed}s"
}

bench_one() {
    local label=$1 port=$2
    local out=$RESULT_DIR/${label}_mc${MC}.jsonl
    log "bench $label mc=$MC → $out"
    curl -s -X POST "http://127.0.0.1:${port}/flush_cache" > /dev/null 2>&1 || true
    sleep 1
    python -m sglang.bench_serving \
        --backend sglang --base-url "http://127.0.0.1:${port}" \
        --dataset-name sharegpt --dataset-path "$SHAREGPT" \
        --num-prompts "$NUM_PROMPTS" --max-concurrency "$MC" \
        --output-file "$out" \
        > "$LOG_DIR/${label}_mc${MC}.log" 2>&1 || log "  bench failed"
}

BPID=$(launch baseline "$BASELINE_CFG" 31000 2)
TPID=$(launch treatment "$TREATMENT_CFG" 32000 3)
trap 'kill -9 $BPID $TPID 2>/dev/null; exit' INT TERM

wait_ready 31000 "$BPID" || { kill -9 $BPID $TPID 2>/dev/null; exit 1; }
wait_ready 32000 "$TPID" || { kill -9 $BPID $TPID 2>/dev/null; exit 1; }

bench_one baseline 31000
bench_one treatment 32000

log "tearing down"
kill -INT $BPID $TPID 2>/dev/null; sleep 5
kill -9 $BPID $TPID 2>/dev/null
log "DONE"
