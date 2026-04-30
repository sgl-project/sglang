#!/bin/bash
# Treatment-only bench (PCG enabled). Reuse baseline numbers from
# results_pcgdisabled/sharegpt_e2e.csv (those didn't depend on the
# policy fix and were valid).
set -o pipefail

cd "$(dirname "$0")/../../.."

EVAL_DIR=experiments/final-tuning/eval
LOG_DIR=$EVAL_DIR/logs
RESULT_DIR=$EVAL_DIR/results
mkdir -p "$LOG_DIR" "$RESULT_DIR"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
SHAREGPT="/data/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"

TREATMENT_CFG=$EVAL_DIR/configs/treatment_mc64.json
PORT=32000
GPUS="3"
TP_SIZE=1
NUM_PROMPTS=${NUM_PROMPTS:-300}
MC_LIST=(32 64 128)

eval "$(conda shell.bash hook)"
conda activate sglang

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/driver_treatment.log" >&2; }

log "launching treatment on GPU $GPUS port $PORT (PCG enabled)"
CUDA_VISIBLE_DEVICES=$GPUS python -m sglang.launch_server \
    --model-path "$BF16_MODEL" \
    --tp "$TP_SIZE" \
    --host 127.0.0.1 --port "$PORT" \
    --trust-remote-code \
    --heter-precision-config "$TREATMENT_CFG" \
    --mem-fraction-static 0.83 \
    > "$LOG_DIR/server_treatment.log" 2>&1 &
PID=$!
log "pid=$PID"
trap 'kill -9 $PID 2>/dev/null; exit' INT TERM

# Wait up to 30 min for server to be ready
elapsed=0
while ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$PID" 2>/dev/null; then
        log "ERROR: server died (see $LOG_DIR/server_treatment.log)"
        exit 1
    fi
    sleep 5; elapsed=$((elapsed + 5))
    if [ $elapsed -ge 1800 ]; then
        log "ERROR: server did not become ready in 30 min"
        kill -9 $PID 2>/dev/null
        exit 1
    fi
done
log "server ready after ${elapsed}s"

for mc in "${MC_LIST[@]}"; do
    out=$RESULT_DIR/treatment_mc${mc}.jsonl
    log "bench_serving treatment mc=$mc → $out"
    curl -s -X POST "http://127.0.0.1:${PORT}/flush_cache" > /dev/null 2>&1 || true
    sleep 1
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${PORT}" \
        --dataset-name sharegpt \
        --dataset-path "$SHAREGPT" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$mc" \
        --output-file "$out" \
        > "$LOG_DIR/treatment_mc${mc}.log" 2>&1 || log "  bench failed (see $LOG_DIR/treatment_mc${mc}.log)"
done

log "tearing down treatment"
kill -INT "$PID" 2>/dev/null
sleep 5
kill -9 "$PID" 2>/dev/null
log "DONE"
