#!/bin/bash
# BCG vs PCG comparison test runner
# Usage: bash run_bcg_comparison.sh

set -e

VENV=/home/yuwei/sgl-workspace/.venv/bin/python
LOG_DIR=/data1/yuweia/bcg_comparison
RESULT_FILE=$LOG_DIR/results.txt
PORT=30000

mkdir -p $LOG_DIR

run_test() {
    local MODEL=$1
    local SET_NAME=$2
    local MODE=$3  # "pcg" or "bcg"
    local EXTRA_ARGS=$4
    local NUM_QUESTIONS=${5:-200}

    local SERVER_LOG="$LOG_DIR/${SET_NAME}_${MODE}_server.log"
    local EVAL_LOG="$LOG_DIR/${SET_NAME}_${MODE}_eval.log"

    echo "=== $SET_NAME / $MODE ===" | tee -a $RESULT_FILE

    # Build server command
    local CMD="$VENV -m sglang.launch_server --model $MODEL --port $PORT --disable-radix-cache"
    if [ "$MODE" = "bcg" ]; then
        CMD="$CMD --enable-breakable-cuda-graph"
    fi
    CMD="$CMD $EXTRA_ARGS"

    echo "  Launching: $CMD"
    eval "$CMD" > "$SERVER_LOG" 2>&1 &
    local SERVER_PID=$!

    # Wait for server to be ready (up to 10 min)
    local ready=0
    for i in $(seq 1 60); do
        sleep 10
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q "200"; then
            echo "  Server ready after ${i}0s"
            ready=1
            break
        fi
    done

    if [ $ready -eq 0 ]; then
        echo "  ERROR: Server failed to start!" | tee -a $RESULT_FILE
        kill -9 $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        sleep 5
        return 1
    fi

    # Run GSM8K eval
    echo "  Running GSM8K eval ($NUM_QUESTIONS questions)..."
    $VENV -m sglang.test.run_eval \
        --base-url "http://localhost:$PORT" \
        --model "$MODEL" \
        --eval-name mgsm_en \
        --num-examples $NUM_QUESTIONS \
        --num-threads 32 \
        > "$EVAL_LOG" 2>&1 || true

    # Extract results
    local SCORE=$(grep -oP '"score":\s*[\d.]+' "$EVAL_LOG" | grep -oP '[\d.]+' | tail -1)
    if [ -z "$SCORE" ]; then
        SCORE=$(grep -i "score\|accuracy" "$EVAL_LOG" | tail -3)
    fi

    # Get throughput from server log
    local THROUGHPUT=$(grep "gen throughput" "$SERVER_LOG" | tail -5 | grep -oP 'gen throughput \(token/s\): [\d.]+' | grep -oP '[\d.]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
    local INPUT_THROUGHPUT=$(grep "input throughput" "$SERVER_LOG" | tail -5 | grep -oP 'input throughput \(token/s\): [\d.]+' | grep -oP '[\d.]+$' | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')

    echo "  Score: $SCORE | Decode throughput: $THROUGHPUT tok/s | Prefill throughput: $INPUT_THROUGHPUT tok/s" | tee -a $RESULT_FILE

    # Kill server
    kill -9 $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5

    return 0
}

clean_model_weights() {
    local MODEL=$1
    local CACHE_DIR="$HOME/.cache/huggingface/hub/models--${MODEL//\//-}"
    if [ -d "$CACHE_DIR" ]; then
        local SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
        echo "Removing model weights: $CACHE_DIR ($SIZE)" | tee -a $RESULT_FILE
        rm -rf "$CACHE_DIR"
    fi
}

echo "========================================" | tee $RESULT_FILE
echo "BCG vs PCG Comparison Test" | tee -a $RESULT_FILE
echo "Date: $(date)" | tee -a $RESULT_FILE
echo "========================================" | tee -a $RESULT_FILE
echo "" | tee -a $RESULT_FILE
