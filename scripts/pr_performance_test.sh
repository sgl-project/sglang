#!/bin/bash
set -e  # Exit immediately if any command fails

# Define cleanup function to clean up resources on exit (including abnormal exit)
cleanup() {
    local exit_code=$?
    echo -e "\nExit detected, starting resource cleanup..."
    # Stop SGLang server process (if exists and running)
    if [ -n "$SGLANG_PID" ] && ps -p "$SGLANG_PID" >/dev/null 2>&1; then
        echo "Stopping SGLang server process (PID: $SGLANG_PID)"
        kill "$SGLANG_PID" >/dev/null 2>&1 || echo "Warning: Failed to stop server process"
        wait "$SGLANG_PID" 2>/dev/null || true
    fi
    # Delete temporary log file (if exists)
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "Deleting temporary log file: $LOG_FILE"
        rm -f "$LOG_FILE" || echo "Warning: Failed to delete temporary log file"
    fi
    echo "Cleanup completed."
    exit $exit_code  # Preserve original exit code
}

# Set trap to catch all exit signals (normal exit, error exit, Ctrl+C, etc.)
trap cleanup EXIT

# Check if at least 2 parameters are passed (target path and model path)
if [ $# -lt 2 ]; then
    echo "Error: Please pass at least 2 parameters!"
    echo "Usage: $0 <Parameter 1: Target Path> <Parameter 2: Model Path> [Parameter 3: TP Count]"
    exit 1  # Non-zero exit code indicates error
fi

target_dir="$1"
model_path="$2"
# TP parameter default value is 1
tp=${3:-1}
extra_args="$4"
dataset="/opt/benchmark/ais_bench/datasets"
config_file="/opt/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"

start_sglang() {
    # Create temporary log file
    LOG_FILE=$(mktemp /tmp/sglang_server.log.XXXXXX)
    echo "Server logs will be temporarily stored in: $LOG_FILE"

    # Define server startup parameter array
    args=(
        --model-path "$model_path"
        --tp "$tp"
        --host 127.0.0.1
        --port 8080
        --attention-backend ascend
        --mem-fraction-static 0.8
        --disable-cuda-graph
    )

    # If the 4th parameter exists, split it into array elements and add to the parameter list
    if [ -n "$extra_args" ]; then
        # Split the 4th parameter into an array by spaces and add to args
        read -ra extraargs <<< "$extra_args"
        args+=("${extraargs[@]}")
    fi

    # Start SGLang server (pass all parameters using the parameter array)
    echo "Starting SGLang server..."
    python -m sglang.launch_server "${args[@]}" > "$LOG_FILE" 2>&1 &
    SGLANG_PID=$!  # Record server process ID
}

wait_sglang_start_success() {
    # Define log identifier for successful server startup
    SUCCESS_LOG="The server is fired up and ready to roll!"
    # Wait for server startup success (check log keyword)
    echo "Waiting for server startup, monitoring log keyword: '$SUCCESS_LOG'"
    WAIT_TIMEOUT=500  # Timeout period (seconds)
    WAIT_INTERVAL=10   # Check interval (seconds)
    ELAPSED=0
    SUCCESS=0

    while [ $ELAPSED -lt $WAIT_TIMEOUT ]; do
        if grep -q "$SUCCESS_LOG" "$LOG_FILE"; then
            SUCCESS=1
            break
        fi
        echo "Server not ready, waited ${ELAPSED}s (remaining timeout: $((WAIT_TIMEOUT - ELAPSED))s)"
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    done

    # Handle server startup result
    if [ $SUCCESS -eq 1 ]; then
        echo "Detected successful server startup log!"
    else
        echo "Error: Server startup timed out (waited ${WAIT_TIMEOUT}s)"
        echo "===== Last 100 lines of server log ====="
        tail -n 100 "$LOG_FILE"
        echo "==========================="
        exit 1  # Trigger cleanup trap
    fi
}

prepare_ais_bench_config() {
    # Copy dataset
    echo "Copying dataset..."
    if [[ ! -d "$dataset" ]]; then
        echo "Error: Path does not exist -> $dataset"
        exit 1
    fi
    cp -r ~/.cache/modelscope/hub/datasets/gsm8k "$dataset" || {
        echo "Error: Failed to copy dataset"
        exit 1
    }

    # Modify Python configuration file
    echo "Modifying configuration files..."
    if [[ ! -f "$config_file" ]]; then
        echo "Error: Path does not exist or is not a file -> $config_file"
        exit 1
    fi
    sed -i "s/localhost/127.0.0.1/g" "$config_file" || {
        echo "Error: Failed to replace localhost"
        exit 1
    }
    sed -i "s|path=\"[^\"]*\"|path=\"$model_path\"|" "$config_file" || {
        echo "Error: Failed to replace path"
        exit 1
    }
    sed -i "s|model=\"[^\"]*\"|model=\"$model_path\"|" "$config_file" || {
        echo "Error: Failed to replace model"
        exit 1
    }
}

# Set batch_size (concurrency) to 10 times the request_rate to meet request_rate requirements,
# set num-prompts to 5 times the concurrency to fully utilize concurrency
run_ais_bench() {
    sed -i 's/request_rate[[:space:]]*=[[:space:]]*.*/request_rate = '$1',/' "$config_file" || {
        echo "Error: Failed to set request_rate"
        exit 1
    }
    sed -i 's/batch_size[[:space:]]*=[[:space:]]*.*/batch_size = '$(( $1 * 10 ))',/' "$config_file" || {
        echo "Error: Failed to set batch_size"
        exit 1
    }
    cat "$config_file"
    # Execute test
    echo "Starting ais_bench test execution..."
    ais_bench \
        --models vllm_api_stream_chat \
        --datasets gsm8k_gen_0_shot_cot_str_perf \
        --summarizer default_perf \
        --mode perf \
        --num-prompts $(( $1 * 10 * 5 )) || {
        echo "Error: ais_bench test execution failed"
        exit 1
    }

    # Move output results
    echo "Moving test results..."
    OUTPUTS_DIR="$(ls -d "outputs/default"/* 2>/dev/null | grep -E '^.*/[0-9]{8}_[0-9]{6}$' | sort | tail -1)/performances/vllm-api-stream-chat"
    if [ ! -d "$OUTPUTS_DIR" ]; then
        echo "Error: No valid output directory found: '$OUTPUTS_DIR'" >&2
        exit 1
    fi
    # Add two fields at once and save to original file
    jq --arg tp_val "$tp" --arg request_rate_val "$1" '. += {"tp": {"total": $tp_val}, "request_rate": {"total": $request_rate_val}}' \
    "$OUTPUTS_DIR/gsm8kdataset.json" > "$OUTPUTS_DIR/gsm8kdataset.json.tmp" && mv "$OUTPUTS_DIR/gsm8kdataset.json.tmp" "$OUTPUTS_DIR/gsm8kdataset.json"
    model=$(basename "$model_path")
    echo "$model"
    mkdir -p "$target_dir/$model"
    mv "$OUTPUTS_DIR" "$target_dir/$model/$1" || {
        echo "Error: Failed to move output directory"
        exit 1
    }
}

main() {
    start_sglang
    wait_sglang_start_success
    prepare_ais_bench_config
    time run_ais_bench 1
    time run_ais_bench 4
    time run_ais_bench 16
}

main

echo "All operations completed successfully!"
