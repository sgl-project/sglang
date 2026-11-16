#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 BASE_URL MODEL_PATH SERVER_COMMIT OUTPUT_DIR" >&2
    exit 2
fi

base_url=$1
model_path=$2
server_commit=$3
output_dir=$4

if [[ ! "$server_commit" =~ ^[[:xdigit:]]{40}$ ]]; then
    echo "SERVER_COMMIT must be the full commit SHA of the running server." >&2
    exit 2
fi

# Use the same client checkout and model snapshot for both server revisions.
# Start each server on the same GPU with the same arguments and radix caching
# disabled so repeated prompts in warmup do not turn into prefix-cache hits.
mkdir -p "$output_dir"
run_dir="$output_dir/$server_commit"
mkdir "$run_dir"

for concurrency in 1 8; do
    result_file="$run_dir/concurrency-$concurrency.jsonl"
    python3 -m sglang.benchmark.serving \
        --backend sglang \
        --base-url "$base_url" \
        --model "$model_path" \
        --dataset-name random-ids \
        --tokenize-prompt \
        --random-input-len 8192 \
        --random-output-len 8 \
        --random-range-ratio 1 \
        --seed 42 \
        --num-prompts 64 \
        --max-concurrency "$concurrency" \
        --warmup-requests 8 \
        --output-file "$result_file"

    python3 - "$result_file" <<'PYTHON'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text())
if result.get("completed") != 64 or result.get("total_input_tokens") != 64 * 8192:
    raise SystemExit(f"{path}: benchmark did not complete the full 64 × 8192 workload")
PYTHON
done
