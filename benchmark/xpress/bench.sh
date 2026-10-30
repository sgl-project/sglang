#!/bin/sh
set -eu

# Usage: TARGET_MODEL=/path/to/Qwen3-8B ./benchmark/xpress/bench.sh xpress
# Restart the server with serve.sh dflash, dspark, or target, then rerun.
mode="${1:-xpress}"
case "$mode" in
    xpress|dflash|dspark|target) ;;
    *)
        echo "Usage: $0 [xpress|dflash|dspark|target]" >&2
        exit 2
        ;;
esac

script_dir="$(CDPATH= cd "$(dirname "$0")" && pwd)"
if [ -d "$script_dir/../../python/sglang" ]; then
    repo_root="$(CDPATH= cd "$script_dir/../.." && pwd)"
elif [ -d "$script_dir/python/sglang" ]; then
    repo_root="$script_dir"
elif [ -n "${SGLANG_REPO:-}" ] && [ -d "$SGLANG_REPO/python/sglang" ]; then
    repo_root="$(CDPATH= cd "$SGLANG_REPO" && pwd)"
else
    echo "Cannot find SGLang source; set SGLANG_REPO to its repository root." >&2
    exit 1
fi
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-python3}"
target_model="${TARGET_MODEL:-}"
port="${PORT:-30000}"
output_dir="${OUTPUT_DIR:-/tmp/sglang-xpress-bench}"
dataset_path="${DATASET_PATH:-$script_dir/data/gsm8k_test_40.json}"
num_prompts="${NUM_PROMPTS:-40}"
input_len="${INPUT_LEN:-1024}"
output_len="${OUTPUT_LEN:-100}"
warmup_requests="${WARMUP_REQUESTS:-4}"
seed="${SEED:-42}"
# This benchmark sends token IDs and reads its sample file locally. Keep the
# tokenizer local as well so an air-gapped run never contacts Hugging Face.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [ -z "$target_model" ] || [ ! -d "$target_model" ]; then
    echo "Set TARGET_MODEL to the local target model/tokenizer directory." >&2
    exit 2
fi
if [ ! -f "$dataset_path" ]; then
    echo "Local dataset not found: $dataset_path" >&2
    exit 2
fi
case "$num_prompts" in
    ''|*[!0-9]*) echo "NUM_PROMPTS must be a positive integer." >&2; exit 2 ;;
esac
if [ "$num_prompts" -eq 0 ]; then
    echo "NUM_PROMPTS must be positive." >&2
    exit 2
fi
if [ "$num_prompts" -gt 40 ]; then
    echo "This small-sample benchmark supports at most 40 prompts." >&2
    exit 2
fi
"$python_bin" - "$dataset_path" "$num_prompts" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    rows = json.load(f)
usable = sum(
    len(row.get("conversations", row.get("conversation", []))) >= 2
    for row in rows
)
requested = int(sys.argv[2])
if usable < requested:
    raise SystemExit(
        f"Local dataset has {usable} usable rows, fewer than NUM_PROMPTS={requested}."
    )
print(f"Using {requested} of {usable} local GSM8K questions.")
PY
# CONCURRENCIES is a whitespace-separated list; disable pathname expansion.
set -f

mkdir -p "$output_dir"
for concurrency in ${CONCURRENCIES:-1 4 16 32}; do
    case "$concurrency" in
        ''|*[!0-9]*)
            echo "Invalid concurrency: $concurrency" >&2
            exit 2
            ;;
    esac
    if [ "$concurrency" -eq 0 ]; then
        echo "Concurrency must be positive." >&2
        exit 2
    fi
    if [ "$concurrency" -gt "$num_prompts" ]; then
        echo "Skipping concurrency $concurrency: only $num_prompts prompts." >&2
        continue
    fi
    result="$output_dir/${mode}-c${concurrency}.jsonl"
    if [ -e "$result" ]; then
        echo "Result already exists: $result. Choose a new OUTPUT_DIR." >&2
        exit 1
    fi
    echo "Benchmarking $mode at concurrency=$concurrency; result=$result" >&2
    "$python_bin" -m sglang.benchmark.serving \
        --backend sglang \
        --host 127.0.0.1 \
        --port "$port" \
        --model "$target_model" \
        --tokenizer "$target_model" \
        --dataset-name random \
        --dataset-path "$dataset_path" \
        --tokenize-prompt \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --random-range-ratio 1 \
        --num-prompts "$num_prompts" \
        --warmup-requests "$warmup_requests" \
        --request-rate inf \
        --max-concurrency "$concurrency" \
        --seed "$seed" \
        --temperature 0 \
        --tag "$mode-c$concurrency" \
        --output-file "$result"
done

"$python_bin" - "$output_dir" "$num_prompts" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = {}
for path in sorted(directory.glob("*.jsonl")):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f"Expected one result in {path}, found {len(lines)}")
    result = json.loads(lines[0])
    if result["completed"] != expected:
        raise SystemExit(
            f"{path}: completed {result['completed']} of {expected} requests"
        )
    mode, concurrency = path.stem.rsplit("-c", 1)
    rows[(mode, int(concurrency))] = result

print("\nmode       concurrency   output tok/s   mean TPOT ms   accept length")
for (mode, concurrency), result in sorted(rows.items()):
    accept = result.get("accept_length")
    accept_text = f"{accept:.3f}" if isinstance(accept, (int, float)) else "n/a"
    print(
        f"{mode:<10} {concurrency:>11}   {result['output_throughput']:>12.1f}"
        f"   {result['mean_tpot_ms']:>12.2f}   {accept_text:>13}"
    )

for concurrency in sorted({concurrency for _, concurrency in rows}):
    for baseline_mode in ("dflash", "dspark", "target"):
        if ("xpress", concurrency) not in rows or (baseline_mode, concurrency) not in rows:
            continue
        result = rows[("xpress", concurrency)]
        baseline = rows[(baseline_mode, concurrency)]["output_throughput"]
        speedup = result["output_throughput"] / baseline
        print(f"XPress / {baseline_mode} output throughput at c={concurrency}: {speedup:.3f}x")
PY
