#!/bin/sh
set -eu

# PR-style local samples: greedy, up to 2048 new tokens.
# Uses 40 questions per dataset, so it is a paired sanity benchmark.
mode="${1:-xpress}"
case "$mode" in
    xpress|dflash|dspark|target) ;;
    *) echo "Usage: $0 [xpress|dflash|dspark|target]" >&2; exit 2 ;;
esac

script_dir="$(CDPATH= cd "$(dirname "$0")" && pwd)"
repo_root="$(CDPATH= cd "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python_bin="${PYTHON_BIN:-python3}"
target_model="${TARGET_MODEL:-}"
port="${PORT:-30000}"
output_dir="${OUTPUT_DIR:-/tmp/sglang-xpress-pr-sample}"
dataset="${DATASET:-gsm8k}"
case "$dataset" in
    gsm8k) dataset_path="$script_dir/data/gsm8k_5shot_40.jsonl" ;;
    math500) dataset_path="$script_dir/data/math500_40.jsonl" ;;
    *) echo "DATASET must be gsm8k or math500." >&2; exit 2 ;;
esac
num_prompts="${NUM_PROMPTS:-40}"
max_output_len="${MAX_OUTPUT_LEN:-2048}"
seed="${SEED:-42}"
if [ -z "$target_model" ] || [ ! -d "$target_model" ]; then
    echo "Set TARGET_MODEL to the local target model/tokenizer directory." >&2
    exit 2
fi
if [ ! -f "$dataset_path" ]; then
    echo "Missing local dataset: $dataset_path" >&2
    exit 2
fi
case "$num_prompts" in
    ''|*[!0-9]*) echo "NUM_PROMPTS must be a positive integer." >&2; exit 2 ;;
esac
case "$max_output_len" in
    ''|*[!0-9]*) echo "MAX_OUTPUT_LEN must be a positive integer." >&2; exit 2 ;;
esac
if [ "$num_prompts" -lt 1 ] || [ "$num_prompts" -gt 40 ] || [ "$max_output_len" -lt 4 ]; then
    echo "Use 1..40 prompts and MAX_OUTPUT_LEN >= 4." >&2
    exit 2
fi
case "$mode" in
    xpress|dspark) rounds="${ROUNDS:-2}" ;;
    *) rounds="${ROUNDS:-1}" ;;
esac
case "$rounds" in
    ''|*[!0-9]*) echo "ROUNDS must be a positive integer." >&2; exit 2 ;;
esac
if [ "$rounds" -lt 1 ]; then
    echo "ROUNDS must be positive." >&2
    exit 2
fi
set -f
mkdir -p "$output_dir"
round=1
while [ "$round" -le "$rounds" ]; do
    for concurrency in ${CONCURRENCIES:-1 4 16 32}; do
        case "$concurrency" in
            ''|*[!0-9]*) echo "Invalid concurrency: $concurrency" >&2; exit 2 ;;
        esac
        if [ "$concurrency" -lt 1 ]; then
            echo "Concurrency must be positive." >&2
            exit 2
        fi
        if [ "$concurrency" -gt "$num_prompts" ]; then
            echo "Skipping c=$concurrency: only $num_prompts prompts." >&2
            continue
        fi
        result="$output_dir/${dataset}-${mode}-r${round}-c${concurrency}.jsonl"
        if [ -e "$result" ]; then
            echo "Result already exists: $result" >&2
            exit 1
        fi
        echo "Benchmarking $dataset $mode round=$round concurrency=$concurrency" >&2
        "$python_bin" -m sglang.benchmark.serving \
            --backend sglang \
            --host 127.0.0.1 \
            --port "$port" \
            --model "$target_model" \
            --tokenizer "$target_model" \
            --dataset-name custom \
            --dataset-path "$dataset_path" \
            --sharegpt-output-len "$max_output_len" \
            --disable-ignore-eos \
            --num-prompts "$num_prompts" \
            --warmup-requests 4 \
            --flush-cache \
            --request-rate inf \
            --max-concurrency "$concurrency" \
            --seed "$seed" \
            --temperature 0 \
            --tag "pr-$dataset-$mode-r$round-c$concurrency" \
            --output-file "$result"
    done
    round=$((round + 1))
done

"$python_bin" - "$output_dir" "$num_prompts" <<'PY'
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

directory, expected = Path(sys.argv[1]), int(sys.argv[2])
rows = defaultdict(list)
for path in sorted(directory.glob("*.jsonl")):
    match = re.fullmatch(
        r"(gsm8k|math500)-(xpress|dflash|dspark|target)-r\d+-c(\d+)",
        path.stem,
    )
    if match is None:
        continue
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f"Expected one result in {path}, found {len(lines)}")
    result = json.loads(lines[0])
    if result["completed"] != expected:
        raise SystemExit(f"{path}: completed {result['completed']} of {expected}")
    rows[(match.group(1), match.group(2), int(match.group(3)))].append(result)

print("dataset  mode       concurrency  rounds  output tok/s  mean TPOT ms  accept length")
for (dataset, mode, concurrency), results in sorted(rows.items()):
    throughput = statistics.mean(row["output_throughput"] for row in results)
    tpot = statistics.mean(row["mean_tpot_ms"] for row in results)
    accepts = [row["accept_length"] for row in results if row.get("accept_length") is not None]
    accept = f"{statistics.mean(accepts):.3f}" if accepts else "n/a"
    print(f"{dataset:<8} {mode:<10} {concurrency:>11} {len(results):>7} {throughput:>13.1f} {tpot:>13.2f} {accept:>14}")

for dataset, concurrency in sorted({(key[0], key[2]) for key in rows}):
    xpress = rows.get((dataset, "xpress", concurrency))
    if not xpress:
        continue
    xpress_rate = statistics.mean(row["output_throughput"] for row in xpress)
    for baseline_mode in ("dflash", "dspark", "target"):
        baseline = rows.get((dataset, baseline_mode, concurrency))
        if baseline:
            baseline_rate = statistics.mean(row["output_throughput"] for row in baseline)
            print(f"{dataset} XPress / {baseline_mode} at c={concurrency}: {xpress_rate / baseline_rate:.3f}x")
PY
