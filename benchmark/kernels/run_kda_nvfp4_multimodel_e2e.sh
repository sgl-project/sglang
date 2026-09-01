#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 MODEL_PATH MODEL_NAME RESULTS_DIR [PORT]" >&2
  exit 2
fi

model_path=$1
model_name=$2
results_dir=$3
port=${4:-31000}
server_pid=

mkdir -p "$results_dir"

stop_server() {
  if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap stop_server EXIT INT TERM

wait_for_server() {
  python - "$port" <<'PY'
import sys
import time
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/health"
deadline = time.time() + 600
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit(f"server did not become healthy: {url}")
PY
}

run_bench() {
  local output_file=$1
  local prompts=$2
  local input_len=$3
  local output_len=$4
  python -m sglang.bench_serving \
    --backend sglang-oai \
    --host 127.0.0.1 \
    --port "$port" \
    --model "$model_path" \
    --dataset-name random-ids \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --random-range-ratio 1 \
    --num-prompts "$prompts" \
    --max-concurrency 1 \
    --request-rate inf \
    --seed 1234 \
    --flush-cache \
    --disable-tqdm \
    --output-file "$output_file"
}

run_mode() {
  local label=$1
  local enabled=$2
  local rounds=$3
  local server_log="$results_dir/${label}_server.log"
  local output_file="$results_dir/${label}.jsonl"
  local warmup_file="$results_dir/${label}_warmup.jsonl"

  rm -f "$output_file" "$warmup_file"
  SGLANG_ENABLE_KDA_NVFP4_GEMM=$enabled setsid python -m sglang.launch_server \
    --trust-remote-code \
    --model-path "$model_path" \
    --tp-size 1 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --chunked-prefill-size 2048 \
    --disable-radix-cache \
    --max-running-requests 8 \
    --cuda-graph-max-bs 8 \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3 \
    --port "$port" \
    >"$server_log" 2>&1 &
  server_pid=$!
  wait_for_server

  run_bench "$warmup_file" 2 256 32 \
    >"$results_dir/${label}_warmup.log" 2>&1
  for round in $(seq 1 "$rounds"); do
    run_bench "$output_file" 10 2048 512 \
      >"$results_dir/${label}_round${round}.log" 2>&1
  done

  if [[ $enabled == 1 ]] && ! grep -q "Using the Humanize2/KDA ModelOpt NVFP4 GEMM" "$server_log"; then
    echo "KDA fast path was not observed for $model_name" >&2
    return 1
  fi
  stop_server
}

printf '%s\n' "$model_path" >"$results_dir/model_path.txt"
printf '%s\n' "$model_name" >"$results_dir/model_name.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
  >"$results_dir/gpu.txt"

run_mode baseline 0 3
run_mode candidate 1 3
run_mode baseline_adjacent 0 1
