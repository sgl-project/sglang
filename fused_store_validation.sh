#!/bin/bash
# E2e validation for the fused-store branch. Run inside the container.
set -u
export PYTHONPATH=/SGL/sglang-fused-store/python:${PYTHONPATH:-}
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1

launch () { # model, extra args...
  local model="$1"; shift
  python -m sglang.launch_server --model-path "$model" \
    --dsv4-attn-backend trtllm --trust-remote-code --tp 8 \
    --moe-runner-backend flashinfer_mxfp4 --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune --host 127.0.0.1 --port 21000 "$@" \
    > /tmp/fused_server.log 2>&1 &
  for i in $(seq 1 240); do
    grep -qa "The server is fired up" /tmp/fused_server.log 2>/dev/null && return 0
    grep -qaE "Scheduler hit an exception" /tmp/fused_server.log 2>/dev/null && return 1
    sleep 10
  done
  return 1
}

echo "=== [gsm8k-flash] launching $(date) ==="
if launch deepseek-ai/DeepSeek-V4-Flash --mem-fraction-static 0.85 --max-running-requests 32; then
  echo "=== [gsm8k-flash] up $(date) ==="
  python - <<'EOF'
from types import SimpleNamespace
from sglang.test.run_eval import run_eval
args = SimpleNamespace(base_url="http://127.0.0.1:21000",
    model="deepseek-ai/DeepSeek-V4-Flash", eval_name="gsm8k",
    api="completion", max_tokens=512, num_examples=200, num_threads=64)
print("GSM8K fused-store trtllm (V4-Flash):", run_eval(args)["score"])
EOF
else
  echo "=== [gsm8k-flash] SERVER FAILED ==="
fi
pkill -9 -f launch_server; sleep 15

echo "=== [itl-pro] launching $(date) ==="
if launch deepseek-ai/DeepSeek-V4-Pro --mem-fraction-static 0.88 --max-running-requests 32; then
  echo "=== [itl-pro] up $(date) ==="
  python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 21000 \
    --dataset-name random --random-input-len 131072 --random-output-len 128 \
    --random-range-ratio 1 --num-prompts 8 --max-concurrency 4 2>&1 \
    | grep -E "throughput \(tok/s\)|Mean TTFT|Median ITL|Mean TPOT|Mean E2E"
else
  echo "=== [itl-pro] SERVER FAILED ==="
fi
pkill -9 -f launch_server
echo "=== FUSED VALIDATION DONE $(date) ==="
