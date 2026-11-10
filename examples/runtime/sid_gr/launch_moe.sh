#!/usr/bin/env bash
# Native Qwen3-MoE with existing GR beam/catalog support; pass TP/EP explicitly.
set -euo pipefail
exec python -m sglang.launch_server \
  --skip-tokenizer-init --enable-gr --dtype bfloat16 \
  --context-length 8192 --page-size 1 --attn-dp-size 1 \
  --attention-backend flashinfer --moe-a2a-backend none \
  --moe-runner-backend triton --chunked-prefill-size -1 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --disable-radix-cache --random-seed 20261014 \
  --mem-fraction-static 0.2 --max-total-tokens 65536 \
  --max-running-requests 512 --host 127.0.0.1 --port 30000 \
  --dist-timeout 120 --watchdog-timeout 120 "$@"
