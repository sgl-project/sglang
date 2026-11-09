#!/usr/bin/env bash
set -euo pipefail

# Usage: launch_server.sh MODEL_PATH REUSE_FLAG [PORT]
model_path=$1
reuse_flag=$2
port=${3:-30000}
benchmark_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$benchmark_root/../.." && pwd)
export PYTHONPATH="$repository_root/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$model_path" == */Olmo-Hybrid-7B ]]; then
  export PYTHONPATH="$benchmark_root/olmo_fixture:$PYTHONPATH"
  export SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_adapter
else
  unset SGLANG_EXTERNAL_MODEL_PACKAGE
fi
export SGLANG_ENABLE_GDN_KV_REUSE="$reuse_flag"
export SGLANG_FLASHINFER_USE_PAGED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python3}" -m sglang.launch_server \
  --model-path "$model_path" --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --attention-backend flashinfer \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer \
  --page-size 256 --chunked-prefill-size 4096 --max-prefill-tokens 8192 \
  --max-total-tokens 32768 --context-length 16384 --max-running-requests 4 \
  --max-mamba-cache-size 32 --mamba-max-states-per-path 1 --mem-fraction-static 0.8 \
  --disable-overlap-schedule --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode disabled --log-level debug --enable-request-time-stats-logging
