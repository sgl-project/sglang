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
if [[ "${BATCH_INVARIANT_MM:-0}" == 1 ]]; then
  export PYTHONPATH="$benchmark_root/batch_invariant_mm:$PYTHONPATH"
  export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
fi
export SGLANG_ENABLE_GDN_KV_REUSE="$reuse_flag"
export SGLANG_FLASHINFER_USE_PAGED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
overlap_args=()
if [[ "${ENABLE_OVERLAP:-0}" != 1 ]]; then
  overlap_args+=(--disable-overlap-schedule)
fi
exec "${PYTHON:-python3}" -m sglang.launch_server \
  --model-path "$model_path" --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --attention-backend flashinfer \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --mamba-ssm-dtype "${MAMBA_SSM_DTYPE:-bfloat16}" --mamba-radix-cache-strategy extra_buffer \
  --page-size 256 --chunked-prefill-size 4096 --max-prefill-tokens 8192 \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-32768}" --context-length 16384 \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-4}" \
  --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE:-32}" --mamba-max-states-per-path 1 --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
  "${overlap_args[@]}" --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode disabled --log-level debug --enable-request-time-stats-logging
