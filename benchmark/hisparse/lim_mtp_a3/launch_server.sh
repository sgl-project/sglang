#!/usr/bin/env bash
# Table 9 topology. Native MTP eligibility must be validated independently.
set -eo pipefail

CASE_ID=${CASE_ID:?Choose full_64k_b5 or offload_64k_b16_copysfa}
SOURCE_ROOT=${SGLANG_SOURCE_ROOT:?Set the isolated framework source root}
MODEL_PATH=${MODEL_PATH:?Set the candidate checkpoint path}
PORT=${PORT:-18898}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-512}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.96}
MTP=${MTP:-1}
GRAPH=${GRAPH:-1}
METADATA_FUSION=${METADATA_FUSION:-}
PRE_METADATA_FUSION=${PRE_METADATA_FUSION:-0}
NATIVE_MTP=${SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE:-}

case "$CASE_ID" in
  full_64k_b5) INPUT_TOKENS=64000; BATCH_SIZE=5; OFFLOAD=0; COPY_SFA=0; INDEXER_IMPL=legacy ;;
  offload_64k_b16_copysfa) INPUT_TOKENS=64000; BATCH_SIZE=16; OFFLOAD=1; COPY_SFA=1; INDEXER_IMPL=lim ;;
  *) echo "Unknown CASE_ID: $CASE_ID" >&2; exit 2 ;;
esac
CACHE_DESCRIPTION=12288
if (( ! OFFLOAD )); then CACHE_DESCRIPTION=full_sequence; fi
for name in PORT CONTEXT_LENGTH OUTPUT_TOKENS CHUNKED_PREFILL_SIZE MTP GRAPH; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "Invalid integer: $name" >&2; exit 2; }
done
OPTIMIZED_MTP=0
if [[ "$MTP" == 1 && "$OFFLOAD" == 1 ]]; then OPTIMIZED_MTP=1; fi
METADATA_FUSION=${METADATA_FUSION:-$OPTIMIZED_MTP}
NATIVE_MTP=${NATIVE_MTP:-$OPTIMIZED_MTP}
for name in MTP GRAPH METADATA_FUSION PRE_METADATA_FUSION NATIVE_MTP; do
  [[ "${!name}" =~ ^[01]$ ]] || { echo "Invalid boolean: $name" >&2; exit 2; }
done
if (( INPUT_TOKENS + OUTPUT_TOKENS + 16 > CONTEXT_LENGTH )); then
  echo 'Context length does not leave room for output and speculative rows' >&2
  exit 2
fi
# Requests own complete pages; align each request before summing the pool.
MAX_TOTAL_TOKENS=$(( BATCH_SIZE * ((INPUT_TOKENS + OUTPUT_TOKENS + 16 + 127) / 128) * 128 ))
args=(python3 -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name GLM-5.2-w4a8
  --attention-backend ascend --device npu
  --tp-size 16 --ep-size 16 --dp-size 1 --attn-dp-size 1
  --pp-size 1 --attn-cp-size 1 --dcp-size 1 --moe-dp-size 1
  --nnodes 1 --node-rank 0
  --trust-remote-code --quantization modelslim
  --dtype bfloat16 --kv-cache-dtype bfloat16 --page-size 128
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens 32768
  --prefill-decode-interval 0 --max-running-requests "$BATCH_SIZE"
  --max-total-tokens "$MAX_TOTAL_TOKENS" --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --moe-a2a-backend deepep --deepep-mode auto --load-balance-method round_robin
  --disable-shared-experts-fusion --disable-radix-cache
  --random-seed 42 --enable-metrics --host 127.0.0.1 --port "$PORT")
if (( MTP )); then
  args+=(--speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi
if (( GRAPH )); then
  args+=(--cuda-graph-bs-decode "$BATCH_SIZE")
else
  args+=(--disable-cuda-graph)
fi
if [[ "${1:-}" == --dry-run ]]; then
  printf 'case=%s global_bs=%s offload=%s copy_sfa=%s C=%s K=2048 mtp=%s graph=%s native_mtp=%s post_metadata=%s pre_metadata=%s\n' \
    "$CASE_ID" "$BATCH_SIZE" "$OFFLOAD" "$COPY_SFA" "$CACHE_DESCRIPTION" "$MTP" "$GRAPH" "$NATIVE_MTP" "$METADATA_FUSION" "$PRE_METADATA_FUSION"
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi
if (( $# != 0 )); then
  echo 'Only --dry-run is accepted; set configuration through the environment' >&2
  exit 2
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset SGLANG_ENABLE_SPEC_V2 SGLANG_NPU_LIM_LIBRARY VLLM_ASCEND_SRC CUSTOM_OP_ROOT
export PYTHONPATH="${KERNEL_PYTHON_ROOT:?Set the verified kernel package root}:$SOURCE_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_SET_CPU_AFFINITY=1 SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export STREAMS_PER_DEVICE=32 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export HCCL_BUFFSIZE=512 HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600
export DEEPEP_HYBRID_DEPLOYMENT=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1 DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1 SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
export SGLANG_NPU_PROFILING=0 TRANSFORMERS_VERBOSITY=error
export SGLANG_NPU_SPARSE_KV_INDEXER="$INDEXER_IMPL" SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD="$OFFLOAD"
export SGLANG_NPU_ENABLE_COPY_SFA="$COPY_SFA" SGLANG_NPU_SPARSE_KV_CACHE_SIZE=12288
export SGLANG_NPU_FUSED_LIM_PRE_METADATA="$PRE_METADATA_FUSION"
export SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA="$METADATA_FUSION"
export SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE="$NATIVE_MTP"
export TORCHINDUCTOR_CACHE_DIR="${TASK_ROOT:-${TMPDIR:-/tmp}/sglang-lim-mtp-a3}/cache/inductor"
export TRITON_CACHE_DIR="${TASK_ROOT:-${TMPDIR:-/tmp}/sglang-lim-mtp-a3}/cache/triton"
python3 - <<'PY'
import hashlib
import importlib.util
import os
from pathlib import Path

root = Path(os.environ["KERNEL_PYTHON_ROOT"]).resolve()
spec = importlib.util.find_spec("sgl_kernel_npu")
if spec is None or spec.origin is None:
    raise RuntimeError("KERNEL_PYTHON_ROOT must select the verified installed wheel")
package = Path(spec.origin).resolve()
if not package.is_relative_to(root):
    raise RuntimeError(f"Kernel package resolves outside requested root: {package}")
library = package.parent / "lib" / "libsgl_kernel_npu.so"
with library.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
expected_path = os.environ.get("EXPECTED_KERNEL_SO_PATH")
expected_hash = os.environ.get("EXPECTED_KERNEL_SO_SHA256")
if expected_path and library.resolve() != Path(expected_path).resolve():
    raise RuntimeError("Selected kernel library differs from controller provenance")
if expected_hash and checksum != expected_hash:
    raise RuntimeError("Selected kernel library SHA256 differs from controller provenance")
print(f"kernel_package={package} kernel_so={library} sha256={checksum}", flush=True)
PY
printf 'case=%s global_bs=%s offload=%s copy_sfa=%s C=%s K=2048 native_mtp=%s post_metadata=%s pre_metadata=%s visible_devices=%s\n' \
  "$CASE_ID" "$BATCH_SIZE" "$OFFLOAD" "$COPY_SFA" "$CACHE_DESCRIPTION" "$NATIVE_MTP" "$METADATA_FUSION" "$PRE_METADATA_FUSION" "${ASCEND_RT_VISIBLE_DEVICES:-unset}"
printf '%q ' "${args[@]}"
printf '\n'
exec "${args[@]}"
