#!/bin/bash
# MiniMax-M3 external endpoint on 4x MI350X/MI355X: TP4, MXFP4 quark checkpoint, fp8 KV, speculative decoding.
# Usage: bash serve_endpoint.sh            (inside the M3 engine image; models under $MODEL_ROOT)
# Env:   MODEL_ROOT=/models   SPEC=eagle3|dspark|none   PORT=30000   GPUS=0,1,2,3   SGLANG_API_KEY=<bearer key, optional>
#        MAXRUN=48 (max running requests)  MAXQUEUE=64 (waiting requests before 429)  MEMFRAC=0.9  PTPC_FP8=1 (optional)  EXTRA="<more server flags>"
set -uo pipefail
: "${MODEL_ROOT:=/models}" "${SPEC:=eagle3}" "${PORT:=30000}" "${GPUS:=0,1,2,3}" "${MAXRUN:=48}" "${MAXQUEUE:=64}" "${MEMFRAC:=0.9}" "${EXTRA:=}"
: "${MODEL:=$MODEL_ROOT/MiniMax-M3-MXFP4}" "${EAGLE_DRAFT:=$MODEL_ROOT/MiniMax-M3-EAGLE3-GQA}" "${DSPARK_DRAFT:=$MODEL_ROOT/MiniMax-M3-DSpark}"
export HIP_VISIBLE_DEVICES=$GPUS CUDA_VISIBLE_DEVICES=$GPUS
# kernels and collectives (measured on gfx950; see ../OPTIMIZATIONS.md section 3)
export SGLANG_USE_AITER=1 NCCL_MIN_NCHANNELS=112 HIP_FORCE_DEV_KERNARG=1
export SGLANG_M3_ALLOW_CUSTOM_AR=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT4 SGLANG_CUSTOM_AR_ONE_STAGE_MAX_BYTES=262144
export SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL=1 SGLANG_MINIMAX_OPT_USE_GLUON_PREFILL=1 SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4
export SGLANG_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1
export SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5 SGLANG_TIMEOUT_KEEP_ALIVE=3600
# MiniMax provider-check behaviour: parse tool calls with no/unknown inventory, 404 unknown model names, 429 on a full queue
export SGLANG_FORWARD_UNKNOWN_TOOLS=1 SGLANG_ENABLE_STRICT_MODEL_NAME=1 SGLANG_ENABLE_QUEUE_FULL_429=1
if [ -n "${PTPC_FP8:-}" ]; then  # quark-excluded dense layers as online per-token FP8 (quality-neutral; speed measured per build, see OPTIMIZATIONS.md section 4)
  HERE=$(cd "$(dirname "$0")" && pwd); aiter_pkg=$(python3 -c "import os, aiter; print(os.path.dirname(aiter.__file__))" 2>/dev/null)
  export SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED=1 SGLANG_USE_AITER_FP8_PER_TOKEN=1 SGLANG_QUARK_ONLINE_FP8_SKIP_MODULES=gate,lm_head,index_qkv_proj SGLANG_FUSED_NORM_FP8_QUANT_MAX_M=16384
  export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=$aiter_pkg/configs/a8w8_bpreshuffle_tuned_gemm.csv:$HERE/../tuned_a8w8_bpreshuffle_m3_gfx950.csv
fi
case $SPEC in
  eagle3) SPECARGS="--speculative-algorithm EAGLE3 --speculative-draft-model-path $EAGLE_DRAFT --speculative-num-steps ${STEPS:-3} --speculative-eagle-topk 1 --speculative-num-draft-tokens ${DRAFT:-4} --speculative-attention-mode decode" ;;
  dspark) SPECARGS="--speculative-algorithm DSPARK --speculative-draft-model-path $DSPARK_DRAFT ${DSPARK_BLOCK:+--speculative-dspark-block-size $DSPARK_BLOCK}" ;;
  none) SPECARGS="" ;;
  *) echo "SPEC must be eagle3|dspark|none"; exit 2 ;;
esac
exec python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name MiniMax-M3 --trust-remote-code \
  --tp 4 --host 0.0.0.0 --port "$PORT" ${SGLANG_API_KEY:+--api-key "$SGLANG_API_KEY"} \
  --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 8192 --mem-fraction-static "$MEMFRAC" \
  $SPECARGS --triton-attention-num-kv-splits 64 --cuda-graph-backend-prefill breakable \
  --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 --enable-metrics --enable-cache-report \
  --max-running-requests "$MAXRUN" --max-queued-requests "$MAXQUEUE" --watchdog-timeout 3600 $EXTRA
