#!/bin/bash
# Usage: TAG=x [GPUS=0,1,2,3 PORT=30000] [STEPS=3 DRAFT=4] [EXTRA2=..] [ENVS2=..] bash launch_v2.sh   (locations from env.sh)
source "$(dirname "$0")/env.sh"
: "${TAG:=v2_mxfp4_eagle3}" "${STEPS:=3}" "${DRAFT:=4}" "${MEMFRAC:=0.85}" "${CHUNK:=8192}"
export KV=fp8_e4m3 CHUNK MEMFRAC TAG
export EXTRA="--speculative-algorithm EAGLE3 --speculative-draft-model-path $DRAFT_MODEL --speculative-num-steps $STEPS --speculative-eagle-topk 1 --speculative-num-draft-tokens $DRAFT --speculative-attention-mode ${SPEC_ATTN:-decode} ${EXTRA2:-}"
export ENVS="SGLANG_M3_ALLOW_CUSTOM_AR=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_MINIMAX_OPT_USE_GLUON_PREFILL=1 ${ENVS2:-}"
exec bash "$M3_HERE/launch.sh"
