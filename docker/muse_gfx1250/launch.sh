#!/bin/bash
# Container: a ROCm SGLang dev image for gfx1250 with an aiter build that supports gfx1250, pip flydsl==0.4.2.
# W = work dir holding this repo (sglang/), hf/ (HF cache with the DFlash draft) and the model.
# usage: launch.sh TAG [ENV=V ...]; env W, MODEL, TPN, GPUS, PORT, PDI, SRI, EXTRA_ARGS
W=${W:-/work}; TAG=$1; shift; mkdir -p $W/logs
export PYTHONPATH=$W/sglang/python HF_HOME=$W/hf HF_HUB_OFFLINE=1
export SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 MUSE_MOE_BACKEND=triton
export NCCL_CUMEM_ENABLE=1 AITER_CUSTOM_AR_FORCE_VMM=1 AITER_CUSTOM_AR_MAX_SIZE=1048576 SGLANG_USE_AITER_AG=0 \
  SGLANG_ROCM_MUSE_FINALIZE_AR_PUSH=0 SGLANG_ROCM_MUSE_A4W4_MOE=0 SGLANG_ROCM_MUSE_COMPACT_A4_VERIFY=0 SGLANG_ROCM_MUSE_GFX1250_TRY_GFX95=1 \
  SGLANG_ROCM_MUSE_EXTEND_GLUON=0 SGLANG_ROCM_MUSE_EXTEND_SWA_GLUON=0 SGLANG_ROCM_MUSE_EXTEND_FLYDSL=0 SGLANG_ROCM_MUSE_VERIFY_FLYDSL=0 \
  SGLANG_ROCM_MUSE_VERIFY_BALANCED=0 SGLANG_ROCM_MUSE_VERIFY_SWA_FLYDSL=0 \
  HIP_FORCE_DEV_KERNARG=1 SGLANG_ROCM_MUSE_GFX1250_EXTEND=1 SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1 \
  SGLANG_OPT_SCHEDULER_SHM_SYNC=1 AITER_CONFIG_GROUPED_FMOE=$W/sglang/docker/muse_gfx1250/muse_gfx1250_tp8_ep8_tuned_grouped_fmoe.csv
[ -n "$GPUS" ] && export HIP_VISIBLE_DEVICES=$GPUS
for kv in "$@"; do export "$kv"; done
env | grep -E "^(SGLANG|MUSE_|NCCL|AITER|HIP_VIS)" | sort > $W/logs/$TAG.env
cd /tmp
nohup python3 -m sglang.launch_server --model-path ${MODEL:-$W/Muse-Spark-Open-816B-A42B-Quark-MXFP4} \
  --tp-size ${TPN:-8} --ep-size ${TPN:-8} --attention-backend aiter --page-size 16 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static ${MEM:-0.95} --max-running-requests 128 --swa-full-tokens-ratio 0.2 \
  --chunked-prefill-size 16384 --cuda-graph-max-bs-prefill 8192 \
  --prefill-decode-interval ${PDI:-0} --scheduler-recv-interval ${SRI:-1} --stream-interval 20 --tokenizer-worker-num 4 \
  --speculative-algorithm DFLASH --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 \
  --speculative-draft-attention-backend triton --speculative-num-draft-tokens 8 --speculative-draft-window-size 2048 \
  --mm-attention-backend triton_attn --cuda-graph-backend-prefill breakable ${EXTRA_ARGS} \
  --reasoning-parser muse --tool-call-parser muse --watchdog-timeout 1800 \
  --host 0.0.0.0 --port ${PORT:-30200} > $W/logs/$TAG.log 2>&1 &
echo started $TAG
