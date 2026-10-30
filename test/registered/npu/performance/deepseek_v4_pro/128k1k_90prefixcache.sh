#!/usr/bin/env bash
set -e
set -o pipefail

# DeepSeek-V4-Pro-0813-w4a8 / Ascend 910C / 2 nodes x 16 NPUs
# TP32 + DP8 Attention + DeepEP + DSpark
# Target: 128k / 1k / ~90% cache / client C32; server max requests 32.
# Run this same file on both nodes. Node rank is detected from the local IP.

MODEL_PATH=/home/weights/DeepSeek-V4-Pro-0813-w4a8
SGLANG_PYTHON=/home/sglang-dspark/python
SERVED_MODEL_NAME=dsv4
SERVER_PORT=6677
MASTER_PORT=5567

NODE_IPS=(80.5.17.39 80.5.17.40)
HCCL_IFS=(enp194s0f0 enp194s0f0)
NUM_NPUS_PER_NODE=16
DP_SIZE=8

# Mixed Prefill/Decode starting point. Try 8192 if TPOT exceeds 20 ms;
# try 32768 if TTFT is too high and TPOT has headroom.
CHUNKED_PREFILL_SIZE=16384

# Mainly controls weight and KV allocation, not raw compute speed.
# If startup or graph capture OOMs, try 0.78, then 0.76 on both nodes.
MEM_FRACTION_STATIC=0.80

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

[[ -d ""$MODEL_PATH"" ]] || {
    echo ""Missing model path: $MODEL_PATH"" >&2
    exit 1
}

[[ -d ""$SGLANG_PYTHON/sglang"" ]] || {
    echo ""Missing SGLang checkout: $SGLANG_PYTHON/sglang"" >&2
    exit 1
}

[[ ""$CHUNKED_PREFILL_SIZE"" =~ ^[0-9]+$ ]] &&
    (( CHUNKED_PREFILL_SIZE > 0 )) || {
    echo ""Invalid CHUNKED_PREFILL_SIZE: $CHUNKED_PREFILL_SIZE"" >&2
    exit 1
}

NODE_RANK=
SERVER_HOST=
read -r -a LOCAL_IPS <<< ""$(hostname -I)""

for i in ""${!NODE_IPS[@]}""; do
    for ip in ""${LOCAL_IPS[@]}""; do
        if [[ ""$ip"" == ""${NODE_IPS[$i]}"" ]]; then
            NODE_RANK=$i
            SERVER_HOST=$ip
            export HCCL_SOCKET_IFNAME=""${HCCL_IFS[$i]}""
            export GLOO_SOCKET_IFNAME=""${HCCL_IFS[$i]}""
            break 2
        fi
    done
done

[[ -n ""$NODE_RANK"" ]] || {
    echo ""Cannot detect node rank; local IPs: ${LOCAL_IPS[*]}"" >&2
    exit 1
}

NNODES=${#NODE_IPS[@]}
TP_SIZE=$((NNODES * NUM_NPUS_PER_NODE))
DIST_INIT_ADDR=""${NODE_IPS[0]}:${MASTER_PORT}""

# Ascend environment
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTHONPATH=""${SGLANG_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}""

# Read the native DSpark block size from the checkpoint.
# For the confirmed checkpoint, dspark_block_size=5, so verify window=6.
# Override DSPARK_BLOCK_SIZE on both nodes only for a measured comparison.
CHECKPOINT_DSPARK_BLOCK_SIZE=""$(python3 - ""$MODEL_PATH/config.json"" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
value = cfg.get(""dspark_block_size"")
if value is None:
    value = (cfg.get(""dflash_config"") or {}).get(""block_size"")
if value is None:
    value = cfg.get(""block_size"")
if value is None:
    print(
        ""DSpark block size missing in config.json; using upstream 0813 default 5"",
        file=sys.stderr,
    )
    value = 5
if int(value) < 1:
    raise SystemExit(f""Invalid checkpoint DSpark block size: {value}"")
print(int(value))
PY
)""

DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-$CHECKPOINT_DSPARK_BLOCK_SIZE}
[[ ""$DSPARK_BLOCK_SIZE"" =~ ^[0-9]+$ ]] && (( DSPARK_BLOCK_SIZE > 0 )) || {
    echo ""Invalid DSPARK_BLOCK_SIZE: $DSPARK_BLOCK_SIZE"" >&2
    exit 1
}
DSPARK_VERIFY_TOKENS=$((DSPARK_BLOCK_SIZE + 1))

if (( DSPARK_BLOCK_SIZE > CHECKPOINT_DSPARK_BLOCK_SIZE )); then
    echo ""Warning: DSpark override $DSPARK_BLOCK_SIZE exceeds checkpoint-native $CHECKPOINT_DSPARK_BLOCK_SIZE"" >&2
fi

# CPU and communication environment
echo performance |
    tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null || true
sysctl -w vm.swappiness=0 >/dev/null || true
sysctl -w kernel.numa_balancing=0 >/dev/null || true

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60
export DEEPEP_HCCL_BUFFSIZE=2000

# NPU memory and MoE streams
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_NPU_USE_MULTI_STREAM=1

# Checkpoint Experts are ModelSlim W4A8_DYNAMIC, not MXFP4.
export SGLANG_DSV4_FP4_EXPERTS=False

# Preserve the Ascend/ModelSlim implementation choices from the running setup.
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

# DeepEP
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_HYBRID_DEPLOYMENT=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

# DSpark on Ascend
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_RAGGED_VERIFY_MODE=static

# Distinct from SGLANG_NPU_USE_MULTI_STREAM above.
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_DSPARK_FAST_SAMPLING=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=0

export SGLANG_DSPARK_QUANT_AUDIT=1
export SGLANG_DSPARK_QUANT_AUDIT_STRICT=0
export TRANSFORMERS_VERBOSITY=error

echo ""==============================================""
echo ""DeepSeek-V4-Pro PR40399 / Ascend 910C""
echo ""node-rank: $NODE_RANK / $NNODES; local IP: $SERVER_HOST""
echo ""dist-init: $DIST_INIT_ADDR; TP: $TP_SIZE; DP: $DP_SIZE""
echo ""SGLang: $(python3 -c 'import sglang; print(sglang.__file__)')""
echo ""static mem: $MEM_FRACTION_STATIC; prefill chunk: $CHUNKED_PREFILL_SIZE""
echo ""DSpark: block $DSPARK_BLOCK_SIZE; verify $DSPARK_VERIFY_TOKENS""
echo ""==============================================""

# Avoid killing unrelated Python processes on a shared node.
if command -v ss >/dev/null 2>&1 &&
    ss -H -ltn ""( sport = :$SERVER_PORT )"" | grep -q .; then
    echo ""Port $SERVER_PORT is occupied on $SERVER_HOST."" >&2
    echo ""Stop your old service before launching this script."" >&2
    exit 1
fi

# Clear settings inherited from earlier experiments.
unset SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES
unset SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE
unset DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ
unset DEEPEP_NORMAL_LONG_SEQ_ROUND
unset DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS

exec python3 -m sglang.launch_server \
    --model-path ""$MODEL_PATH"" \
    --served-model-name ""$SERVED_MODEL_NAME"" \
    --host ""$SERVER_HOST"" \
    --port ""$SERVER_PORT"" \
    --nnodes ""$NNODES"" \
    --node-rank ""$NODE_RANK"" \
    --dist-init-addr ""$DIST_INIT_ADDR"" \
    --tp-size ""$TP_SIZE"" \
    --dp ""$DP_SIZE"" \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --watchdog-timeout 9000 \
    --max-running-requests 32 \
    --mem-fraction-static ""$MEM_FRACTION_STATIC"" \
    --quantization modelslim \
    --chunked-prefill-size ""$CHUNKED_PREFILL_SIZE"" \
    --prefill-decode-interval 2 \
    --enable-prefill-delayer \
    --kv-cache-dtype auto \
    --moe-dense-tp-size 1 \
    --cuda-graph-bs-decode 1 2 4 6 8 10 12 14 16 \
    --load-balance-method round_robin \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-metrics \
    --enable-cache-report \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path ""$MODEL_PATH"" \
    --speculative-draft-model-quantization modelslim \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens ""$DSPARK_VERIFY_TOKENS"" \
    --speculative-dspark-block-size ""$DSPARK_BLOCK_SIZE""