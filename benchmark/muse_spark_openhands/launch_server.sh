#!/usr/bin/env bash
# The two server commands of the sweep. Runs the server in the foreground; run_sweep.sh starts one per rung.
#
#   bash launch_server.sh hicache <cc>   # cc >= 16: HiCache 200 GB per rank
#   bash launch_server.sh plain <cc>     # cc <= 8: no HiCache
#
# <cc> is the rung's concurrency: --max-running-requests and --cuda-graph-max-bs-decode track it. Everything else is
# fixed: TP8/EP8 NVFP4 with MegaMoE + LayerNorm-SP, DFlash2 draft with 8-token draft block and forced acceptance 4.5,
# four tokenizer workers behind Granian (--enable-http2).
set -euo pipefail

MODE=${1:?usage: launch_server.sh hicache|plain <cc>}
MAX_RUNNING_REQUESTS=${2:?usage: launch_server.sh hicache|plain <cc>}
MODEL_PATH=${MODEL_PATH:-meta-models/Muse-Spark-Open-816B-A42B-NVFP4}
DRAFT_PATH=${DRAFT_PATH:-RadixArk/Muse-Spark-Open-DFlash2}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-meta-models/Muse-Spark-Open-816B-A42B-NVFP4}
SERVER_PYTHON=${SERVER_PYTHON:-python3}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30100}
HICACHE_GB=${HICACHE_GB:-200}
# Appended verbatim to the command (e.g. --skip-server-warmup for trees whose multimodal startup warmup fails).
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}

# Forced DFlash acceptance: the draft model is loaded and run, the verifier accepts 4 or 5 tokens per step so the mean
# is exactly 4.5, and decode cost matches that acceptance length. Token mode real-draft-token emits the draft's own
# proposals plus a target-derived bonus token; neither mode measures generation quality.
export SGLANG_SIMULATE_ACC_LEN=4.5 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=${SIM_ACC_TOKEN_MODE:-real-draft-token}
# Read by the uvicorn front end only; exported so the environment matches the recorded run.
export SGLANG_TIMEOUT_KEEP_ALIVE=600
# Preserve /proc environment visibility across process-title updates for the startup audit.
export SPT_NOENV=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8192
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1 TZ=UTC
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [ -d "$MODEL_PATH" ]; then export HF_HUB_OFFLINE=1; fi
if [ -n "${CACHE_DIR:-}" ]; then
    # JIT kernel caches survive container restarts when CACHE_DIR sits on persistent storage.
    export TMPDIR="${TMPDIR:-/tmp}" XDG_CACHE_HOME="$CACHE_DIR" TRITON_CACHE_DIR="$CACHE_DIR/triton" \
        TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor" CUDA_CACHE_PATH="$CACHE_DIR/cuda" \
        SGLANG_CACHE_DIR="$CACHE_DIR/sglang" FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/sglang"
    mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$SGLANG_CACHE_DIR"
fi

set -x
case "$MODE" in
hicache)
    exec "$SERVER_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
        --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm \
        --tp-size 8 --ep-size 8 --attention-backend trtllm_mha --page-size 128 --mem-fraction-static 0.9 \
        --reasoning-parser muse --tool-call-parser muse \
        --enable-layernorm-sp --enable-megamoe-for-layernorm-sp \
        --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT_PATH" \
        --speculative-draft-attention-backend fa4 --speculative-num-draft-tokens 8 --sampling-defaults model --kv-cache-dtype fp8_e4m3 \
        --enable-hierarchical-cache --hicache-size "$HICACHE_GB" \
        --max-running-requests "$MAX_RUNNING_REQUESTS" --cuda-graph-max-bs-decode "$MAX_RUNNING_REQUESTS" \
        --context-length 131072 --enable-cache-report \
        --tokenizer-worker-num 4 --enable-http2 --host "$HOST" --port "$PORT" $EXTRA_SERVER_ARGS
    ;;
plain)
    exec "$SERVER_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
        --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm \
        --tp-size 8 --ep-size 8 --attention-backend trtllm_mha --page-size 128 --mem-fraction-static 0.9 \
        --reasoning-parser muse --tool-call-parser muse \
        --enable-layernorm-sp --enable-megamoe-for-layernorm-sp \
        --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT_PATH" \
        --speculative-draft-attention-backend fa4 --speculative-num-draft-tokens 8 --sampling-defaults model --kv-cache-dtype fp8_e4m3 \
        --max-running-requests "$MAX_RUNNING_REQUESTS" --cuda-graph-max-bs-decode "$MAX_RUNNING_REQUESTS" \
        --context-length 131072 --enable-cache-report \
        --tokenizer-worker-num 4 --enable-http2 --host "$HOST" --port "$PORT" $EXTRA_SERVER_ARGS
    ;;
*)
    echo "unknown mode '$MODE' (hicache|plain)" >&2
    exit 2
    ;;
esac
