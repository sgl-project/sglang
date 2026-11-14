#!/usr/bin/env bash
# 03_launch.sh — launch SGLang with DeepSeek-V4-Flash on the detected Ascend slice (single node, PD-mixed).
#
# Flags and env vars are taken from the validated configs in
# docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx (PD-mixed,
# half A3 node = 8 dies) and re-scaled to the detected number of dies. Every deviation is labelled.
#
# Usage: bash deploy/03_launch.sh [--foreground] [--dry-run]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$HERE/.." && pwd)

FOREGROUND=0; DRY=0
for a in "$@"; do
  [ "$a" = "--foreground" ] && FOREGROUND=1
  [ "$a" = "--dry-run" ] && DRY=1
done
[ "$DRY" -eq 1 ] && export FORCE_FEASIBLE=1

# shellcheck source=../env/00_common.sh
source "$PROJ/env/00_common.sh"

SERVER_LOG="$LOG_DIR/server.log"
: > "$SERVER_LOG" 2>/dev/null || true

# ---------------- CANN environment (required in every fresh shell) ----------
CANN_ENVS=(
  /usr/local/Ascend/ascend-toolkit/set_env.sh
  /usr/local/Ascend/nnal/atb/set_env.sh
  /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
  /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
)
for e in "${CANN_ENVS[@]}"; do
  [ -f "$e" ] && source "$e" || echo "WARN: missing $e (CANN incomplete?)" >&2
done

# ---------------- runtime env (from the validated tutorial) ----------------
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export FORCE_DRAFT_MODEL_NON_QUANT=1
export HCCL_OP_EXPANSION_MODE=AIV
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export ASCEND_USE_FIA=1                 # required on 950PR/DT series
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

# Single-device-slice: no cross-node links, keep HCCL on local loopback semantics.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}

# ZBAL is part of the validated 8-die config; on a smaller slice it is optional.
# Enable only after a baseline works (then scale SGLANG_ZBAL_LOCAL_MEM_SIZE with die count).
if [ "${ENABLE_ZBAL:-0}" = "1" ]; then
  export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:14699"
  export SGLANG_ZBAL_LOCAL_MEM_SIZE=${SGLANG_ZBAL_LOCAL_MEM_SIZE:-$((62084 * NPUS / 8))}
  export ZBAL_ENABLE_GRAPH=1
  export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
fi

# ---------------- sizing: documented 8-die values, scaled by die count -----
MEM_FRACTION=${MEM_FRACTION:-0.75}          # docs: 0.8 on 8 dies; 0.75 is the 4-die starting point
PREFILL_MAX_REQ=${PREFILL_MAX_REQ:-$((NPUS * 20))}     # docs: 160 on 8 dies
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-$((NPUS * 10000))}  # docs: 80000
CHUNKED=${CHUNKED:-$((NPUS * 16384))}                 # docs: 131072
MAX_RUNNING=${MAX_RUNNING:-$((NPUS * 20))}            # docs: 160 on 8 dies
CONTEXT_LEN=${CONTEXT_LEN:-131072}                    # model ceiling is 1048576; cap to keep KV affordable
GRAPH_BS=${GRAPH_BS:-"1 2 4 8 16"}

CMD=(python3 -m sglang.launch_server
  --model-path "$MODEL_DIR"
  --host 127.0.0.1 --port "$PORT"
  --trust-remote-code
  --device npu
  --attention-backend dsv4
  --page-size 128
  --tp-size "$TP_SIZE"
  --dp-size "$DP_SIZE"
  --enable-dp-attention
  --enable-dp-lm-head
  --moe-a2a-backend deepep
  --deepep-mode auto
  --quantization modelslim
  --kv-cache-dtype bfloat16
  --mem-fraction-static "$MEM_FRACTION"
  --prefill-max-requests "$PREFILL_MAX_REQ"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --chunked-prefill-size "$CHUNKED"
  --max-running-requests "$MAX_RUNNING"
  --cuda-graph-bs-decode $GRAPH_BS
  --speculative-algorithm EAGLE
  --speculative-num-steps 2
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 3
  --context-length "$CONTEXT_LEN"
  --enable-metrics
  --watchdog-timeout 9000
)
# Optional: reasoning-output parsing for chat mode. The parser name registered in this SGLang build is
# "deepseek-v4"; enable only after a baseline launch works, since an unknown parser name fails startup.
if [ "${ENABLE_REASONING_PARSER:-0}" = "1" ]; then
  CMD+=(--reasoning-parser deepseek-v4)
fi

{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)  chip=$CHIP npus=$NPUS hbm=${HBM_TOTAL_GB}GB image=$IMAGE"
  echo "# command:"
  printf '#   %q' "${CMD[0]}"; printf ' %q' "${CMD[@]:1}"; echo
  echo "# env overrides: MEM_FRACTION=$MEM_FRACTION ZBAL=${ENABLE_ZBAL:-0} GLOO=$GLOO_SOCKET_IFNAME HCCL=$HCCL_SOCKET_IFNAME"
} >> "$LOG_DIR/launch-command.txt"

if [ "$DRY" -eq 1 ]; then
  echo "dry run — command that would start:"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

# ---------------- preflight ------------------------------------------------
python3 - <<'PY' || { echo "preflight failed: torch_npu / device not usable"; exit 10; }
import sys
try:
    import torch, torch_npu
except Exception as e:
    print("import torch_npu failed:", e); sys.exit(1)
print("torch", torch.__version__, "| npu available:", torch.npu.is_available(), "| devices:", torch.npu.device_count())
sys.exit(0 if torch.npu.is_available() else 1)
PY
[ -f "$MODEL_DIR/config.json" ] || { echo "no config.json in $MODEL_DIR — run deploy/02_weights.sh first"; exit 11; }

# ---------------- start ---------------------------------------------------
echo "starting server; log: $SERVER_LOG"
if [ "$FOREGROUND" -eq 1 ]; then
  exec "${CMD[@]}" 2>&1 | tee -a "$SERVER_LOG"
fi

nohup "${CMD[@]}" >> "$SERVER_LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/server.pid"
echo "pid $PID"

echo -n "waiting for readiness (up to 45 min: 300 GB checkpoint load is slow on first start) "
for i in $(seq 1 2700); do
  if grep -q "The server is fired up and ready to roll" "$SERVER_LOG" 2>/dev/null; then
    echo; echo "READY after $((i * 5))s"
    curl -s -o /dev/null -w "health: %{http_code}\n" "http://127.0.0.1:$PORT/health" || true
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo; echo "server process died — last 60 lines:"
    tail -n 60 "$SERVER_LOG"
    exit 12
  fi
  [ $((i % 12)) -eq 0 ] && echo -n "."
  sleep 5
done
echo; echo "timed out waiting for readiness — last 80 lines:"; tail -n 80 "$SERVER_LOG"; exit 13
