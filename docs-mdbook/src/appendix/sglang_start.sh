#!/bin/bash
# SGLang 统一启动脚本 - 自动适配GPU数量
# 适配: L40S (sm_89), CUDA 12.1, SGLang 0.5.17, sglang-kernel 0.4.5
# RadixTree: prefix-aware DP Router + KV cache复用
#
# GPU数自动推断架构:
#   1卡 → TP=1 DP=1 (单卡验证)
#   2卡 → TP=2 DP=1 (TP验证)
#   4卡 → TP=2 DP=2 (DP验证)
#   6卡 → TP=2 DP=3 (生产)
#   7卡 → 多实例拆分（如 tool call 4卡 TP2 DP2 + 检视 2卡 TP2 DP1 + 轻量 1卡 TP1）
#
# 用法:
#   bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8
#   bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-35B-A3B-FP8
#   bash sglang_start.sh --model-path /path/to/model --port 8001 --gpu-ids 0,1
#
# 默认值即生产配置（无需额外参数）: MTP 开 / 代理 8080 / tool_call=16 / thinking=12 /
#   keep-alive 开 / 预热开 / priority 开 / round_robin / mem=0.85 / context=98304 / max-running=12
# 覆盖开关: --no-speculative / --no-proxy / --proxy-port 0 / --no-keep-alive / --skip-warmup
# 自适应限流: --adaptive-limit 开启（控制器自动调代理 limits，默认关）
# 注意: 多实例拆分（如 7 卡 4+2+1）时各实例必须用 --no-proxy 或不同 --proxy-port，
#        避免都占用默认 8080。

set -e
source ~/.bashrc 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/sglang_proxy.py"

# ==================== 参数 ====================
PORT=8000
MODEL_PATH=""
GPU_IDS=""
TP_SIZE=0
DP_SIZE=0
CONTEXT_LENGTH=0
MEM_FRACTION_STATIC=0
MAX_RUNNING_REQUESTS=0
CHUNKED_PREFILL_SIZE=4096
KV_CACHE_DTYPE=fp8_e5m2
MAMBA_RADIX_CACHE_STRATEGY=extra_buffer
MAMBA_BACKEND=triton
ENABLE_SPECULATIVE=false
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_EAGLE_TOPK=1
SPECULATIVE_NUM_DRAFT_TOKENS=4
KILL_EXISTING=false
SKIP_WARMUP=false
SCHEDULE_POLICY=lpm
KEEP_ALIVE=true
KEEP_ALIVE_INTERVAL=45
LOAD_BALANCE_METHOD=round_robin
ENABLE_PRIORITY=true
PROXY_PORT=8080
PROXY_TOOL_CALL_LIMIT=16
PROXY_THINKING_LIMIT=12
ADAPTIVE_LIMIT=false
ADAPTIVE_INTERVAL=15
ADAPTIVE_MIN_TOOL_CALL=4
ADAPTIVE_MAX_TOOL_CALL=24

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --dp-size) DP_SIZE="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2 ;;
        --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
        --enable-speculative) ENABLE_SPECULATIVE=true; shift ;;
        --no-speculative) ENABLE_SPECULATIVE=false; shift ;;
        --speculative-num-steps) SPECULATIVE_NUM_STEPS="$2"; shift 2 ;;
        --speculative-eagle-topk) SPECULATIVE_EAGLE_TOPK="$2"; shift 2 ;;
        --speculative-num-draft-tokens) SPECULATIVE_NUM_DRAFT_TOKENS="$2"; shift 2 ;;
        --kill-existing) KILL_EXISTING=true; shift ;;
        --warmup) SKIP_WARMUP=false; shift ;;
        --skip-warmup) SKIP_WARMUP=true; shift ;;
        --schedule-policy) SCHEDULE_POLICY="$2"; shift 2 ;;
        --keep-alive) KEEP_ALIVE=true; shift ;;
        --no-keep-alive) KEEP_ALIVE=false; shift ;;
        --keep-alive-interval) KEEP_ALIVE_INTERVAL="$2"; shift 2 ;;
        --load-balance-method) LOAD_BALANCE_METHOD="$2"; shift 2 ;;
        --priority-scheduling) ENABLE_PRIORITY=true; shift ;;
        --proxy-port) PROXY_PORT="$2"; shift 2 ;;
        --no-proxy) PROXY_PORT=0; shift ;;
        --proxy-tool-call-limit) PROXY_TOOL_CALL_LIMIT="$2"; shift 2 ;;
        --proxy-thinking-limit) PROXY_THINKING_LIMIT="$2"; shift 2 ;;
        --adaptive-limit) ADAPTIVE_LIMIT=true; shift ;;
        --adaptive-interval) ADAPTIVE_INTERVAL="$2"; shift 2 ;;
        --adaptive-min-tool-call) ADAPTIVE_MIN_TOOL_CALL="$2"; shift 2 ;;
        --adaptive-max-tool-call) ADAPTIVE_MAX_TOOL_CALL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "错误: --model-path 必传"
    echo "用法: bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8"
    exit 1
fi

# ==================== 自动推断参数 ====================
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
# K8s pod 内优先按 CUDA_VISIBLE_DEVICES 计数（nvidia-smi 可能列出宿主全部卡）
if [ -n "${CUDA_VISIBLE_DEVICES}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
# 显式指定 --gpu-ids 时以它为准（裸机多实例拆分场景）
if [ -n "$GPU_IDS" ]; then
    GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
fi

if [ "$TP_SIZE" -eq 0 ]; then
    if [ "$GPU_COUNT" -le 1 ]; then TP_SIZE=1
    else TP_SIZE=2
    fi
fi

if [ "$DP_SIZE" -eq 0 ]; then
    DP_SIZE=$(( GPU_COUNT / TP_SIZE ))
    [ "$DP_SIZE" -lt 1 ] && DP_SIZE=1
fi

# 奇数卡数 + TP=2 会闲置一张卡，提示多实例拆分
if [ $(( GPU_COUNT % 2 )) -eq 1 ] && [ "$TP_SIZE" -eq 2 ]; then
    echo "警告: 卡数 $GPU_COUNT 为奇数，TP=2 自动推断会闲置 1 卡；"
    echo "       多实例请分别用 --gpu-ids 指定卡组（如 tool call 4卡 + 检视 2卡 + 轻量 1卡）。"
fi

# 默认GPU列表
if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(seq -s, 0 $(( GPU_COUNT - 1 )))
fi

# 默认值按生产验证结论: mem=0.85 / context=98304 / max-running=12(per-worker)
if [ "$MEM_FRACTION_STATIC" = "0" ]; then
    MEM_FRACTION_STATIC=0.85
fi

if [ "$CONTEXT_LENGTH" -eq 0 ]; then
    CONTEXT_LENGTH=98304
fi

if [ "$MAX_RUNNING_REQUESTS" -eq 0 ]; then
    # max-running-requests是per-worker值，不是全局
    MAX_RUNNING_REQUESTS=12
fi

# 优先级调度（启用时把 schedule-policy 切到 priority；请求侧需带 priority 字段才生效）
PRIORITY_ARGS=""
if [ "$ENABLE_PRIORITY" = true ]; then
    SCHEDULE_POLICY=priority
    PRIORITY_ARGS="--enable-priority-scheduling --default-priority-value 0"
fi

SERVED_NAME=$(basename "$MODEL_PATH")

# ==================== 环境 ====================
export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export NO_COLOR=1
export TERM=dumb

if [ "$KILL_EXISTING" = true ] && pgrep -f "sglang.launch_server|sglang_proxy.py" > /dev/null 2>&1; then
    echo "终止残留进程（server + proxy）..."
    pkill -9 -f "sglang.launch_server|sglang_proxy.py|sglang_adaptive_limits.py" 2>/dev/null || true
    sleep 5
fi

echo "=========================================="
echo " SGLang 启动"
echo " GPU: $GPU_COUNT ($GPU_IDS)"
echo " TP=$TP_SIZE DP=$DP_SIZE"
echo " 模型: $SERVED_NAME"
echo " 上下文: $CONTEXT_LENGTH  mem: $MEM_FRACTION_STATIC"
echo " 并发: $MAX_RUNNING_REQUESTS/worker × $DP_SIZE = $(( MAX_RUNNING_REQUESTS * DP_SIZE ))总  端口: $PORT"
echo " RadixTree: enabled"
echo " Speculative(MTP): $ENABLE_SPECULATIVE"
echo " schedule-policy: $SCHEDULE_POLICY  预热: $([ "$SKIP_WARMUP" = true ] && echo skip || echo on)"
echo " keep-alive: $([ "$KEEP_ALIVE" = true ] && echo "on 每${KEEP_ALIVE_INTERVAL}s" || echo off)"
echo " load-balance: $LOAD_BALANCE_METHOD (DP 路由; round_robin 避免前缀粘滞热点)"
echo " priority-scheduling: $([ "$ENABLE_PRIORITY" = true ] && echo on || echo off)"
echo " proxy: $([ "$PROXY_PORT" != "0" ] && echo "on 端口$PROXY_PORT (限并发+priority)" || echo off)"
echo " proxy-limits: tool_call=$PROXY_TOOL_CALL_LIMIT thinking=$PROXY_THINKING_LIMIT (运行时可调: POST /admin/limits)"
echo " adaptive-limit: $([ "$ADAPTIVE_LIMIT" = true ] && echo "on (自动调 tool_call ${ADAPTIVE_MIN_TOOL_CALL}~${ADAPTIVE_MAX_TOOL_CALL}, 每${ADAPTIVE_INTERVAL}s)" || echo off)"
echo "=========================================="

# ==================== 常驻 keep-alive ====================
# 服务 ready 后每 KEEP_ALIVE_INTERVAL 秒发一个轻量请求（短 prompt、非 thinking、max_tokens 4）
# 作用: 覆盖懒加载/空闲后首请求开销，兼做健康检查；不能替代 10.7 方案一的启动前预热
if [ "$KEEP_ALIVE" = true ]; then
    (
        for i in $(seq 1 120); do
            curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
            sleep 5
        done
        while true; do
            curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":4}" \
                || echo "[keep-alive] 请求失败，server 可能未就绪"
            sleep "$KEEP_ALIVE_INTERVAL"
        done
    ) &
    KEEP_ALIVE_PID=$!
fi

# ==================== 前置代理（限并发 + 注入 priority） ====================
# 客户端连代理端口，代理转发到本机 SGLang；仅启动脚本层改动即可生效
if [ "$PROXY_PORT" != "0" ]; then
    (
        for i in $(seq 1 120); do
            curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
            sleep 5
        done
        exec python3.12 "$PROXY_SCRIPT" --backend "http://127.0.0.1:$PORT" --listen "$PROXY_PORT" \
            --tool-call-limit "$PROXY_TOOL_CALL_LIMIT" --thinking-limit "$PROXY_THINKING_LIMIT"
    ) &
    PROXY_PID=$!
    # 代理自检：等它就绪，失败则告警（不阻塞主服务）
    (
        for i in $(seq 1 60); do
            curl -sf -o /dev/null "http://127.0.0.1:$PROXY_PORT/health" && { echo "[proxy] 就绪: 端口 $PROXY_PORT"; exit 0; }
            sleep 2
        done
        echo "[proxy] 警告: 120s 内未就绪，检查端口 $PROXY_PORT 是否被占用"
    ) &
fi

# ==================== 自适应限流控制器 ====================
# 定时读后端 /metrics（num_queue_reqs/token_usage），自动调代理 /admin/limits
ADAPTIVE_PID=""
if [ "$ADAPTIVE_LIMIT" = true ] && [ "$PROXY_PORT" != "0" ]; then
    (
        # 等代理就绪再启动控制器（最多 2 分钟）
        for i in $(seq 1 60); do
            curl -sf -o /dev/null "http://127.0.0.1:$PROXY_PORT/health" && break
            sleep 2
        done
        python3.12 "$SCRIPT_DIR/sglang_adaptive_limits.py" \
            --backend "http://127.0.0.1:$PORT" \
            --proxy "http://127.0.0.1:$PROXY_PORT" \
            --tool-call "$PROXY_TOOL_CALL_LIMIT" \
            --thinking "$PROXY_THINKING_LIMIT" \
            --min-tool-call "$ADAPTIVE_MIN_TOOL_CALL" \
            --max-tool-call "$ADAPTIVE_MAX_TOOL_CALL" \
            --interval "$ADAPTIVE_INTERVAL"
    ) &
    ADAPTIVE_PID=$!
fi

MANAGED_PIDS=""
[ -n "$KEEP_ALIVE_PID" ] && MANAGED_PIDS="$MANAGED_PIDS $KEEP_ALIVE_PID"
[ -n "$PROXY_PID" ] && MANAGED_PIDS="$MANAGED_PIDS $PROXY_PID"
[ -n "$ADAPTIVE_PID" ] && MANAGED_PIDS="$MANAGED_PIDS $ADAPTIVE_PID"
trap 'kill $MANAGED_PIDS 2>/dev/null || true' EXIT

# K8s pod 内 GPU_IDS 为空时沿用环境已有的 CUDA_VISIBLE_DEVICES（否则空值会禁用全部 GPU）
CUDA_VISIBLE_DEVICES=${GPU_IDS:-$CUDA_VISIBLE_DEVICES} python3.12 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --tp-size $TP_SIZE \
    --dp-size $DP_SIZE \
    --load-balance-method $LOAD_BALANCE_METHOD \
    --mem-fraction-static $MEM_FRACTION_STATIC \
    --context-length $CONTEXT_LENGTH \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    $PRIORITY_ARGS \
    --mamba-radix-cache-strategy $MAMBA_RADIX_CACHE_STRATEGY \
    --mamba-backend $MAMBA_BACKEND \
    --enable-flashinfer \
    --attention-backend flashinfer \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-cuda-graph \
    --enable-cache-report \
    --schedule-policy $SCHEDULE_POLICY \
    $([ "$SKIP_WARMUP" = true ] && echo "--skip-server-warmup") \
    --enable-metrics \
    --log-level info \
    $([ "$ENABLE_SPECULATIVE" = true ] && echo "--speculative-algorithm NEXTN --speculative-num-steps $SPECULATIVE_NUM_STEPS --speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK --speculative-num-draft-tokens $SPECULATIVE_NUM_DRAFT_TOKENS")
