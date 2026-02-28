#!/bin/bash
# A03_multiple_gpu: 多卡 SP=2 分层测试，与 A03_test_step_by_step 同结构，用于确认 SP 下每层 shape
# 用法: 先运行 A01_multiple_gpu.sh 启动 2 卡服务，再在另一终端执行 ./A03_multiple_gpu.sh [--port 30000]
# 服务端 stdout 搜 [A03_GLM]：多卡下每 rank 会打日志，期望 hidden_states.shape=(1,16,32,64)（沿 H 切分）

set -e

PORT="30000"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done
BASE_URL="http://127.0.0.1:${PORT}"
OUT_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/A03_multiple_gpu"
mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUT_DIR}/run_${TIMESTAMP}.log"
MODEL="zai-org/GLM-Image"

cd /data/users/yandache/workspaces/sglang
[ -f "env_sglang/bin/activate" ] && source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== A03 多卡 SP=2 分层测试 $TIMESTAMP ==="
echo "BASE_URL=$BASE_URL"
echo "日志文件: $LOG_FILE"
echo ""

# Step 1: 健康检查 + 模型信息
echo "--- Step 1: 健康检查与 pipeline 路径 ---"
if ! curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" | grep -q 200; then
    echo "失败: $BASE_URL/health 未返回 200。请先运行 A01_multiple_gpu.sh 启动服务。"
    exit 1
fi
echo "health: OK"

MODEL_INFO=$(curl -s "$BASE_URL/v1/models" 2>/dev/null || curl -s "$BASE_URL/models" 2>/dev/null || true)
if echo "$MODEL_INFO" | grep -q "GlmImagePipeline"; then
    echo "pipeline_class: GlmImagePipeline (GlmImagePipelineConfig / SpatialImagePipelineConfig) OK"
else
    echo "pipeline 信息: $MODEL_INFO"
fi
echo ""

# Step 2: 单次图像生成请求
echo "--- Step 2: 单次 /v1/images/generations 请求 (512x512) ---"
RESP_FILE="${OUT_DIR}/response_step2_${TIMESTAMP}.json"
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$RESP_FILE" \
    -X POST "$BASE_URL/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"a cat\",
        \"n\": 1,
        \"size\": \"512x512\",
        \"response_format\": \"b64_json\"
    }")
echo "HTTP status: $HTTP_CODE"
if [[ "$HTTP_CODE" != "200" ]]; then
    echo "响应内容 (前 2000 字符):"
    head -c 2000 "$RESP_FILE" | cat
    echo ""
    echo ""
    echo ">>> 若为 500: 请查看【启动 A01_multiple_gpu 的终端】里的完整 traceback。"
    echo ">>> 在服务端 stdout 中搜索 '[A03_GLM]'："
    echo "    - 多卡下每个 rank 都会打日志，同一时刻可能有两段（rank0/rank1）。"
    echo "    - SP=2 时期望 per-rank: hidden_states.shape=(1, 16, 32, 64)（沿 H 切分）；"
    echo "    - 若仍是 (1,16,64,64) 说明未切分；若报 rearrange 等错说明走了 3D 分支。"
    echo "    最后一条 [A03_GLM] 之后的 traceback 即出错位置。"
else
    echo "请求成功。响应已写入 $RESP_FILE"
fi
echo ""

echo "--- 多卡检查点（参考 A02_multiple_gpu.md）---"
echo "在服务端日志中确认："
echo "  - _preprocess_sp_latents AFTER: batch.latents.shape 每 rank 应为 (1, 16, 32, 64)"
echo "  - GlmImageTransformer2DModel.forward: hidden_states.shape=(1, 16, 32, 64) dim=4"
echo ""
echo "本次完整日志: $LOG_FILE"
