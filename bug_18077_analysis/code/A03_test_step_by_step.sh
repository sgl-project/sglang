#!/bin/bash
# A03_test_step_by_step: 分层测试，定位 GLM-Image 500 错误到底出在哪一阶段
# 用法: 先运行 A01_single_gpu.sh 启动服务，再在另一终端执行 ./A03_test_step_by_step.sh [--port 30000]
# 服务端 stdout 会打出 DenoisingStage (GLM/SpatialImage): latent_model_input.shape=... 若请求能走到 denoising 阶段

set -e

PORT="30000"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done
BASE_URL="http://127.0.0.1:${PORT}"
OUT_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/A03_step_by_step"
mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUT_DIR}/run_${TIMESTAMP}.log"
MODEL="zai-org/GLM-Image"

cd /data/users/yandache/workspaces/sglang
[ -f "env_sglang/bin/activate" ] && source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== A03 分层测试 $TIMESTAMP ==="
echo "BASE_URL=$BASE_URL"
echo "日志文件: $LOG_FILE"
echo ""

# Step 1: 健康检查 + 模型信息
echo "--- Step 1: 健康检查与 pipeline 路径 ---"
if ! curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" | grep -q 200; then
    echo "失败: $BASE_URL/health 未返回 200。请先运行 A01_single_gpu.sh 启动服务。"
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

# Step 2: 单次图像生成请求，抓完整响应与错误
echo "--- Step 2: 单次 /v1/images/generations 请求 ---"
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
    echo ">>> 若为 500: 请查看【启动 A01 的终端】里的完整 traceback。"
    echo ">>> 在服务端 stdout 中搜索 '[A03_GLM]'，按出现顺序可判断请求走到了哪一层："
    echo "     GlmImageBeforeDenoisingStage START -> after prepare_latents -> END -> DenoisingStage START -> _preprocess_sp_latents -> _predict_noise_with_cfg -> GlmImageTransformer2DModel.forward"
    echo "    最后一条 [A03_GLM] 之后的 traceback 即出错位置。"
else
    echo "请求成功。响应已写入 $RESP_FILE"
fi
echo ""

# Step 3: 提醒后续可加的更细粒度测试
echo "--- Step 3: 后续可分层加测 ---"
echo "可在本脚本中继续加："
echo "  - 仅验证 GlmImageBeforeDenoisingStage 出口的 batch.latents shape（需在 stage 内打日志）"
echo "  - 验证 _preprocess_sp_latents 前后 shape（DenoisingStage 内）"
echo "  - 验证 transformer.forward 入口 hidden_states.shape（glm_image.py DiT）"
echo ""
echo "本次完整日志: $LOG_FILE"
