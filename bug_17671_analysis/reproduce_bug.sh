#!/bin/bash
# Issue #17671 复现脚本
# 用于验证 SGLang Docker 镜像缺少 diffusion 依赖

set -e

echo "=========================================="
echo "Issue #17671 复现脚本"
echo "验证 lmsysorg/sglang:dev 镜像缺少 diffusion 依赖"
echo "=========================================="
echo ""

# Step 0: 拉取最新 dev 镜像
echo "Step 0: 拉取最新 dev 镜像..."
docker pull lmsysorg/sglang:dev
echo "✓ 镜像拉取完成"
echo ""

# Step 1: 验证 diffusers 模块缺失
echo "Step 1: 验证 diffusers 模块缺失..."
echo "执行: docker run --rm -it lmsysorg/sglang:dev python -c \"import diffusers\""
echo ""

if docker run --rm -it lmsysorg/sglang:dev python -c "import diffusers" 2>&1; then
    echo "❌ 意外：diffusers 模块存在！"
    exit 1
else
    echo "✓ 确认：diffusers 模块缺失（ModuleNotFoundError）"
fi
echo ""

# Step 2: 检查镜像信息
echo "Step 2: 检查镜像信息..."
echo "执行: docker image inspect lmsysorg/sglang:dev | head -n 40"
docker image inspect lmsysorg/sglang:dev | head -n 40
echo ""

# Step 3: 使用 tiny 模型测试（可选，需要GPU）
echo "Step 3: 使用 tiny 模型测试（需要GPU）..."
echo "执行: docker run --gpus all --rm -it -v ~/.cache/huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format --prompt \"test\" --save-output --backend diffusers"
echo ""
echo "注意：这一步需要GPU，如果失败说明缺少diffusion依赖"
echo ""

# Step 4: 验证安装后是否恢复
echo "Step 4: 验证安装后是否恢复..."
echo "进入容器并安装 diffusion extras..."
echo ""
echo "执行以下命令："
echo "  docker run --gpus all --rm -it -v ~/.cache/huggingface:/root/.cache/huggingface lmsysorg/sglang:dev bash"
echo ""
echo "在容器内执行："
echo "  uv pip install 'sglang[diffusion]' --prerelease=allow"
echo "  python -c 'import diffusers; print(\"✓ diffusers installed\")'"
echo "  sglang generate --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format --prompt \"test\" --save-output --backend diffusers"
echo ""

echo "=========================================="
echo "复现完成！"
echo "=========================================="
