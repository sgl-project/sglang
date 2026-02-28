#!/bin/bash
# A01_single_gpu: 单卡启动 GLM-Image 服务器（tp-size=1，用于与多卡 A01+A02 对比）
# 用法: ./A01_single_gpu.sh
# 启动后，在另一终端运行 A02_single_gpu.sh 做单卡 benchmark（random dataset，与 A02 同结构），
# 并验证调用路径为 GlmImagePipelineConfig / SpatialImagePipelineConfig。

set -e

# 1. 环境变量
if [ -f "/data/users/yandache/_shared/tools/env.sh" ]; then
    source /data/users/yandache/_shared/tools/env.sh
    echo "✓ 已加载环境变量配置 (_shared/tools/env.sh)"
else
    export SPACE=/data/users/yandache
    export HF_HOME="$SPACE/_shared/cache/hf"
    export TRANSFORMERS_CACHE="$SPACE/_shared/cache/hf/transformers"
    export XDG_CACHE_HOME="$SPACE/_shared/cache/xdg/cache"
    export TMPDIR="$SPACE/tmp"
fi
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

# 2. 项目目录与虚拟环境
cd /data/users/yandache/workspaces/sglang
source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

# 3. 检查 diffusion
if ! python3 -c "import diffusers" 2>/dev/null; then
    echo "错误: 未安装 diffusers，请 pip install -e \".[diffusion]\""
    exit 1
fi

echo ""
echo "=== 启动单卡服务器 (1 GPU, tp-size=1) ==="
echo "模型: zai-org/GLM-Image"
echo "端口: 30000"
echo "TP=1（单卡）"
echo ""
echo "启动后请在另一终端运行: ./A02_single_gpu.sh"
echo "按 Ctrl+C 关闭服务器"
echo ""

# 4. 仅使用单卡
export CUDA_VISIBLE_DEVICES=0

# 5. 启动服务
echo "=== 启动 sglang serve (1 GPU, tp-size 1) ==="
sglang serve \
  --model-path zai-org/GLM-Image \
  --tp-size 1 \
  --port 30000 \
  --trust-remote-code
