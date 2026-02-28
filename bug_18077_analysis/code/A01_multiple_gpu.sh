#!/bin/bash
# A01_multiple_gpu: 启动 2 卡 SP=2 服务器（tp=1, sp=2），用于多卡分步测试
# 用法: ./A01_multiple_gpu.sh
# 启动后，在另一终端运行 ./A03_multiple_gpu.sh 做分层验证（参考 A03 设计），观察 [A03_GLM] 与 sharded shape

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

# 3. 检查
if ! python3 -c "import diffusers" 2>/dev/null; then
    echo "错误: 未安装 diffusers，请 pip install -e \".[diffusion]\""
    exit 1
fi

echo ""
echo "=== 启动多卡 SP 服务器 (2 GPU, tp=1, sp=2) ==="
echo "模型: zai-org/GLM-Image"
echo "端口: 30000"
echo "TP=1, SP-degree=2"
echo ""
echo "启动后请在另一终端运行: ./A03_multiple_gpu.sh"
echo "按 Ctrl+C 关闭服务器"
echo ""

# 4. 2 卡可见
export CUDA_VISIBLE_DEVICES=0,1

# 5. 启动：显式 tp=1, sp=2
sglang serve \
  --model-path zai-org/GLM-Image \
  --num-gpus 2 \
  --tp-size 1 \
  --sp-degree 2 \
  --port 30000 \
  --trust-remote-code
