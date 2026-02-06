#!/bin/bash
# 02: 启动 GLM-Image 服务器
# 按照 BME2-Workspace-Standard 规范设置环境变量

set -e

# 1. 设置环境变量（按照 file_rules.md 规范）
if [ -f "/data/users/yandache/_shared/tools/env.sh" ]; then
    source /data/users/yandache/_shared/tools/env.sh
    echo "✓ 已加载环境变量配置 (_shared/tools/env.sh)"
else
    echo "⚠️  警告: _shared/tools/env.sh 不存在，使用默认设置"
    export SPACE=/data/users/yandache
    export HF_HOME="$SPACE/_shared/cache/hf"
    export TRANSFORMERS_CACHE="$SPACE/_shared/cache/hf/transformers"
    export XDG_CACHE_HOME="$SPACE/_shared/cache/xdg/cache"
    export TMPDIR="$SPACE/tmp"
fi

# 2. 确保目录存在
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

# 3. 显示配置信息
echo ""
echo "=== 环境配置 ==="
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "TMPDIR: $TMPDIR"
echo ""

# 4. 验证缓存路径
echo "=== 验证缓存路径 ==="
python3 << 'EOF'
import os
try:
    from huggingface_hub import constants
    actual_path = constants.HF_HOME
    expected_path = os.environ.get('HF_HOME', '')
    print(f"环境变量 HF_HOME: {expected_path}")
    print(f"HuggingFace 实际使用: {actual_path}")
    if expected_path and actual_path.startswith(expected_path):
        print("✓ 缓存路径正确")
    else:
        print("⚠️  警告：缓存路径可能不正确")
except Exception as e:
    print(f"⚠️  无法验证: {e}")
EOF

echo ""

# 5. 切换到项目目录并激活环境
cd /data/users/yandache/workspaces/sglang
source env_sglang/bin/activate

# 6. 检查 diffusion 依赖（必需）
echo "=== 检查 diffusion 依赖 ==="
if ! python3 -c "import diffusers" 2>/dev/null; then
    echo "⚠️  错误: 未安装 diffusers 库"
    echo ""
    echo "解决方案: 安装 sglang[diffusion]"
    echo "  cd repo/sglang-src/python"
    echo "  pip install -e \".[diffusion]\""
    echo ""
    echo "原因: diffusion 是 optional dependency，必须显式安装才能使用 GLM-Image"
    exit 1
fi
echo "✓ diffusers 已安装"
echo ""

# 7. 启动服务器
echo "=== 启动 GLM-Image 服务器 ==="
echo "模型: zai-org/GLM-Image"
echo "端口: 30000"
echo "缓存位置: $HF_HOME"
echo ""
echo "注意："
echo "  - 首次运行会下载模型（约 15GB），请确保有足够空间"
echo "  - 必须安装 sglang[diffusion] 才能使用 GLM-Image"
echo ""
echo "⚠️  重要：此脚本只启动服务器，性能测试需要："
echo "   1. 在另一个终端运行基准测试脚本（03_run_benchmark.sh）"
echo ""

# 检查是否指定了后端
BACKEND="${1:-sglang}"
if [ "$BACKEND" != "sglang" ] && [ "$BACKEND" != "diffusers" ]; then
    echo "用法: $0 [sglang|diffusers]"
    echo "默认使用 sglang 后端"
    BACKEND="sglang"
fi

echo "使用后端: $BACKEND"
echo "使用参数: --trust-remote-code (GLM-Image 需要自定义代码)"
echo ""
echo "提示："
echo "  - 按 Ctrl+C 优雅关闭服务器"
echo "  - 关闭时会等待后台任务完成"
echo "  - 如需强制退出，再次按 Ctrl+C"
echo ""

sglang serve \
    --model-path zai-org/GLM-Image \
    --backend "$BACKEND" \
    --port 30000 \
    --trust-remote-code
