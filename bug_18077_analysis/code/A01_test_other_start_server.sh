#!/bin/bash
# 多卡环境验证：用标准 LLM（Qwen2.5-1.5B）TP=2 测试，确认本机多卡能被 sglang 识别并跑通
# 用法: ./A01_test_other_start_server.sh

set -e

# 1. 环境变量（与 01 一致）
if [ -f "/data/users/yandache/_shared/tools/env.sh" ]; then
    source /data/users/yandache/_shared/tools/env.sh
    echo "✓ 已加载环境变量配置"
else
    export SPACE=/data/users/yandache
    export HF_HOME="$SPACE/_shared/cache/hf"
    export TRANSFORMERS_CACHE="$SPACE/_shared/cache/hf/transformers"
    export XDG_CACHE_HOME="$SPACE/_shared/cache/xdg/cache"
    export TMPDIR="$SPACE/tmp"
fi
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

# 2. 只用 2 张卡（默认 0,1）；若有其他进程占卡，可指定空闲的，例如: GPU_IDS=2,3 ./A01_test_other_start_server.sh
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1}"

# 3. 激活环境并进入项目
cd /data/users/yandache/workspaces/sglang
source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

echo ""
echo "=== 多卡验证：标准 LLM TP=2 (Qwen2.5-1.5B-Instruct) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "端口: 30000"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES  （指定空闲卡: GPU_IDS=2,3 ./A01_test_other_start_server.sh）"
echo "成功标志: 日志里出现多卡/TP 相关信息且无报错"
echo "按 Ctrl+C 关闭"
echo ""

# 标准 LLM 用 sglang serve，多卡用 --tp-size（不是 --tp）
exec sglang serve \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --tp-size 2 \
    --port 30000 \
    --host 0.0.0.0 \
    --trust-remote-code
