#!/bin/bash
# A01: 启动多卡 SP 服务器（2 GPU + 序列并行）
# 用法: ./A01_start_server.sh
# 启动后，在另一终端运行 A02_run_multi_gpu_bench.sh 做多卡测试；结果可与单卡 results/ 用 03_compare_single_vs_multi_gpu.sh 对比。

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
echo "=== 启动多卡 SP 服务器 (2 GPU) ==="
echo "模型: zai-org/GLM-Image"
echo "端口: 30000"
echo "TP=1, SP-degree=2, Ulysses-degree=2"
echo ""
echo "启动后请在另一终端运行: ./A02_run_multi_gpu_bench.sh"
echo "按 Ctrl+C 关闭服务器"
echo ""

# 4. 结果目录：保存日志与 GPU 状态，用于确认两卡均被调用且在工作
BENCH_BASE="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark"
BENCH_DIR="${BENCH_BASE}/02112026"
mkdir -p "$BENCH_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_PREFIX="${BENCH_DIR}/run_${TS}"
echo "结果与日志目录: $BENCH_DIR"
echo "本次运行前缀: ${LOG_PREFIX}"
echo ""

# 5. 启动前记录 GPU 状态（确认可见 GPU 数量）
echo "=== 启动前 GPU 状态 (CUDA_VISIBLE_DEVICES 未限制) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits > "${LOG_PREFIX}_gpu_before_start.txt" 2>&1 || true
nvidia-smi >> "${LOG_PREFIX}_gpu_before_start.txt" 2>&1 || true
echo "已保存到: ${LOG_PREFIX}_gpu_before_start.txt"
echo ""

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 6. 后台定期采样 GPU 使用率，用于确认两卡是否都在工作（采样结果写入 BENCH_DIR）
(
  while true; do
    echo "=== $(date -Iseconds) ===" >> "${LOG_PREFIX}_gpu_usage_during_run.log"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader >> "${LOG_PREFIX}_gpu_usage_during_run.log" 2>&1 || true
    nvidia-smi >> "${LOG_PREFIX}_gpu_usage_during_run.log" 2>&1 || true
    sleep 15
  done
) &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

# 7. 启动服务：输出同时写入控制台和日志
echo "=== 启动 sglang serve (2 GPU) ==="
sglang serve \
  --model-path zai-org/GLM-Image \
  --num-gpus 2 \
  --port 30000 \
  --trust-remote-code 2>&1 | tee "${LOG_PREFIX}_server.log"