#!/usr/bin/env bash
# 唯一配置入口。按实际环境修改标记为「需修改」的默认值。

export SGLANG_ROOT="${SGLANG_ROOT:-/home/l00889655/sglang/sglang}"
export KERNEL_ROOT="${KERNEL_ROOT:-/home/l00889655/sglang/sgl-kernel-npu}"
export MODEL_PATH="${MODEL_PATH:-/mnt/paas/weights/Qwen3.5-4B}"   # hybrid(linear attention) 模型

export NPU_DEVICES="${NPU_DEVICES:-0,1,2,3}"  # 空闲物理卡 ID 列表
export PORT="${PORT:-30000}"

# 驱逐压力参数：target 请求驱逐后重放，filler 请求占满 device mamba slot
export TARGET_PROMPT_CHARS="${TARGET_PROMPT_CHARS:-20000}"
export TARGET_NEW_TOKENS="${TARGET_NEW_TOKENS:-64}"
export NUM_FILLER="${NUM_FILLER:-5}"
export FILLER_PROMPT_CHARS="${FILLER_PROMPT_CHARS:-12000}"
export FILLER_NEW_TOKENS="${FILLER_NEW_TOKENS:-8}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

# 服务端容量控制：小 device 池 + 大 host 池，制造 L2 驱逐与回读
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export HICACHE_RATIO="${HICACHE_RATIO:-4}"

# Ascend 运行环境；set_env.sh 会引用未定义变量，source 期间临时关闭 -u。
set +u
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"

export SGLANG_HICACHE_HOST_MEM_BACKEND="${SGLANG_HICACHE_HOST_MEM_BACKEND:-acl}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
# Use the packaged build that registers both kv_exchange and
# sgl_chunk_gated_delta_rule. Override this path for an installed wheel.
export KERNEL_PYTHON_PATH="${KERNEL_PYTHON_PATH:-$KERNEL_ROOT/build/torch/package}"
export PYTHONPATH="$SGLANG_ROOT/python:$KERNEL_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}"
