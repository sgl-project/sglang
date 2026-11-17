#!/usr/bin/env bash
# Set paths for the target machine before running the case.
export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B}"
export SGLANG_ROOT="${SGLANG_ROOT:-/path/to/sglang}"
export KERNEL_ROOT="${KERNEL_ROOT:-/path/to/sgl-kernel-npu}"
export KERNEL_PYTHON_PATH="${KERNEL_PYTHON_PATH:-$KERNEL_ROOT/python/sgl_kernel_npu}"
export GSM8K_TRAIN_PATH="${GSM8K_TRAIN_PATH:-/root/.cache/modelscope/hub/datasets/grade_school_math/train.jsonl}"
export GSM8K_TEST_PATH="${GSM8K_TEST_PATH:-/root/.cache/modelscope/hub/datasets/grade_school_math/test.jsonl}"
export ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}"
export NPU_DEVICES="${NPU_DEVICES:-0}"
export PORT="${PORT:-30000}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
export GSM8K_NUM_QUESTIONS="${GSM8K_NUM_QUESTIONS:-200}"
export GSM8K_NUM_SHOTS="${GSM8K_NUM_SHOTS:-8}"
export GSM8K_ACCURACY="${GSM8K_ACCURACY:-}" # Optional: measured baseline threshold.
export GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-512}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
export PAGE_SIZE="${PAGE_SIZE:-128}"
export HICACHE_RATIO="${HICACHE_RATIO:-6}"
export FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-4096}"
export MAX_FILLER_REQUESTS="${MAX_FILLER_REQUESTS:-12}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
export SGLANG_HICACHE_HOST_MEM_BACKEND="${SGLANG_HICACHE_HOST_MEM_BACKEND:-acl}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True,pin_memory_expandable_segments:True}"

set +u
source "$ASCEND_ENV_SCRIPT"
set -u
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
IFS=, read -r -a _npu_cards <<< "$NPU_DEVICES"
export TP_SIZE="${#_npu_cards[@]}"
export PYTHONPATH="$SGLANG_ROOT/python:$KERNEL_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}"
