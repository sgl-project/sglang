#!/bin/bash
set -euo pipefail

export PYTHONPATH=/workspace/sglang/python
export HOME=/host_scratch/.wf_megamoe_home
export USER=xutingz
export LOGNAME=xutingz
export HF_HOME=/host_scratch/.cache/huggingface
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX=/tmp/pycache_megamoe_cap_eval_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/host_scratch/.cache/torchinductor
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=${MEGA_MOE_CAP:-8320}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS=${MEGA_MOE_CAP_BUCKETS:-}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_PREINIT_ALL_CAP_BUCKETS=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_PREINIT_ALL_CAP_BUCKETS:-0}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_CAP_BUCKET_MIN_FREE_GB=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_CAP_BUCKET_MIN_FREE_GB:-4.0}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS:-0}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND:-0}
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export OPENAI_API_KEY=EMPTY
export PYTHONUNBUFFERED=1

MODEL=${MODEL:-/host_scratch/model/DeepSeek-V4-Flash}
RESULT_ROOT=/host_scratch/megamoe_cap_eval_${SLURM_JOB_ID}
mkdir -p "$RESULT_ROOT"

PORT=${PORT:-30000}
CASE_PORT_STRIDE=${CASE_PORT_STRIDE:-1000}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-megamoe}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-}
DEEPEP_CONFIG=${DEEPEP_CONFIG:-}
EP_NUM_REDUNDANT_EXPERTS=${EP_NUM_REDUNDANT_EXPERTS:-0}
EP_DISPATCH_ALGORITHM=${EP_DISPATCH_ALGORITHM:-}
INIT_EXPERT_LOCATION=${INIT_EXPERT_LOCATION:-}
ENABLE_EPLB=${ENABLE_EPLB:-0}
EPLB_ALGORITHM=${EPLB_ALGORITHM:-}
EPLB_REBALANCE_NUM_ITERATIONS=${EPLB_REBALANCE_NUM_ITERATIONS:-}
EPLB_REBALANCE_LAYERS_PER_CHUNK=${EPLB_REBALANCE_LAYERS_PER_CHUNK:-}
EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD=${EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD:-}
EXPERT_DISTRIBUTION_RECORDER_MODE=${EXPERT_DISTRIBUTION_RECORDER_MODE:-}
EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE=${EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE:-}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
DISABLE_SHARED_EXPERTS_FUSION=${DISABLE_SHARED_EXPERTS_FUSION:-0}
SKIP_SERVER_WARMUP=${SKIP_SERVER_WARMUP:-1}
TP_SIZE=${TP_SIZE:-2}
DP_SIZE=${DP_SIZE:-1}
MOE_DENSE_TP_SIZE=${MOE_DENSE_TP_SIZE:-1}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-0}
ALLOW_READY_LOG_HEALTH=${ALLOW_READY_LOG_HEALTH:-0}
WAIT_HEALTH_MODE=${WAIT_HEALTH_MODE:-health}
WAIT_HEALTH_ATTEMPTS=${WAIT_HEALTH_ATTEMPTS:-1200}
READY_LOG_STABILIZE_SECONDS=${READY_LOG_STABILIZE_SECONDS:-60}

MMLU_NSUB=${MMLU_NSUB:-60}
MMLU_NUM_EXAMPLES=${MMLU_NUM_EXAMPLES:-}
MMLU_NTRAIN=${MMLU_NTRAIN:-5}
MMLU_SIMPLE_EXAMPLES=${MMLU_SIMPLE_EXAMPLES:-}
GSM8K_EXAMPLES=${GSM8K_EXAMPLES:-500}
GPQA_EXAMPLES=${GPQA_EXAMPLES:-198}
EVAL_THREADS=${EVAL_THREADS:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
DSV4_THINKING_MODE=${DSV4_THINKING_MODE:-chat}
GPQA_THINKING_MODE=${GPQA_THINKING_MODE:-chat}
GEN_PROMPT_MODE=${GEN_PROMPT_MODE:-dsv4-chat}
MMLU_PROMPT_MODE=${MMLU_PROMPT_MODE:-raw}
MMLU_SCORING=${MMLU_SCORING:-regex}
MMLU_MAX_NEW_TOKENS=${MMLU_MAX_NEW_TOKENS:-}
RUN_EVAL_API=${RUN_EVAL_API:-chat}
if [ -z "$MMLU_MAX_NEW_TOKENS" ]; then
  if [ "$MMLU_SCORING" = "simple-chat" ]; then
    MMLU_MAX_NEW_TOKENS=512
  elif [ "$MMLU_SCORING" = "sglang-bench-gen" ]; then
    MMLU_MAX_NEW_TOKENS=1
  else
    MMLU_MAX_NEW_TOKENS=2
  fi
fi
MMLU_CHOICE_PREFIX=${MMLU_CHOICE_PREFIX:-auto}

RUN_SMOKE=${RUN_SMOKE:-1}
RUN_MMLU=${RUN_MMLU:-1}
RUN_GSM8K=${RUN_GSM8K:-1}
RUN_GPQA=${RUN_GPQA:-1}
RUN_PREFILL_BENCH=${RUN_PREFILL_BENCH:-1}
RUN_PREFILL_PREWARM=${RUN_PREFILL_PREWARM:-0}
RUN_NO_WATERFILL=${RUN_NO_WATERFILL:-1}
RUN_WATERFILL=${RUN_WATERFILL:-1}
CASE_ORDER=${CASE_ORDER:-no_waterfill,waterfill}
SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE=${SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE:-64}
SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK=${SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK:-}

PERF_NUM_PROMPTS=${PERF_NUM_PROMPTS:-128}
PERF_INPUT_LEN=${PERF_INPUT_LEN:-8192}
PERF_OUTPUT_LEN=${PERF_OUTPUT_LEN:-1}
PERF_CONCURRENCY=${PERF_CONCURRENCY:-16}
PERF_PREWARM_PROMPTS=${PERF_PREWARM_PROMPTS:-16}
PERF_PREWARM_CONCURRENCY=${PERF_PREWARM_CONCURRENCY:-$PERF_CONCURRENCY}
PERF_RANDOM_RANGE_RATIO=${PERF_RANDOM_RANGE_RATIO:-0}
PERF_DISABLE_STREAM=${PERF_DISABLE_STREAM:-0}
PERF_WARMUP_REQUESTS=${PERF_WARMUP_REQUESTS:-8}
PERF_MEASURE_REPEATS=${PERF_MEASURE_REPEATS:-1}
PERF_BENCH_TIMEOUT_SEC=${PERF_BENCH_TIMEOUT_SEC:-900}
PERF_DATASET_NAME=${PERF_DATASET_NAME:-random}
PERF_DATASET_PATH=${PERF_DATASET_PATH:-}
PERF_SHAREGPT_OUTPUT_LEN=${PERF_SHAREGPT_OUTPUT_LEN:-4}
PERF_MMLU_DATA_DIR=${PERF_MMLU_DATA_DIR:-/host_scratch/mmlu_data}
PERF_MMLU_NUM_EXAMPLES=${PERF_MMLU_NUM_EXAMPLES:-$PERF_NUM_PROMPTS}
PERF_MMLU_NSUB=${PERF_MMLU_NSUB:-$MMLU_NSUB}
PERF_MMLU_NTRAIN=${PERF_MMLU_NTRAIN:-$MMLU_NTRAIN}
PERF_MMLU_PROMPT_MODE=${PERF_MMLU_PROMPT_MODE:-raw}
PERF_ENABLE_PROFILE=${PERF_ENABLE_PROFILE:-0}
PERF_PROFILE_ACTIVITIES=${PERF_PROFILE_ACTIVITIES:-GPU}
PERF_PROFILE_START_STEP=${PERF_PROFILE_START_STEP:-1}
PERF_PROFILE_STEPS=${PERF_PROFILE_STEPS:-0}

prepare_perf_dataset() {
  local dataset_name=$1
  if [ "$dataset_name" != "mmlu_custom" ]; then
    return 0
  fi
  PERF_DATASET_PATH="$RESULT_ROOT/mmlu_custom_n${PERF_MMLU_NUM_EXAMPLES}_${PERF_MMLU_PROMPT_MODE}.jsonl"
  export PERF_DATASET_PATH
  if [ -s "$PERF_DATASET_PATH" ]; then
    return 0
  fi
  echo "=== BUILD_MMLU_CUSTOM_DATASET path=$PERF_DATASET_PATH n=${PERF_MMLU_NUM_EXAMPLES} prompt_mode=${PERF_MMLU_PROMPT_MODE} $(date) ==="
  python - <<PY
import json
import os
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, "/workspace/sglang/scripts")
from dsv4_native_eval import (  # noqa: E402
    download_mmlu_data,
    encode_user_prompt,
    load_dsv4_encoder,
    mmlu_format_example,
    mmlu_gen_prompt,
)

data_dir = "${PERF_MMLU_DATA_DIR}"
out = Path("${PERF_DATASET_PATH}")
download_mmlu_data(data_dir)
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
subjects = sorted(
    f.split("_test.csv")[0]
    for f in os.listdir(os.path.join(data_dir, "test"))
    if f.endswith("_test.csv")
)[: int("${PERF_MMLU_NSUB}")]
prompts = []
for subject in subjects:
    dev_df = pd.read_csv(
        os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
    )[: int("${PERF_MMLU_NTRAIN}")]
    test_df = pd.read_csv(
        os.path.join(data_dir, "test", subject + "_test.csv"), header=None
    )
    k = int("${PERF_MMLU_NTRAIN}")
    few_shot = mmlu_gen_prompt(dev_df, subject, k)
    while k > 0 and len(tokenizer.encode(few_shot)) > 1536:
        k -= 1
        few_shot = mmlu_gen_prompt(dev_df, subject, k)
    for i in range(test_df.shape[0]):
        prompts.append(few_shot + mmlu_format_example(test_df, i, False))

n = int("${PERF_MMLU_NUM_EXAMPLES}")
if n > 0:
    prompts = prompts[:n]
if "${PERF_MMLU_PROMPT_MODE}" == "dsv4-chat":
    encoder = load_dsv4_encoder("${MODEL}")
    prompts = [encode_user_prompt(encoder, p, "${DSV4_THINKING_MODE}") for p in prompts]
elif "${PERF_MMLU_PROMPT_MODE}" != "raw":
    raise ValueError("PERF_MMLU_PROMPT_MODE must be raw or dsv4-chat")

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for prompt in prompts:
        f.write(
            json.dumps(
                {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": " The answer is A."},
                    ]
                },
                ensure_ascii=False,
            )
            + "\\n"
        )
print(json.dumps({"path": str(out), "rows": len(prompts), "subjects": len(subjects)}))
PY
}

perf_dataset_args() {
  local dataset_name=$1
  if [ "$dataset_name" = "random" ]; then
    printf '%s\n' \
      --dataset-name random \
      --random-input-len "$PERF_INPUT_LEN" \
      --random-output-len "$PERF_OUTPUT_LEN" \
      --random-range-ratio "$PERF_RANDOM_RANGE_RATIO"
  elif [ "$dataset_name" = "mmlu_custom" ]; then
    printf '%s\n' \
      --dataset-name custom \
      --dataset-path "$PERF_DATASET_PATH" \
      --sharegpt-output-len "$PERF_SHAREGPT_OUTPUT_LEN" \
      --sharegpt-context-len "$MAX_PREFILL_TOKENS"
  elif [ "$dataset_name" = "custom" ]; then
    printf '%s\n' \
      --dataset-name custom \
      --dataset-path "$PERF_DATASET_PATH" \
      --sharegpt-output-len "$PERF_SHAREGPT_OUTPUT_LEN" \
      --sharegpt-context-len "$MAX_PREFILL_TOKENS"
  else
    printf '%s\n' \
      --dataset-name "$dataset_name" \
      --dataset-path "$PERF_DATASET_PATH"
  fi
}

python -m compileall -q \
  /workspace/sglang/python/sglang/srt/layers/moe/utils.py \
  /workspace/sglang/python/sglang/srt/layers/moe/topk.py \
  /workspace/sglang/python/sglang/srt/layers/moe/hash_topk.py \
  /workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py \
  /workspace/sglang/python/sglang/srt/layers/quantization/fp8.py \
  /workspace/sglang/python/sglang/srt/models/deepseek_v2.py \
  /workspace/sglang/python/sglang/srt/models/deepseek_v4.py \
  /workspace/sglang/python/sglang/srt/models/qwen2_moe.py

python - <<'PY'
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

model = Path(os.environ["MODEL"])
cfg = json.loads((model / "config.json").read_text())
print("MODEL_PATH", model)
print("MODEL_TYPE", cfg.get("model_type"))
print("QUANT_METHOD", cfg.get("quantization_config", {}).get("quant_method"))
print("MEGA_MOE_CAP", __import__("os").environ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"])
print("KV_CACHE_DTYPE", __import__("os").environ.get("KV_CACHE_DTYPE"))
tok = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
print("TOKENIZER_OK", type(tok).__name__, "eos=", tok.eos_token_id)
assert cfg.get("model_type") == "deepseek_v4"
PY

wait_health() {
  local pid=$1
  local log=$2
  for _ in $(seq 1 "$WAIT_HEALTH_ATTEMPTS"); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "SERVER_EXITED_BEFORE_HEALTH log=$log"
      tail -n 260 "$log" || true
      return 1
    fi
    if [ "$WAIT_HEALTH_MODE" = "ready-log" ] && grep -q "The server is fired up and ready to roll" "$log"; then
      echo "SERVER_READY_LOG_OK"
      sleep "$READY_LOG_STABILIZE_SECONDS"
      return 0
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/megamoe_cap_eval_health.out 2>/tmp/megamoe_cap_eval_health.err; then
      echo "HEALTH_OK"
      return 0
    fi
    if [ "$ALLOW_READY_LOG_HEALTH" = "1" ] && grep -q "The server is fired up and ready to roll" "$log"; then
      echo "SERVER_READY_LOG_OK"
      return 0
    fi
    sleep 2
  done
  echo "HEALTH_TIMEOUT log=$log"
  tail -n 260 "$log" || true
  return 1
}

cleanup_stale_server_processes() {
  local port=$1
  pkill -u "$USER" -f "sglang.launch_server.*--model-path ${MODEL}.*--port ${port}" >/dev/null 2>&1 || true
  pkill -u "$USER" -f "sglang.launch_server.*--port ${port}.*--model-path ${MODEL}" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! pgrep -u "$USER" -f "sglang.launch_server.*--model-path ${MODEL}.*--port ${port}" >/dev/null 2>&1 \
      && ! pgrep -u "$USER" -f "sglang.launch_server.*--port ${port}.*--model-path ${MODEL}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  pkill -KILL -u "$USER" -f "sglang.launch_server.*--model-path ${MODEL}.*--port ${port}" >/dev/null 2>&1 || true
  pkill -KILL -u "$USER" -f "sglang.launch_server.*--port ${port}.*--model-path ${MODEL}" >/dev/null 2>&1 || true
  sleep 2
}

smoke_completion() {
  echo "=== SMOKE_COMPLETION_BEGIN $(date) ==="
  python - <<PY
import sys

import requests

sys.path.insert(0, "${MODEL}/encoding")
import encoding_dsv4

prompt = encoding_dsv4.encode_messages(
    [{"role": "user", "content": "Question: What is 2+2?\\nAnswer:"}],
    thinking_mode="${DSV4_THINKING_MODE}",
    drop_thinking=True,
)
resp = requests.post(
    "http://127.0.0.1:${PORT}/generate",
    json={
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1.0,
            "max_new_tokens": 16,
            "stop": [encoding_dsv4.eos_token],
        },
    },
    timeout=600,
)
resp.raise_for_status()
print("SMOKE_ANSWER", repr(resp.json().get("text", "")))
PY
  echo "=== SMOKE_COMPLETION_DONE $(date) ==="
}

run_prefill_prewarm() {
  local case_dir=$1
  local stream_args=()
  if [ "$PERF_DISABLE_STREAM" = "1" ]; then
    stream_args+=(--disable-stream)
  fi
  prepare_perf_dataset "$PERF_DATASET_NAME"
  mapfile -t dataset_args < <(perf_dataset_args "$PERF_DATASET_NAME")
  echo "=== PREFILL_PREWARM_BEGIN prompts=${PERF_PREWARM_PROMPTS} input=${PERF_INPUT_LEN} output=${PERF_OUTPUT_LEN} concurrency=${PERF_PREWARM_CONCURRENCY} $(date) ==="
    timeout --kill-after=60s "${PERF_BENCH_TIMEOUT_SEC}s" python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    "${dataset_args[@]}" \
    --num-prompts "$PERF_PREWARM_PROMPTS" \
    --request-rate inf \
    --max-concurrency "$PERF_PREWARM_CONCURRENCY" \
    --warmup-requests 0 \
    --ready-check-timeout-sec 0 \
    --output-file "$case_dir/prefill_prewarm.jsonl" \
    --disable-tqdm \
    "${stream_args[@]}" \
    >"$case_dir/prefill_prewarm.log" 2>&1
  echo "=== PREFILL_PREWARM_DONE $(date) ==="
  tail -n 80 "$case_dir/prefill_prewarm.log" || true
}

run_native_mmlu() {
  local case_dir=$1
  echo "=== EVAL_BEGIN mmlu nsub=${MMLU_NSUB} $(date) ==="
  local num_examples_args=()
  if [ -n "$MMLU_NUM_EXAMPLES" ]; then
    num_examples_args+=(--num-examples "$MMLU_NUM_EXAMPLES")
  fi
  python /workspace/sglang/scripts/dsv4_native_eval.py \
      --task mmlu \
      --model-path "$MODEL" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --thinking-mode "$DSV4_THINKING_MODE" \
      --nsub "$MMLU_NSUB" \
      --mmlu-ntrain "$MMLU_NTRAIN" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --max-new-tokens "$MMLU_MAX_NEW_TOKENS" \
      --mmlu-prompt-mode "$MMLU_PROMPT_MODE" \
      --mmlu-scoring "$MMLU_SCORING" \
      --mmlu-choice-prefix "$MMLU_CHOICE_PREFIX" \
      "${num_examples_args[@]}" \
      --mmlu-data-dir /host_scratch/mmlu_data \
      --result-file "$case_dir/mmlu_result.jsonl" \
      --raw-result-file "$case_dir/mmlu_raw.jsonl" \
      >"$case_dir/mmlu.log" 2>&1
  echo "=== EVAL_DONE mmlu $(date) ==="
  tail -n 50 "$case_dir/mmlu.log" || true
}

run_bench_mmlu() {
  local case_dir=$1
  echo "=== BENCH_MMLU_BEGIN nsub=${MMLU_NSUB} $(date) ==="
  local num_examples_args=()
  if [ -n "$MMLU_NUM_EXAMPLES" ]; then
    num_examples_args+=(--num-examples "$MMLU_NUM_EXAMPLES")
  fi
  python /workspace/sglang/benchmark/mmlu/bench_sglang.py \
      --host 127.0.0.1 \
      --port "$PORT" \
      --backend srt \
      --parallel "$EVAL_THREADS" \
      --nsub "$MMLU_NSUB" \
      --ntrain "$MMLU_NTRAIN" \
      "${num_examples_args[@]}" \
      --data_dir /host_scratch/mmlu_data \
      --result-file "$case_dir/mmlu_result.jsonl" \
      --raw-result-file "$case_dir/mmlu_raw.jsonl" \
      >"$case_dir/mmlu.log" 2>&1
  echo "=== BENCH_MMLU_DONE $(date) ==="
  tail -n 80 "$case_dir/mmlu.log" || true
}

run_simple_eval() {
  local case_dir=$1
  local task=$2
  local examples=$3
  local max_tokens=$4
  local result_path="$case_dir/${task}_result.jsonl"
  local log_path="$case_dir/${task}.log"
  if [ "$task" = "gpqa" ]; then
    result_path="$case_dir/gpqa_result.json"
  fi

  echo "=== SIMPLE_EVAL_BEGIN $task examples=${examples:-all} $(date) ==="
  local cmd=(
    python -m sglang.test.run_eval
    --eval-name "$task"
    --api "$RUN_EVAL_API"
    --host 127.0.0.1
    --port "$PORT"
    --model "$MODEL"
    --num-threads "$EVAL_THREADS"
    --max-tokens "$max_tokens"
    --temperature 0
    --top-p 1.0
  )
  if [ -n "$examples" ] && [ "$examples" != "all" ]; then
    cmd+=(--num-examples "$examples")
  fi
  "${cmd[@]}" >"$log_path" 2>&1
  echo "=== SIMPLE_EVAL_DONE $task $(date) ==="
  tail -n 80 "$log_path" || true

  python - <<PY
import ast
import json
import pathlib
import re

task = "$task"
log_path = pathlib.Path("$log_path")
result_path = pathlib.Path("$result_path")
text = log_path.read_text(errors="replace")

score_match = re.search(r"Score:\\s*([0-9.]+)", text)
lat_match = re.search(r"Total latency:\\s*([0-9.]+)\\s*s", text)
tput_match = re.search(r"Output throughput:\\s*([0-9.]+)\\s*token/s", text)

metrics = {}
for line in reversed(text.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        try:
            metrics = ast.literal_eval(line)
            if isinstance(metrics, dict):
                break
        except Exception:
            pass

result = {
    "task": task,
    "backend": "sglang-test-run-eval-chat",
    "accuracy": float(score_match.group(1)) if score_match else metrics.get("score"),
    "score": float(score_match.group(1)) if score_match else metrics.get("score"),
    "latency": float(lat_match.group(1)) if lat_match else metrics.get("latency"),
    "output_throughput": float(tput_match.group(1)) if tput_match else metrics.get("output_throughput"),
    "num_requests": int("$examples") if "$examples" and "$examples" != "all" else None,
    "other": {
        "api": "$RUN_EVAL_API",
        "num_threads": int("$EVAL_THREADS"),
        "max_tokens": int("$max_tokens"),
        "metrics": metrics,
    },
}
result_path.parent.mkdir(parents=True, exist_ok=True)
if result_path.suffix == ".jsonl":
    result_path.write_text(json.dumps(result) + "\\n")
else:
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

run_native_gsm8k() {
  local case_dir=$1
  echo "=== EVAL_BEGIN gsm8k examples=${GSM8K_EXAMPLES} $(date) ==="
  python /workspace/sglang/scripts/dsv4_native_eval.py \
      --task gsm8k \
      --model-path "$MODEL" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --thinking-mode "$DSV4_THINKING_MODE" \
      --prompt-mode "$GEN_PROMPT_MODE" \
      --num-examples "$GSM8K_EXAMPLES" \
      --num-shots 8 \
      --batch-size "$EVAL_BATCH_SIZE" \
      --max-new-tokens 512 \
      --result-file "$case_dir/gsm8k_result.jsonl" \
      --raw-result-file "$case_dir/gsm8k_raw.jsonl" \
      >"$case_dir/gsm8k.log" 2>&1
  echo "=== EVAL_DONE gsm8k $(date) ==="
  tail -n 50 "$case_dir/gsm8k.log" || true
}

run_native_gpqa() {
  local case_dir=$1
  echo "=== EVAL_BEGIN gpqa examples=${GPQA_EXAMPLES} $(date) ==="
  python /workspace/sglang/scripts/dsv4_native_eval.py \
      --task gpqa \
      --model-path "$MODEL" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --thinking-mode "$GPQA_THINKING_MODE" \
      --prompt-mode "$GEN_PROMPT_MODE" \
      --num-examples "$GPQA_EXAMPLES" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --max-new-tokens 1024 \
      --result-file "$case_dir/gpqa_result.json" \
      --raw-result-file "$case_dir/gpqa_raw.jsonl" \
      >"$case_dir/gpqa.log" 2>&1
  echo "=== EVAL_DONE gpqa $(date) ==="
  tail -n 60 "$case_dir/gpqa.log" || true
}

run_case() {
  local name=$1
  local waterfill=$2
  local case_force_local_shared=${3:-${SGLANG_WATERFILL_FORCE_LOCAL_SHARED:-0}}
  local case_extra_server_args=${4:-}
  local case_disable_shared_experts_fusion=${5:-$DISABLE_SHARED_EXPERTS_FUSION}
  local case_fuse_megamoe_predispatch=${6:-${SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH:-}}
  local case_port=$PORT
  if [ "$name" = "waterfill" ] || [ "$name" = "fused" ]; then
    case_port=$((PORT + CASE_PORT_STRIDE))
  elif [ "$name" = "waterfill_local" ] || [ "$name" = "fused_waterfill" ]; then
    case_port=$((PORT + 2 * CASE_PORT_STRIDE))
  elif [ "$name" = "waterfill_fusedpredispatch" ]; then
    case_port=$((PORT + 3 * CASE_PORT_STRIDE))
  fi
  local PORT=$case_port
  local case_dir="$RESULT_ROOT/$name"
  mkdir -p "$case_dir"
  local server_log="$case_dir/server.log"
  rm -f "$server_log"
  cleanup_stale_server_processes "$PORT"

  local extra=()
  if [ "$waterfill" = "1" ]; then
    extra+=(--enable-deepep-waterfill)
  fi
  if [ "$case_disable_shared_experts_fusion" = "1" ]; then
    extra+=(--disable-shared-experts-fusion)
  fi
  if [ "$ENABLE_DP_ATTENTION" = "1" ]; then
    extra+=(--enable-dp-attention)
  fi
  if [ -n "$MOE_RUNNER_BACKEND" ]; then
    extra+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
  fi
  if [ -n "$DEEPEP_CONFIG" ]; then
    extra+=(--deepep-config "$DEEPEP_CONFIG")
  fi
  if [ "$EP_NUM_REDUNDANT_EXPERTS" != "0" ]; then
    extra+=(--ep-num-redundant-experts "$EP_NUM_REDUNDANT_EXPERTS")
  fi
  if [ -n "$EP_DISPATCH_ALGORITHM" ]; then
    extra+=(--ep-dispatch-algorithm "$EP_DISPATCH_ALGORITHM")
  fi
  if [ -n "$INIT_EXPERT_LOCATION" ]; then
    extra+=(--init-expert-location "$INIT_EXPERT_LOCATION")
  fi
  if [ "$ENABLE_EPLB" = "1" ]; then
    extra+=(--enable-eplb)
  fi
  if [ -n "$EPLB_ALGORITHM" ]; then
    extra+=(--eplb-algorithm "$EPLB_ALGORITHM")
  fi
  if [ -n "$EPLB_REBALANCE_NUM_ITERATIONS" ]; then
    extra+=(--eplb-rebalance-num-iterations "$EPLB_REBALANCE_NUM_ITERATIONS")
  fi
  if [ -n "$EPLB_REBALANCE_LAYERS_PER_CHUNK" ]; then
    extra+=(--eplb-rebalance-layers-per-chunk "$EPLB_REBALANCE_LAYERS_PER_CHUNK")
  fi
  if [ -n "$EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD" ]; then
    extra+=(--eplb-min-rebalancing-utilization-threshold "$EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD")
  fi
  if [ -n "$EXPERT_DISTRIBUTION_RECORDER_MODE" ]; then
    extra+=(--expert-distribution-recorder-mode "$EXPERT_DISTRIBUTION_RECORDER_MODE")
  fi
  if [ -n "$EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE" ]; then
    extra+=(--expert-distribution-recorder-buffer-size "$EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE")
  fi
  if [ -n "$EXTRA_SERVER_ARGS" ]; then
    read -r -a extra_server_args <<<"$EXTRA_SERVER_ARGS"
    extra+=("${extra_server_args[@]}")
  fi
  if [ -n "$case_extra_server_args" ]; then
    read -r -a extra_server_args <<<"$case_extra_server_args"
    extra+=("${extra_server_args[@]}")
  fi
  if [ "$SKIP_SERVER_WARMUP" = "1" ]; then
    extra+=(--skip-server-warmup)
  fi
  local common_args=()
  if [ "$DISABLE_CUDA_GRAPH" = "1" ]; then
    common_args+=(--disable-cuda-graph)
  fi
  if [ "$DISABLE_RADIX_CACHE" = "1" ]; then
    common_args+=(--disable-radix-cache)
  fi
  local unset_env_args=()
  local set_env_args=()
  if [ -n "$case_fuse_megamoe_predispatch" ]; then
    set_env_args+=(
      "SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH=$case_fuse_megamoe_predispatch"
    )
  else
    unset_env_args+=(-u SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH)
  fi
  if [ -n "$SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK" ]; then
    set_env_args+=(
      "SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK=$SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK"
    )
  else
    unset_env_args+=(-u SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK)
  fi
  if [ -n "${SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED:-}" ]; then
    set_env_args+=(
      "SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED=$SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED"
    )
  else
    unset_env_args+=(-u SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED)
  fi

  echo "=== START_CASE $name waterfill=$waterfill force_local_shared=$case_force_local_shared fuse_megamoe_predispatch=$case_fuse_megamoe_predispatch one_way_remote_shared=${SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED:-auto} min_batch_for_balance=${SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE} shared_replicas=${SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK} backend=$MOE_A2A_BACKEND runner=${MOE_RUNNER_BACKEND:-auto} disable_shared_experts_fusion=$case_disable_shared_experts_fusion case_extra_server_args=$case_extra_server_args tp=$TP_SIZE dp=$DP_SIZE cap=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK} port=$PORT $(date) ==="
  SGLANG_WATERFILL_FORCE_LOCAL_SHARED="$case_force_local_shared" \
  SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS="${SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS:-1}" \
  SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE="${SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE}" \
  env "${unset_env_args[@]}" "${set_env_args[@]}" setsid python -m sglang.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp "$TP_SIZE" \
    --dp "$DP_SIZE" \
    --moe-dense-tp-size "$MOE_DENSE_TP_SIZE" \
    --moe-a2a-backend "$MOE_A2A_BACKEND" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --host 127.0.0.1 \
    --port "$PORT" \
    "${common_args[@]}" \
    "${extra[@]}" \
    >"$server_log" 2>&1 &
  local server_pid=$!
  local server_pgid=""
  server_pgid=$(ps -o pgid= -p "$server_pid" 2>/dev/null | tr -d ' ' || true)
  local cleanup_done=0

  cleanup_case() {
    set +e
    if [ "$cleanup_done" = "1" ]; then
      set -e
      return
    fi
    cleanup_done=1
    if [ -n "$server_pgid" ]; then
      kill -TERM "-$server_pgid" >/dev/null 2>&1 || true
    fi
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    sleep 10
    if [ -n "$server_pgid" ]; then
      kill -KILL "-$server_pgid" >/dev/null 2>&1 || true
    fi
    kill -KILL "$server_pid" >/dev/null 2>&1 || true
    pkill -u "$USER" -f "sglang.launch_server.*--model-path ${MODEL}.*--port ${PORT}" >/dev/null 2>&1 || true
    pkill -u "$USER" -f "sglang.launch_server.*--port ${PORT}.*--model-path ${MODEL}" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
    cleanup_stale_server_processes "$PORT"
    sleep 5
    set -e
    return 0
  }
  trap cleanup_case RETURN

  wait_health "$server_pid" "$server_log"
  if [ "$RUN_SMOKE" = "1" ]; then
    smoke_completion
  fi
  if [ "$RUN_PREFILL_PREWARM" = "1" ]; then
    run_prefill_prewarm "$case_dir"
  fi

  if [ "$RUN_MMLU" = "1" ]; then
    if [ "$MMLU_SCORING" = "simple-chat" ]; then
      run_simple_eval "$case_dir" mmlu "$MMLU_SIMPLE_EXAMPLES" "$MMLU_MAX_NEW_TOKENS"
    elif [ "$MMLU_SCORING" = "sglang-bench-gen" ]; then
      run_bench_mmlu "$case_dir"
    else
      run_native_mmlu "$case_dir"
    fi
  fi
  if [ "$RUN_GSM8K" = "1" ]; then
    if [ "$GEN_PROMPT_MODE" = "simple-chat" ]; then
      run_simple_eval "$case_dir" gsm8k "$GSM8K_EXAMPLES" 1024
    else
      run_native_gsm8k "$case_dir"
    fi
  fi
  if [ "$RUN_GPQA" = "1" ]; then
    if [ "$GEN_PROMPT_MODE" = "simple-chat" ]; then
      run_simple_eval "$case_dir" gpqa "$GPQA_EXAMPLES" 1024
    else
      run_native_gpqa "$case_dir"
    fi
  fi

  if [ "$RUN_PREFILL_BENCH" = "1" ]; then
    local stream_args=()
    if [ "$PERF_DISABLE_STREAM" = "1" ]; then
      stream_args+=(--disable-stream)
    fi
    prepare_perf_dataset "$PERF_DATASET_NAME"
    mapfile -t dataset_args < <(perf_dataset_args "$PERF_DATASET_NAME")
    local last_run_file=""
    local last_run_log=""
    for perf_run in $(seq 1 "$PERF_MEASURE_REPEATS"); do
      last_run_file="$case_dir/prefill_bench_run${perf_run}.jsonl"
      last_run_log="$case_dir/prefill_bench_run${perf_run}.log"
      local profile_args=()
      if [ "$PERF_ENABLE_PROFILE" = "1" ]; then
        read -r -a profile_activities <<<"$PERF_PROFILE_ACTIVITIES"
        profile_args+=(
          --profile
          --profile-activities "${profile_activities[@]}"
          --profile-output-dir "$case_dir/profile_run${perf_run}"
          --profile-prefix "${name}-run${perf_run}"
        )
        if [ -n "$PERF_PROFILE_START_STEP" ] && [ "$PERF_PROFILE_START_STEP" != "0" ]; then
          profile_args+=(--profile-start-step "$PERF_PROFILE_START_STEP")
        fi
        if [ -n "$PERF_PROFILE_STEPS" ] && [ "$PERF_PROFILE_STEPS" != "0" ]; then
          profile_args+=(--profile-steps "$PERF_PROFILE_STEPS")
        fi
      fi
      echo "=== PREFILL_BENCH_BEGIN $name run=${perf_run}/${PERF_MEASURE_REPEATS} $(date) ==="
      timeout --kill-after=60s "${PERF_BENCH_TIMEOUT_SEC}s" python -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 \
        --port "$PORT" \
        --model "$MODEL" \
        --tokenizer "$MODEL" \
        "${dataset_args[@]}" \
        --num-prompts "$PERF_NUM_PROMPTS" \
        --request-rate inf \
        --max-concurrency "$PERF_CONCURRENCY" \
        --warmup-requests "$PERF_WARMUP_REQUESTS" \
        --ready-check-timeout-sec 0 \
        --output-file "$last_run_file" \
        --disable-tqdm \
        "${stream_args[@]}" \
        "${profile_args[@]}" \
        >"$last_run_log" 2>&1
      echo "=== PREFILL_BENCH_DONE $name run=${perf_run}/${PERF_MEASURE_REPEATS} $(date) ==="
      tail -n 80 "$last_run_log" || true
    done
    cp "$last_run_file" "$case_dir/prefill_bench.jsonl"
    cp "$last_run_log" "$case_dir/prefill_bench.log"
  fi

  python - <<PY
import json, pathlib, re
case_dir = pathlib.Path("$case_dir")
server_log = (case_dir / "server.log").read_text(errors="replace")
server_args_match = re.search(r"server_args=ServerArgs\((.*?)\)", server_log, re.S)
effective_kv_cache_dtype = "$KV_CACHE_DTYPE"
if server_args_match:
    kv_match = re.search(r"kv_cache_dtype='([^']+)'", server_args_match.group(1))
    if kv_match:
        effective_kv_cache_dtype = kv_match.group(1)
summary = {
    "case": "$name",
    "model": "$MODEL",
    "waterfill": bool(int("$waterfill")),
    "disable_static_waterfill": bool(int("${SGLANG_DISABLE_STATIC_WATERFILL:-0}")),
    "waterfill_log_stats_interval": int("${SGLANG_WATERFILL_LOG_STATS_INTERVAL:-0}"),
    "waterfill_force_local_shared": bool(int("$case_force_local_shared")),
    "waterfill_static_allow_all_ranks": bool(int("${SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS:-1}")),
    "waterfill_local_pref_numer": int("${SGLANG_WATERFILL_LOCAL_PREF_NUMER:-11}"),
    "waterfill_local_pref_denom": int("${SGLANG_WATERFILL_LOCAL_PREF_DENOM:-10}"),
    "waterfill_remote_cost_tokens": int("${SGLANG_WATERFILL_REMOTE_COST_TOKENS:-0}"),
    "waterfill_source_aware_static_load": bool(int("${SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD:-0}")),
    "waterfill_static_block_load_m": int("${SGLANG_WATERFILL_STATIC_BLOCK_LOAD_M:-0}"),
    "waterfill_reuse_topk_buffer": bool(int("${SGLANG_WATERFILL_REUSE_TOPK_BUFFER:-0}")),
    "waterfill_reuse_topk_buffer_cache_size": int("${SGLANG_WATERFILL_REUSE_TOPK_BUFFER_CACHE_SIZE:-8}"),
    "waterfill_min_batch_for_balance": int("${SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE:-64}"),
    "waterfill_fuse_megamoe_predispatch": (
        None
        if "$case_fuse_megamoe_predispatch" == ""
        else bool(int("$case_fuse_megamoe_predispatch"))
    ),
    "waterfill_one_way_remote_shared": (
        None
        if "${SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED:-}" == ""
        else bool(int("${SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED:-0}"))
    ),
    "waterfill_rank2_single_block_count_max_tokens": int("${SGLANG_WATERFILL_RANK2_SINGLE_BLOCK_COUNT_MAX_TOKENS:-512}"),
    "waterfill_shared_replicas_per_rank": (
        None
        if "${SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK:-}" == ""
        else int("${SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK}")
    ),
    "mega_moe_log_topk_stats_interval": int("${SGLANG_MEGA_MOE_LOG_TOPK_STATS_INTERVAL:-0}"),
    "mega_moe_log_timing_interval": int("${SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL:-0}"),
    "mega_moe_cap": int("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK}"),
    "mega_moe_cap_buckets": "${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS}",
    "mega_moe_preinit_all_cap_buckets": bool(int("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_PREINIT_ALL_CAP_BUCKETS}")),
    "mega_moe_cap_bucket_min_free_gb": float("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_CAP_BUCKET_MIN_FREE_GB}"),
    "mega_moe_use_fp4_acts": bool(int("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS:-0}")),
    "mega_moe_use_mxf4_kind": bool(int("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND:-0}")),
    "max_prefill_tokens": int("$MAX_PREFILL_TOKENS"),
    "chunked_prefill_size": int("$CHUNKED_PREFILL_SIZE"),
    "requested_kv_cache_dtype": "$KV_CACHE_DTYPE",
    "kv_cache_dtype": effective_kv_cache_dtype,
    "moe_a2a_backend": "$MOE_A2A_BACKEND",
    "moe_runner_backend": "${MOE_RUNNER_BACKEND:-auto}",
    "deepep_config": "$DEEPEP_CONFIG",
    "ep_num_redundant_experts": int("$EP_NUM_REDUNDANT_EXPERTS"),
    "ep_dispatch_algorithm": "$EP_DISPATCH_ALGORITHM",
    "init_expert_location": "$INIT_EXPERT_LOCATION",
    "enable_eplb": bool(int("$ENABLE_EPLB")),
    "eplb_algorithm": "$EPLB_ALGORITHM",
    "eplb_rebalance_num_iterations": "$EPLB_REBALANCE_NUM_ITERATIONS",
    "eplb_rebalance_layers_per_chunk": "$EPLB_REBALANCE_LAYERS_PER_CHUNK",
    "eplb_min_rebalancing_utilization_threshold": "$EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD",
    "expert_distribution_recorder_mode": "$EXPERT_DISTRIBUTION_RECORDER_MODE",
    "expert_distribution_recorder_buffer_size": "$EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE",
    "extra_server_args": "$EXTRA_SERVER_ARGS",
    "case_extra_server_args": "$case_extra_server_args",
    "disable_shared_experts_fusion": bool(int("$case_disable_shared_experts_fusion")),
    "skip_server_warmup": bool(int("$SKIP_SERVER_WARMUP")),
    "tp_size": int("$TP_SIZE"),
    "dp_size": int("$DP_SIZE"),
    "moe_dense_tp_size": int("$MOE_DENSE_TP_SIZE"),
    "port": int("$PORT"),
    "case_port_stride": int("$CASE_PORT_STRIDE"),
    "enable_dp_attention": bool(int("$ENABLE_DP_ATTENTION")),
    "allow_ready_log_health": bool(int("$ALLOW_READY_LOG_HEALTH")),
    "wait_health_mode": "$WAIT_HEALTH_MODE",
    "ready_log_stabilize_seconds": int("$READY_LOG_STABILIZE_SECONDS"),
    "run_prefill_prewarm": bool(int("$RUN_PREFILL_PREWARM")),
    "perf_prewarm_prompts": int("$PERF_PREWARM_PROMPTS"),
    "perf_prewarm_concurrency": int("$PERF_PREWARM_CONCURRENCY"),
    "perf_measure_repeats": int("$PERF_MEASURE_REPEATS"),
    "perf_bench_timeout_sec": int("$PERF_BENCH_TIMEOUT_SEC"),
    "perf_warmup_requests": int("$PERF_WARMUP_REQUESTS"),
    "perf_random_range_ratio": float("$PERF_RANDOM_RANGE_RATIO"),
    "perf_disable_stream": bool(int("$PERF_DISABLE_STREAM")),
    "perf_dataset_name": "$PERF_DATASET_NAME",
    "perf_dataset_path": "$PERF_DATASET_PATH",
    "perf_sharegpt_output_len": int("$PERF_SHAREGPT_OUTPUT_LEN"),
    "perf_mmlu_num_examples": int("$PERF_MMLU_NUM_EXAMPLES"),
    "perf_mmlu_nsub": int("$PERF_MMLU_NSUB"),
    "perf_mmlu_ntrain": int("$PERF_MMLU_NTRAIN"),
    "perf_mmlu_prompt_mode": "$PERF_MMLU_PROMPT_MODE",
    "mmlu_prompt_mode": "$MMLU_PROMPT_MODE",
    "gen_prompt_mode": "$GEN_PROMPT_MODE",
    "mmlu_scoring": "$MMLU_SCORING",
    "mmlu_max_new_tokens": int("$MMLU_MAX_NEW_TOKENS"),
    "mmlu_choice_prefix": "$MMLU_CHOICE_PREFIX",
    "mmlu_ntrain": int("$MMLU_NTRAIN"),
    "run_eval_api": "$RUN_EVAL_API",
    "fallback_triton_config_lines": len(re.findall(r"Using default MoE kernel config", server_log)),
    "scheduler_exceptions": len(re.findall(r"Scheduler hit an exception", server_log)),
}
for eval_name in ("mmlu", "gsm8k"):
    p = case_dir / f"{eval_name}_result.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    summary[eval_name] = rows[-1] if rows else None
p = case_dir / "gpqa_result.json"
summary["gpqa"] = json.loads(p.read_text()) if p.exists() else None
bench_path = case_dir / "prefill_bench.jsonl"
bench_lines = [json.loads(x) for x in bench_path.read_text().splitlines() if x.strip()] if bench_path.exists() else []
summary["prefill_bench"] = bench_lines[-1] if bench_lines else None
bench_runs = []
for p in sorted(case_dir.glob("prefill_bench_run*.jsonl")):
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    if rows:
        bench_runs.append({"file": p.name, **rows[-1]})
summary["prefill_bench_runs"] = bench_runs
(case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print("SUMMARY_JSON", case_dir / "summary.json")
print(json.dumps(summary, indent=2, sort_keys=True)[:5000])
PY

  grep -Ei 'Mega MoE|enable_deepep_waterfill|Prepared .*waterfill|Shared experts fusion|Using default MoE kernel config|Scheduler hit|Traceback|ERROR|out of memory|OOM|Prefill batch' "$server_log" | tail -220 || true

  cleanup_case
  trap - RETURN
  echo "=== END_CASE $name $(date) ==="
}

CASE_ORDER=${CASE_ORDER//:/,}
IFS=',' read -r -a case_order <<<"$CASE_ORDER"
for case_name in "${case_order[@]}"; do
  case "$case_name" in
    unfused)
      run_case unfused 0 "${SGLANG_WATERFILL_FORCE_LOCAL_SHARED:-0}" "" 1
      ;;
    fused)
      run_case fused 0 "${SGLANG_WATERFILL_FORCE_LOCAL_SHARED:-0}" "--enforce-shared-experts-fusion" 0
      ;;
    fused_waterfill)
      run_case fused_waterfill 1 0 "--enforce-shared-experts-fusion" 0
      ;;
    no_waterfill)
      if [ "$RUN_NO_WATERFILL" = "1" ]; then
        run_case no_waterfill 0
      fi
      ;;
    waterfill)
      if [ "$RUN_WATERFILL" = "1" ]; then
        run_case waterfill 1 0
      fi
      ;;
    waterfill_local)
      if [ "$RUN_WATERFILL" = "1" ]; then
        run_case waterfill_local 1 1
      fi
      ;;
    waterfill_fusedpredispatch)
      if [ "$RUN_WATERFILL" = "1" ]; then
        run_case waterfill_fusedpredispatch 1 0 "" "$DISABLE_SHARED_EXPERTS_FUSION" 1
      fi
      ;;
    *)
      echo "UNKNOWN_CASE_IN_CASE_ORDER $case_name"
      exit 2
      ;;
  esac
done

python - <<'PY'
import json, pathlib, os

root = pathlib.Path("/host_scratch") / f"megamoe_cap_eval_{os.environ['SLURM_JOB_ID']}"
cases = []
case_order = [x for x in os.environ.get("CASE_ORDER", "no_waterfill,waterfill").replace(":", ",").split(",") if x]
for name in case_order:
    p = root / name / "summary.json"
    if p.exists():
        cases.append(json.loads(p.read_text()))

summary = {"root": str(root), "cases": cases}
case_by_name = {case.get("case"): case for case in cases}
if "no_waterfill" in case_by_name and "waterfill" in case_by_name:
    base, wf = case_by_name["no_waterfill"], case_by_name["waterfill"]
    for eval_name in ("mmlu", "gsm8k", "gpqa"):
        a = (base.get(eval_name) or {}).get("accuracy", (base.get(eval_name) or {}).get("score"))
        b = (wf.get(eval_name) or {}).get("accuracy", (wf.get(eval_name) or {}).get("score"))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            summary[f"{eval_name}_accuracy_delta_waterfill_minus_no_waterfill"] = b - a
    for key in ("request_throughput", "input_throughput", "total_token_throughput", "median_ttft_ms", "mean_ttft_ms"):
        a = (base.get("prefill_bench") or {}).get(key)
        b = (wf.get("prefill_bench") or {}).get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            summary[f"prefill_bench_{key}_waterfill"] = b
            summary[f"prefill_bench_{key}_no_waterfill"] = a
            if a != 0 and "throughput" in key:
                summary[f"prefill_bench_{key}_speedup"] = b / a
if "waterfill_local" in case_by_name and "waterfill" in case_by_name:
    base, wf = case_by_name["waterfill_local"], case_by_name["waterfill"]
    for key in ("request_throughput", "input_throughput", "total_token_throughput", "median_ttft_ms", "mean_ttft_ms"):
        a = (base.get("prefill_bench") or {}).get(key)
        b = (wf.get("prefill_bench") or {}).get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            summary[f"prefill_bench_{key}_waterfill"] = b
            summary[f"prefill_bench_{key}_waterfill_local"] = a
            if a != 0 and "throughput" in key:
                summary[f"prefill_bench_{key}_waterfill_vs_local_speedup"] = b / a

(root / "compare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print("COMPARE_SUMMARY", root / "compare_summary.json")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
