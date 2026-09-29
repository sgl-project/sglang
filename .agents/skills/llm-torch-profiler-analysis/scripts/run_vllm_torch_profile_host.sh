#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_vllm_torch_profile_host.sh \
    --model Qwen/Qwen3-8B \
    --run-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example_vllm_formal \
    --port 31088 \
    --gpus 1

  run_vllm_torch_profile_host.sh \
    --model openai/gpt-oss-20b \
    --run-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example_vllm_4gpu \
    --port 31088 \
    --gpus 2,3,4,5 \
    --tensor-parallel-size 4

Options:
  --model TEXT                   Hugging Face model id.
  --run-dir PATH                Shared /data directory for logs and traces.
  --port INT                    Host port for vllm serve.
  --gpus TEXT                   CUDA_VISIBLE_DEVICES value, for example 1 or 2,3,4,5.
  --gpu TEXT                    Alias for --gpus.
  --image TEXT                  Container image.
  --hf-cache PATH               Host Hugging Face cache path.
  --gpu-memory-util FLOAT       vLLM --gpu-memory-utilization.
  --max-model-len INT           vLLM --max-model-len.
    --tensor-parallel-size INT    vLLM --tensor-parallel-size. Defaults to the visible GPU count.
    --profiler-active-iterations INT
                                 Torch-profiler active iterations.
    --enforce-eager               Launch vLLM with --enforce-eager for mapping traces.
  --trust-remote-code           Pass --trust-remote-code.
  --request-max-tokens INT      Generation length for the probe request.
  --prompt TEXT                 Probe prompt.
  --warmup-steps INT            Warmup steps before profiling. Defaults to 10.
  --profile-workload TEXT       legacy|prefill|decode|both. Defaults to both.
  --prefill-input-len INT       Synthetic prefill prompt length. Defaults to 4090.
  --prefill-output-len INT      Synthetic prefill output length. Defaults to 1.
  --decode-input-len INT        Synthetic decode prompt length. Defaults to 1.
  --decode-output-len INT       Synthetic decode output length. Defaults to 2048.
  --container-name TEXT         Override container name.
  --help                        Show this message.

Environment:
  HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set.

Notes:
  - Run this on the H100 host, not inside `sglang_bbuf`.
  - This uses the vLLM torch-profiler flow: `--profiler-config`, then POST
    `/start_profile` and `/stop_profile`.
  - Default capture is two labeled profiles: prefill 4090->1 and decode 1->2048.
  - Current vLLM profiler config already defaults `torch_profiler_with_stack=true`.
  - A small benchmark summary is written after profiling.
EOF
}

IMAGE="vllm/vllm-openai:latest"
HF_CACHE="/data/.cache/huggingface"
GPU_MEMORY_UTIL=0.90
MAX_MODEL_LEN=4096
TP_SIZE=""
ENFORCE_EAGER=0
TRUST_REMOTE_CODE=0
REQUEST_MAX_TOKENS=12
PROFILER_ACTIVE_ITERATIONS=5
PROMPT="Explain the difference between CUDA graph mode and eager mode in two sentences."
WARMUP_STEPS=10
PROFILE_WORKLOAD="both"
PREFILL_INPUT_LEN=4090
PREFILL_OUTPUT_LEN=1
DECODE_INPUT_LEN=1
DECODE_OUTPUT_LEN=2048
CONTAINER_NAME=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
RUN_DIR=""
PORT=""
GPUS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --gpu)
      GPUS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --hf-cache)
      HF_CACHE="$2"
      shift 2
      ;;
    --gpu-memory-util)
      GPU_MEMORY_UTIL="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TP_SIZE="$2"
      shift 2
      ;;
    --profiler-active-iterations)
      PROFILER_ACTIVE_ITERATIONS="$2"
      shift 2
      ;;
    --enforce-eager)
      ENFORCE_EAGER=1
      shift
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1
      shift
      ;;
    --request-max-tokens)
      REQUEST_MAX_TOKENS="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --profile-workload)
      PROFILE_WORKLOAD="$2"
      shift 2
      ;;
    --prefill-input-len)
      PREFILL_INPUT_LEN="$2"
      shift 2
      ;;
    --prefill-output-len)
      PREFILL_OUTPUT_LEN="$2"
      shift 2
      ;;
    --decode-input-len)
      DECODE_INPUT_LEN="$2"
      shift 2
      ;;
    --decode-output-len)
      DECODE_OUTPUT_LEN="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running." >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ -z "$MODEL" || -z "$RUN_DIR" || -z "$PORT" || -z "$GPUS" ]]; then
  usage >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "Could not parse --gpus: $GPUS" >&2
  exit 2
fi
if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$GPU_COUNT"
fi
if (( TP_SIZE < 1 || TP_SIZE > GPU_COUNT )); then
  echo "--tensor-parallel-size must be between 1 and the visible GPU count ($GPU_COUNT)." >&2
  exit 2
fi
if (( PROFILER_ACTIVE_ITERATIONS < 1 )); then
  echo "--profiler-active-iterations must be >= 1." >&2
  exit 2
fi

PROFILE_DIR="$RUN_DIR/vllm_profile"
LOG_PATH="$RUN_DIR/server.log"
ANALYSIS_PATH="$RUN_DIR/analysis_vllm_live.txt"
BENCHMARK_PATH="$RUN_DIR/benchmark_vllm.json"

if [[ -z "$CONTAINER_NAME" ]]; then
  model_slug="${MODEL##*/}"
  model_slug="${model_slug//\//-}"
  model_slug="${model_slug//./-}"
  model_slug="${model_slug//_/-}"
  gpu_slug="${GPUS//,/-}"
  CONTAINER_NAME="vllm-${model_slug}-g${gpu_slug}-p${PORT}"
  if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
    CONTAINER_NAME="${CONTAINER_NAME}-eager"
  fi
fi

docker exec sglang_bbuf bash -lc "mkdir -p '$PROFILE_DIR'"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

profiler_config=$(python3 - <<PY
import json
print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": ${PROFILE_DIR@Q},
    "active_iterations": int(${PROFILER_ACTIVE_ITERATIONS@Q}),
}))
PY
)

docker_args=(
  run -d --rm
  --name "$CONTAINER_NAME"
  --gpus all
  --ipc=host
  --network host
  -e "CUDA_VISIBLE_DEVICES=$GPUS"
  -e "HF_TOKEN=$HF_TOKEN"
  -e "HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN"
  -e "VLLM_RPC_TIMEOUT=1800000"
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$RUN_DIR:$RUN_DIR"
)

docker_cmd=(
  "$IMAGE"
  "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --profiler-config "$profiler_config"
)

if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
  docker_cmd+=(--enforce-eager)
fi
if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
  docker_cmd+=(--trust-remote-code)
fi

docker "${docker_args[@]}" "${docker_cmd[@]}" >/dev/null
cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  echo "Server did not become ready on port ${PORT}. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -n 120 >&2 || true
  exit 1
fi

python3 "$SCRIPT_DIR/analyze_llm_torch_profile.py" \
  --framework vllm \
  --url "http://127.0.0.1:${PORT}" \
  --output-dir "$PROFILE_DIR" \
  --num-steps "$PROFILER_ACTIVE_ITERATIONS" \
  --warmup-steps "$WARMUP_STEPS" \
  --probe-requests 1 \
  --no-profile-by-stage \
  --profile-workload "$PROFILE_WORKLOAD" \
  --probe-prompt "$PROMPT" \
  --probe-max-new-tokens "$REQUEST_MAX_TOKENS" \
  --prefill-input-len "$PREFILL_INPUT_LEN" \
  --prefill-output-len "$PREFILL_OUTPUT_LEN" \
  --decode-input-len "$DECODE_INPUT_LEN" \
  --decode-output-len "$DECODE_OUTPUT_LEN" \
  > "$ANALYSIS_PATH"

profile_found=0
for _ in $(seq 1 240); do
  if find "$PROFILE_DIR" -type f \( -name '*.pt.trace.json' -o -name '*.pt.trace.json.gz' -o -name '*.trace.json' -o -name '*.trace.json.gz' \) | grep -q .; then
    profile_found=1
    break
  fi
  sleep 2
done
if [[ "$profile_found" -ne 1 ]]; then
  echo "No vLLM profiler traces appeared under $PROFILE_DIR" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -n 120 >&2 || true
  exit 1
fi

python3 "$SCRIPT_DIR/probe_llm_server.py" \
  --framework vllm \
  --url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  | docker exec -i sglang_bbuf bash -lc "cat > '$BENCHMARK_PATH'" >/dev/null

docker logs "$CONTAINER_NAME" 2>&1 | docker exec -i sglang_bbuf bash -lc "cat > '$LOG_PATH'" || true
sed -n '1,240p' "$ANALYSIS_PATH"
echo "PROFILE_DIR=$PROFILE_DIR"
echo "LOG_PATH=$LOG_PATH"
echo "ANALYSIS_PATH=$ANALYSIS_PATH"
echo "BENCHMARK_PATH=$BENCHMARK_PATH"
