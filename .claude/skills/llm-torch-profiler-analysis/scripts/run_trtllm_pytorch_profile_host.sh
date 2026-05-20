#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_trtllm_pytorch_profile_host.sh \
    --model Qwen/Qwen3-8B \
    --run-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example \
    --stage prefill \
    --port 32188 \
    --gpus 0

  run_trtllm_pytorch_profile_host.sh \
    --model openai/gpt-oss-20b \
    --run-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example_4gpu \
    --stage prefill \
    --port 32188 \
    --gpus 2,3,4,5 \
    --tp-size 4

Options:
  --model TEXT                     Hugging Face model id.
  --run-dir PATH                  Shared /data run directory for logs and traces.
  --stage prefill|decode          Capture window. Prefill profiles 4090->1 by
                                  default; decode profiles 1->2048 by default.
  --port INT                      Host port for trtllm-serve.
  --gpus TEXT                     CUDA_VISIBLE_DEVICES value, for example 0 or 2,3,4,5.
  --gpu TEXT                      Alias for --gpus.
  --tp-size INT                   Tensor parallel size. Defaults to the visible GPU count.
  --image TEXT                    Container image.
  --shared-root PATH              Shared validation root mounted into the container.
  --hf-cache PATH                 Host Hugging Face cache path.
  --override-py-executor PATH     Optional py_executor.py override path.
  --disable-cudagraph             Generate/use a YAML override with cuda_graph_config: null.
  --input-len INT                 Synthetic prompt length for this stage.
                                  Defaults: prefill 4090, decode 1.
  --request-max-tokens INT        Generation length for this stage.
                                  Defaults: prefill 1, decode 2048.
  --output-len INT                Alias for --request-max-tokens.
  --prompt TEXT                   Probe prompt. Defaults to a synthetic prompt
                                  sized by --input-len.
  --warmup-steps INT              Warmup steps before the profiler window. Defaults to 10.
  --active-steps INT              Active profiler steps to capture. Defaults to 5.
  --max-seq-len INT               Serve max sequence length.
  --kv-fraction FLOAT             KV cache free GPU memory fraction.
  --container-name TEXT           Override container name.
  --trust-remote-code             Pass --trust_remote_code to trtllm-serve.
  --help                          Show this message.

Environment:
  HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set.

Notes:
  - Run this on the H100 host, not inside `sglang_bbuf`.
  - It always pins TensorRT-LLM to `--backend pytorch`.
  - The default image tag is floating; record the resolved TensorRT-LLM version
    in the run manifest and pass --image for reproducible validation.
  - Profiling uses `TLLM_PROFILE_START_STOP` and `TLLM_TORCH_PROFILE_TRACE`.
  - For Python-location recovery, prefer a `py_executor.py` override with `with_stack=True`.
  - A small benchmark summary is written after the trace is emitted.
EOF
}

IMAGE="nvcr.io/nvidia/tensorrt-llm/release:latest"
SHARED_ROOT="/data/bbuf/validate/unified_llm_profiler_skill"
HF_CACHE="/data/.cache/huggingface"
OVERRIDE_PY_EXECUTOR=""
DISABLE_CUDAGRAPH=0
REQUEST_MAX_TOKENS=""
INPUT_LEN=""
PROMPT=""
WARMUP_STEPS=10
ACTIVE_STEPS=5
MAX_SEQ_LEN=4096
KV_FRACTION=0.85
CONTAINER_NAME=""
TRUST_REMOTE_CODE=0
TP_SIZE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
RUN_DIR=""
STAGE=""
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
    --stage)
      STAGE="$2"
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
    --tp-size)
      TP_SIZE="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --shared-root)
      SHARED_ROOT="$2"
      shift 2
      ;;
    --hf-cache)
      HF_CACHE="$2"
      shift 2
      ;;
    --override-py-executor)
      OVERRIDE_PY_EXECUTOR="$2"
      shift 2
      ;;
    --disable-cudagraph)
      DISABLE_CUDAGRAPH=1
      shift
      ;;
    --input-len)
      INPUT_LEN="$2"
      shift 2
      ;;
    --request-max-tokens)
      REQUEST_MAX_TOKENS="$2"
      shift 2
      ;;
    --output-len)
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
    --active-steps)
      ACTIVE_STEPS="$2"
      shift 2
      ;;
    --max-seq-len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --kv-fraction)
      KV_FRACTION="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1
      shift
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

if [[ -z "$MODEL" || -z "$RUN_DIR" || -z "$STAGE" || -z "$PORT" || -z "$GPUS" ]]; then
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
  echo "--tp-size must be between 1 and the visible GPU count ($GPU_COUNT)." >&2
  exit 2
fi

case "$STAGE" in
  prefill)
    TRACE_PATH="$RUN_DIR/trace-prefill.json"
    LOG_PATH="$RUN_DIR/server-prefill.log"
    BENCHMARK_PATH="$RUN_DIR/benchmark-prefill.json"
    if [[ -z "$INPUT_LEN" ]]; then
      INPUT_LEN=4090
    fi
    if [[ -z "$REQUEST_MAX_TOKENS" ]]; then
      REQUEST_MAX_TOKENS=1
    fi
    ;;
  decode)
    TRACE_PATH="$RUN_DIR/trace-decode.json"
    LOG_PATH="$RUN_DIR/server-decode.log"
    BENCHMARK_PATH="$RUN_DIR/benchmark-decode.json"
    if [[ -z "$INPUT_LEN" ]]; then
      INPUT_LEN=1
    fi
    if [[ -z "$REQUEST_MAX_TOKENS" ]]; then
      REQUEST_MAX_TOKENS=2048
    fi
    ;;
  *)
    echo "--stage must be prefill or decode." >&2
    exit 2
    ;;
esac

if (( WARMUP_STEPS < 0 || ACTIVE_STEPS < 1 )); then
  echo "--warmup-steps must be >= 0 and --active-steps must be >= 1." >&2
  exit 2
fi

case "$STAGE" in
  prefill)
    profile_start=$((WARMUP_STEPS + 1))
    ;;
  decode)
    profile_start=$((WARMUP_STEPS + 2))
    ;;
esac
profile_stop=$((profile_start + ACTIVE_STEPS - 1))
PROFILE_START_STOP="${profile_start}-${profile_stop}"

if [[ -z "$CONTAINER_NAME" ]]; then
  model_slug="${MODEL##*/}"
  model_slug="${model_slug//\//-}"
  model_slug="${model_slug//./-}"
  model_slug="${model_slug//_/-}"
  model_slug="${model_slug// /-}"
  gpu_slug="${GPUS//,/-}"
  CONTAINER_NAME="trtllm-${model_slug}-${STAGE}-g${gpu_slug}-p${PORT}"
fi

EXTRA_LLM_OPTIONS=""
if [[ "$DISABLE_CUDAGRAPH" -eq 1 ]]; then
  EXTRA_CFG_PATH="$SHARED_ROOT/tmp/trt_no_cudagraph.yaml"
  docker exec sglang_bbuf bash -lc "mkdir -p '$(dirname "$EXTRA_CFG_PATH")' && printf 'cuda_graph_config: null\n' > '$EXTRA_CFG_PATH'"
  EXTRA_LLM_OPTIONS="--extra_llm_api_options $EXTRA_CFG_PATH"
fi

docker exec sglang_bbuf bash -lc "mkdir -p '$RUN_DIR'"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker_args=(
  run -d --rm
  --name "$CONTAINER_NAME"
  --gpus all
  --ipc=host
  --network host
  --entrypoint bash
  -e "CUDA_VISIBLE_DEVICES=$GPUS"
  -e "HF_TOKEN=$HF_TOKEN"
  -e "HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN"
  -e "TLLM_PROFILE_START_STOP=$PROFILE_START_STOP"
  -e "TLLM_LLMAPI_ENABLE_NVTX=1"
  -e "TLLM_TORCH_PROFILE_TRACE=$TRACE_PATH"
  -e "RUN_DIR=$RUN_DIR"
  -e "LOG_PATH=$LOG_PATH"
  -e "MODEL_ID=$MODEL"
  -e "SERVE_PORT=$PORT"
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$SHARED_ROOT:$SHARED_ROOT"
)

if [[ -n "$OVERRIDE_PY_EXECUTOR" ]]; then
  docker_args+=(
    -v "$OVERRIDE_PY_EXECUTOR:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/py_executor.py:ro"
  )
fi

trust_remote_code_arg=""
if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
  trust_remote_code_arg="--trust_remote_code"
fi

container_cmd=$(
  cat <<EOF
mkdir -p "$RUN_DIR" && trtllm-serve serve "$MODEL" \
  --backend pytorch \
  --tp_size "$TP_SIZE" \
  --gpus_per_node "$GPU_COUNT" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --kv_cache_free_gpu_memory_fraction "$KV_FRACTION" \
  $trust_remote_code_arg \
  $EXTRA_LLM_OPTIONS \
  > "$LOG_PATH" 2>&1
EOF
)

docker_args+=("$IMAGE" -lc "$container_cmd")
docker "${docker_args[@]}" >/dev/null

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

python3 - <<PY
import json
import sys
import urllib.request

sys.path.insert(0, ${SCRIPT_DIR@Q})
from profile_common import extract_openai_chat_text, synthetic_prompt

prompt = ${PROMPT@Q} or synthetic_prompt(int(${INPUT_LEN@Q}))
stage = ${STAGE@Q}
warmup_steps = int(${WARMUP_STEPS@Q})
active_steps = int(${ACTIVE_STEPS@Q})
request_count = warmup_steps + active_steps if stage == "prefill" else 1

payload = {
    "model": ${MODEL@Q},
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": int(${REQUEST_MAX_TOKENS@Q}),
}
for request_idx in range(request_count):
    req = urllib.request.Request(
        "http://127.0.0.1:${PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
text, source = extract_openai_chat_text(body)
print(text[:400] if text else f"[empty completion; source={source}]")
PY

for _ in $(seq 1 120); do
  if [[ -s "$TRACE_PATH" ]]; then
    break
  fi
  sleep 2
done

if [[ ! -s "$TRACE_PATH" ]]; then
  echo "Trace was not written: $TRACE_PATH" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/probe_llm_server.py" \
  --framework trtllm \
  --url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  | docker exec -i sglang_bbuf bash -lc "cat > '$BENCHMARK_PATH'" >/dev/null

echo "TRACE_PATH=$TRACE_PATH"
echo "LOG_PATH=$LOG_PATH"
echo "BENCHMARK_PATH=$BENCHMARK_PATH"
