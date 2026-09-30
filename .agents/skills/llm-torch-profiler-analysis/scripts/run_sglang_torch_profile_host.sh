#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_sglang_torch_profile_host.sh \
    --model Qwen/Qwen3-8B \
    --run-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example_sglang \
    --port 30088 \
    --gpus 0

  run_sglang_torch_profile_host.sh \
    --model openai/gpt-oss-20b \
    --run-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example_sglang_4gpu \
    --port 30088 \
    --gpus 2,3,4,5 \
    --tp-size 4

Options:
  --model TEXT                  Model id or local path for SGLang.
  --run-dir PATH               Shared /data directory for logs and traces.
  --port INT                   Server port.
  --gpus TEXT                  CUDA_VISIBLE_DEVICES value, for example 0 or 2,3,4,5.
  --gpu TEXT                   Alias for --gpus.
  --tp-size INT                Tensor parallel size. Defaults to the visible GPU count.
  --trust-remote-code          Pass --trust-remote-code.
  --mem-fraction FLOAT         SGLang static memory fraction.
  --request-max-tokens INT     Generation length for the probe request.
  --prompt TEXT                Probe prompt.
  --warmup-steps INT           Warmup steps before profiling. Defaults to 10.
  --profile-workload TEXT      legacy|prefill|decode|both. Defaults to both.
  --prefill-input-len INT      Synthetic prefill prompt length. Defaults to 4090.
  --prefill-output-len INT     Synthetic prefill output length. Defaults to 1.
  --decode-input-len INT       Synthetic decode prompt length. Defaults to 1.
  --decode-output-len INT      Synthetic decode output length. Defaults to 2048.
  --repo-dir PATH              SGLang repo path inside `sglang_bbuf`.
  --server-extra TEXT          Extra args appended to launch_server.
  --help                       Show this message.

Notes:
  - Run this on the H100 host. It uses `docker exec sglang_bbuf`.
  - The server is launched first, then the profiler capture runs with
    stage-separated prefill/decode workloads and `--profile-by-stage`.
  - A small benchmark summary is written after profiling.
EOF
}

MODEL=""
RUN_DIR=""
PORT=""
GPUS=""
TP_SIZE=""
TRUST_REMOTE_CODE=0
MEM_FRACTION=0.85
REQUEST_MAX_TOKENS=12
PROMPT="Explain the difference between CUDA graph mode and eager mode in two sentences."
WARMUP_STEPS=10
PROFILE_WORKLOAD="both"
PREFILL_INPUT_LEN=4090
PREFILL_OUTPUT_LEN=1
DECODE_INPUT_LEN=1
DECODE_OUTPUT_LEN=2048
SGLANG_REPO_DIR="${SGLANG_REPO_DIR:-/data/bbuf/repos/sglang}"
SERVER_EXTRA=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    --tp-size)
      TP_SIZE="$2"
      shift 2
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1
      shift
      ;;
    --mem-fraction)
      MEM_FRACTION="$2"
      shift 2
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
    --repo-dir)
      SGLANG_REPO_DIR="$2"
      shift 2
      ;;
    --server-extra)
      SERVER_EXTRA="$2"
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
  echo "--tp-size must be between 1 and the visible GPU count ($GPU_COUNT)." >&2
  exit 2
fi

LOG_PATH="$RUN_DIR/sglang_server.log"
ANALYSIS_PATH="$RUN_DIR/analysis_sglang.txt"
PROFILE_ROOT="$RUN_DIR/sglang_profile_live"
BENCHMARK_PATH="$RUN_DIR/benchmark_sglang.json"
PID_PATH="$RUN_DIR/sglang_server.pid"
LAUNCH_PATTERN="[s]glang.launch_server.*--port $PORT"
SERVER_ARGS="python3 -m sglang.launch_server --model-path \"$MODEL\" --port \"$PORT\" --tp-size \"$TP_SIZE\" --mem-fraction-static \"$MEM_FRACTION\""

if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
  SERVER_ARGS="$SERVER_ARGS --trust-remote-code"
fi
if [[ -n "$SERVER_EXTRA" ]]; then
  SERVER_ARGS="$SERVER_ARGS $SERVER_EXTRA"
fi

docker exec sglang_bbuf bash -lc "mkdir -p '$RUN_DIR' '$PROFILE_ROOT'"
docker exec sglang_bbuf bash -lc "pkill -f '$LAUNCH_PATTERN' >/dev/null 2>&1 || true"
docker exec sglang_bbuf bash -lc "mkdir -p '$RUN_DIR' '$PROFILE_ROOT' && cd '$SGLANG_REPO_DIR' && rm -f '$PID_PATH' && (CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=python nohup $SERVER_ARGS > '$LOG_PATH' 2>&1 < /dev/null & echo \$! > '$PID_PATH')"

cleanup() {
  docker exec sglang_bbuf bash -lc "pkill -f '$LAUNCH_PATTERN' >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
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
  echo "SGLang server did not become ready on port ${PORT}. Recent logs:" >&2
  ssh_log=$(docker exec sglang_bbuf bash -lc "tail -n 120 '$LOG_PATH'" 2>/dev/null || true)
  printf '%s\n' "$ssh_log" >&2
  exit 1
fi

python3 - <<PY
import json
import urllib.request

payload = {
    "text": ${PROMPT@Q},
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": int(${REQUEST_MAX_TOKENS@Q}),
    },
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:${PORT}/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    body = json.loads(resp.read().decode())
text = body.get("text", "")
print(text[:400])
PY

docker exec sglang_bbuf bash -lc "cd '$SCRIPT_DIR' && python3 analyze_llm_torch_profile.py --framework sglang --url http://127.0.0.1:${PORT} --output-dir '$PROFILE_ROOT' --num-steps 5 --warmup-steps '$WARMUP_STEPS' --probe-requests 1 --profile-by-stage --profile-workload '$PROFILE_WORKLOAD' --prefill-input-len '$PREFILL_INPUT_LEN' --prefill-output-len '$PREFILL_OUTPUT_LEN' --decode-input-len '$DECODE_INPUT_LEN' --decode-output-len '$DECODE_OUTPUT_LEN' > '$ANALYSIS_PATH'"
python3 "$SCRIPT_DIR/probe_llm_server.py" \
  --framework sglang \
  --url "http://127.0.0.1:${PORT}" \
  | docker exec -i sglang_bbuf bash -lc "cat > '$BENCHMARK_PATH'" >/dev/null
docker exec sglang_bbuf bash -lc "sed -n '1,240p' '$ANALYSIS_PATH'"
echo "BENCHMARK_PATH=$BENCHMARK_PATH"
