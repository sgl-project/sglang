#!/usr/bin/env bash
# Launch one SGLang server per spec, run driver.py against it, stop it, repeat.
#
# Usage: run_bench.sh OUT_DIR SPEC [SPEC ...]
#   SPEC = "LABEL|SGLANG_CHECKOUT|EXTRA_SERVER_ARGS"
#     LABEL             name of the result subdirectory, e.g. main1, fix-mixed
#     SGLANG_CHECKOUT   path to an sglang checkout; its python/ is put first on PYTHONPATH
#     EXTRA_SERVER_ARGS appended to the common server arguments (may be empty)
#
# Environment (defaults in brackets):
#   MODEL           [Qwen/Qwen3.8-27B-FP8]  Hugging Face id or local path
#   MODEL_REVISION  [017b9c7af6b5689d5dd426a76e0bc077eb5ca20a]  set to "" for a local path
#   PYTHON          [python3]   interpreter with sglang's dependencies installed
#   PORT            [31023]
#   SERVER_TIMEOUT  [1800]      seconds to wait for /health (includes model download)
#   CUDA_VISIBLE_DEVICES        two GPUs, e.g. 0,1
#
# Exits non-zero on the first server or driver failure; the server is always stopped.
set -euo pipefail

if (( $# < 2 )); then
  sed -n '2,20p' "$0"
  exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
out=$1
shift
MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}
MODEL_REVISION=${MODEL_REVISION-017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}
PYTHON=${PYTHON:-python3}
PORT=${PORT:-31023}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1800}
base_pythonpath=${PYTHONPATH:-}
mkdir -p "$out"

# Common server arguments shared by every run in the PR tables.
common_args=(
  --host 127.0.0.1 --port "$PORT"
  --model-path "$MODEL"
  --tp 2 --attn-dp-size 2 --context-length 73728
  --attention-backend fa3 --mm-attention-backend sdpa --kv-cache-dtype fp8_e4m3
  --chunked-prefill-size 8192 --disable-radix-cache --enable-metrics
  --mem-fraction-static 0.80 --load-balance-method total_tokens
)
if [[ -n "$MODEL_REVISION" ]]; then
  common_args+=(--revision "$MODEL_REVISION")
fi

server_pid=
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for spec in "$@"; do
  IFS='|' read -r label src extra <<< "$spec"
  if [[ -z "$label" || ! -d "$src/python/sglang" ]]; then
    echo "Bad spec '$spec': need LABEL|SGLANG_CHECKOUT|EXTRA_ARGS" >&2
    exit 2
  fi
  if [[ -e "$out/$label" ]]; then
    echo "$out/$label already exists; use a new OUT_DIR or LABEL" >&2
    exit 2
  fi
  commit=$(git -C "$src" rev-parse HEAD)
  dirty=$(git -C "$src" status --porcelain --untracked-files=no | wc -l)
  echo "[$(date -Is)] $label: sglang $commit (uncommitted files: $dirty) extra='$extra'"
  printf '%s\n' "$commit" > "$out/$label.commit"

  # shellcheck disable=SC2206  # extra args are intentionally word-split
  extra_args=($extra)
  PYTHONPATH="$src/python${base_pythonpath:+:$base_pythonpath}" \
    setsid "$PYTHON" -m sglang.launch_server "${common_args[@]}" "${extra_args[@]}" \
    > "$out/$label-server.log" 2>&1 &
  server_pid=$!

  deadline=$((SECONDS + SERVER_TIMEOUT))
  until curl -fsS --max-time 4 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[$(date -Is)] $label: server exited during startup" >&2
      tail -n 40 "$out/$label-server.log" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[$(date -Is)] $label: server not healthy after ${SERVER_TIMEOUT}s" >&2
      tail -n 40 "$out/$label-server.log" >&2
      exit 1
    fi
    sleep 3
  done
  echo "[$(date -Is)] $label: healthy"

  "$PYTHON" -u "$here/driver.py" --url "http://127.0.0.1:$PORT" --out "$out/$label"
  stop_server
  echo "[$(date -Is)] $label: done"
done
echo "[$(date -Is)] all runs finished"
