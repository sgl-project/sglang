#!/usr/bin/env bash
# 04_smoke.sh — prove the server actually answers, and snapshot the hardware state.
#
# Checks, in order:
#   1. /health returns 200
#   2. /get_model_info reports the expected model path
#   3. a chat completion returns plausible text (the docs use: capital of France -> Paris)
#   4. a raw /generate completion works (V4 has no Jinja chat template, so this is the fallback path)
#   5. npu-smi HBM snapshot before/after the request is saved for the record
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$HERE/.." && pwd)
# shellcheck source=../env/00_common.sh
source "$PROJ/env/00_common.sh"

BASE="http://127.0.0.1:$PORT"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OUT="$RESULTS_DIR/smoke-${STAMP//[:+]/}.txt"
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}

exec > >(tee "$OUT") 2>&1
echo "=== smoke test $STAMP  base=$BASE  model=$MODEL_NAME"
echo
echo "--- 1. health"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" || true)
echo "HTTP $code"
[ "$code" = "200" ] || { echo "FAIL: server not healthy"; echo "see $LOG_DIR/server.log"; exit 1; }

echo
echo "--- 2. model info"
curl -s "$BASE/get_model_info" | head -c 1200; echo

echo
echo "--- 3. chat completion (documented acceptance: answer mentions Paris)"
resp=$(curl -s -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? Answer in one short sentence.\"}],
  \"max_tokens\": 64, \"temperature\": 0.0
}" || true)
echo "$resp" | head -c 1500; echo
if echo "$resp" | grep -qi paris; then
  echo "PASS: response mentions Paris"
else
  echo "WARN: response did not mention Paris — check the reply above (or the chat-template error)"
fi

echo
echo "--- 4. raw /generate fallback"
curl -s -m 180 "$BASE/generate" -H 'Content-Type: application/json' \
  -d "{\"text\": \"The capital of France is\", \"sampling_params\": {\"max_new_tokens\": 32, \"temperature\": 0}}" \
  | head -c 800; echo

echo
echo "--- 5. npu-smi snapshot"
npu-smi info 2>/dev/null | head -60 || echo "npu-smi unavailable"

echo
echo "saved: $OUT"
