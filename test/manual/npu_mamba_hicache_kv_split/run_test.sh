#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

OUT=${1:?usage: bash run_test.sh <output_dir>}
mkdir -p "$OUT"

trap 'kill -TERM "${SERVER_PID:-}" 2>/dev/null || true
echo "输出目录: $OUT"
echo "  服务日志: $OUT/server.log"
echo "  评测日志: $OUT/eval.log"
echo "  测试报告: $OUT/report.md"' EXIT

bash "$HERE/run_server.sh" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not ready"; exit 1; }

set +e
python3 -u "$HERE/run_case.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --target-prompt-chars "$TARGET_PROMPT_CHARS" \
  --target-new-tokens "$TARGET_NEW_TOKENS" \
  --num-filler "$NUM_FILLER" \
  --filler-prompt-chars "$FILLER_PROMPT_CHARS" \
  --filler-new-tokens "$FILLER_NEW_TOKENS" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --output "$OUT" 2>&1 | tee "$OUT/eval.log"
EVAL_RC=${PIPESTATUS[0]}
set -e

python3 -u "$HERE/report.py" "$OUT" \
  --sglang-root "$SGLANG_ROOT" \
  --model "$MODEL_PATH" \
  --npu-devices "$NPU_DEVICES" \
  --port "$PORT" 2>&1 | tee "$OUT/report.log"
exit "$EVAL_RC"
