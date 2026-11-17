#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
OUT=${1:?usage: bash run_test.sh <output_dir> [l2|device]}
MODE=${2:-l2}
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
SERVER_PID=""
cleanup() {
  set +e
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  echo "结果目录: $OUT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

overall=0
read -r -a layouts <<< "${LAYOUTS:-layer_first page_first}"
for layout in "${layouts[@]}"; do
  echo "启动 $layout 服务"
  bash "$HERE/run_server.sh" "$layout" "$MODE" > "$OUT/server.$layout.log" 2>&1 &
  SERVER_PID=$!
  ready=0
  for ((i=0; i<STARTUP_TIMEOUT; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if ((ready)); then
    set +e
    python3 -u "$HERE/test_qwen3_8b_gsm8k_l2.py" --layout "$layout" --mode "$MODE" \
      --output "$OUT/$layout.json" > "$OUT/eval.$layout.log" 2>&1
    rc=$?
    set -e
  else
    echo "$layout 服务未就绪" > "$OUT/eval.$layout.log"
    rc=1
  fi
  echo "$rc" > "$OUT/$layout.exit_code"
  if ((rc)); then overall=1; fi
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
done
exit "$overall"
