#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
OUT=${1:?usage: bash run_test.sh <output_dir>}
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
rm -f "$OUT/layer_first.json" "$OUT/page_first.json" \
  "$OUT/layer_first.exit_code" "$OUT/page_first.exit_code" "$OUT/report.md"
SERVER_PID=""

cleanup() {
  set +e
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  python3 -u "$HERE/report.py" "$OUT" > "$OUT/report.log" 2>&1
  echo "输出目录: $OUT"
  for layout in layer_first page_first; do
    echo "  $layout 服务日志: $OUT/server.$layout.log"
    echo "  $layout 评测日志: $OUT/eval.$layout.log"
    echo "  $layout 结果: $OUT/$layout.json"
  done
  echo "  测试报告: $OUT/report.md"
  echo "  报告日志: $OUT/report.log"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

overall=0
for layout in layer_first page_first; do
  echo "启动 $layout 服务"
  bash "$HERE/run_server.sh" "$layout" > "$OUT/server.$layout.log" 2>&1 &
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
    echo "运行 $layout 请求用例"
    set +e
    python3 -u "$HERE/run_case.py" --layout "$layout" \
      --output "$OUT/$layout.json" > "$OUT/eval.$layout.log" 2>&1
    case_rc=$?
    set -e
  else
    echo "$layout 服务未就绪，查看 server.$layout.log" | tee "$OUT/eval.$layout.log"
    case_rc=1
  fi
  echo "$case_rc" > "$OUT/$layout.exit_code"
  if ((case_rc)); then overall=1; fi
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
done

if ((overall)); then echo "测试失败"; else echo "测试通过"; fi
exit "$overall"
