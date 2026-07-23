#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:?usage: play.sh RESULTS_DIR [PORT]}"
PORT="${2:-8765}"
[[ -d "${RESULT_DIR}" ]] || {
  echo "result directory not found: ${RESULT_DIR}" >&2
  exit 2
}
RESULT_DIR="$(cd "${RESULT_DIR}" && pwd)"
[[ -f "${RESULT_DIR}/report.json" && -f "${RESULT_DIR}/player/index.html" ]] || {
  echo "missing report/player under ${RESULT_DIR}" >&2
  exit 2
}
python3 - "${RESULT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
with (result_dir / "report.json").open(encoding="utf-8") as source:
    cases = json.load(source)["cases"]
if len(cases) != 10:
    raise SystemExit(f"expected 10 cases in report.json, found {len(cases)}")
missing = [
    str(result_dir / "cases" / case["id"] / name)
    for case in cases
    for name in ("baseline.mp4", "sglang.mp4")
    if not (result_dir / "cases" / case["id"] / name).is_file()
]
if missing:
    raise SystemExit("missing video artifact(s):\n" + "\n".join(missing))
print("Validated 10 cases and 20 video artifacts")
PY
URL="http://127.0.0.1:${PORT}/player/"
echo "Serving ${URL}"
(
  sleep 1
  if command -v open >/dev/null 2>&1; then open "${URL}"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "${URL}"
  else echo "Open ${URL}"
  fi
) &
cd "${RESULT_DIR}"
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
