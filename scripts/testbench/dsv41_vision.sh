#!/usr/bin/env bash
# DSV4.1 vision release check. Unit tests need one GPU. Live tests need the
# DeepSeek-V4.1 server on :11435.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -n "${SGLANG_V100_PYTHON:-}" && -x "${SGLANG_V100_PYTHON}" ]]; then
  PYTHON="${SGLANG_V100_PYTHON}"
elif [[ -x "${HOME}/work/sglang-v100-venv/bin/python" ]]; then
  PYTHON="${HOME}/work/sglang-v100-venv/bin/python"
else
  echo "dsv41_vision: set SGLANG_V100_PYTHON" >&2
  exit 1
fi
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "${ROOT}/scripts/testbench/dsv41_vision.py" "$@"
