#!/usr/bin/env bash
# Verify an existing Beam CUDA 13 runtime. Never rewrite its dependencies.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
REPO="$(cd "$HERE/../.." && pwd)"
PYTHONPATH="${BEAM_SERVER_PYTHONPATH:-$REPO/python}" \
  "${SERVER_PYTHON:?set SERVER_PYTHON in site.env}" -B "$HERE/check_runtime.py" \
  --out "$WORK_DIR/server-runtime.json"
