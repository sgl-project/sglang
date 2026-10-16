#!/usr/bin/env bash
# Invoked inside guarded_node.py by reproduce.py prepare.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
exec 9> "$WORK_DIR/.prepare.lock"
flock -n 9 || die "another preparation is using $WORK_DIR"
bash "$HERE/setup_server.sh"
bash "$HERE/setup_client.sh"
bash "$HERE/prepare_dataset.sh"
if [ ! -e "$WORK_DIR/plot-env" ]; then
    "$CLIENT_BASE_PYTHON" -m venv "$WORK_DIR/plot-env"
    "$WORK_DIR/plot-env/bin/python" -m pip install --disable-pip-version-check 'matplotlib==3.10.8'
fi
"$WORK_DIR/plot-env/bin/python" -c \
  'import importlib.metadata as m; assert m.version("matplotlib") == "3.10.8"; import matplotlib.pyplot'
