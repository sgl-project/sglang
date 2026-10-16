#!/usr/bin/env bash
# One foreground node rank. Use guarded_node.py for unattended execution.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/launch_server.py" "$@"
