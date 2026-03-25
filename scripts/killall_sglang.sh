#!/bin/bash
# DEPRECATED: Use 'python3 python/sglang/cli/killall.py' or 'killall_sglang' instead.
echo "WARNING: killall_sglang.sh is deprecated. Use 'python3 python/sglang/cli/killall.py' or 'killall_sglang' instead." >&2
exec python3 "$(dirname "$0")/../python/sglang/cli/killall.py" "$@"
