#!/usr/bin/env bash
set -euo pipefail

# TP production shortcut for the HY4 + FlexKV CP/DP-capable launcher.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PARALLEL_MODE=tp
exec "${SCRIPT_DIR}/launch_hy4_mi355x_cptp_flexkv.sh" "$@"
