#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# run_controller.sh runs pd_flip_controller.py inside Docker by default and
# supplies monitor options such as --ttft-slo / --tpot-slo /
# --commit-threshold from env.local.
exec "${SCRIPT_DIR}/run_controller.sh" monitor
