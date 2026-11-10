#!/usr/bin/env bash
# Strict PPU nightly execution with persisted JUnit, summary, and logs.
set -euo pipefail

suite="${1:?suite required}"
partition_id="${2-partition id or empty for full suite}"
evidence_dir="${3:?evidence directory required}"
suite_timeout="${4:?suite timeout in seconds required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "$evidence_dir"
python3 scripts/ci/ppu/nightly_skip_policy.py --json > "$evidence_dir/allowed_skips.json"
mapfile -t skip_args < <(python3 scripts/ci/ppu/nightly_skip_policy.py)
mapfile -t deselect_args < <(python3 scripts/ci/ppu/nightly_skip_policy.py --deselect)
partition_args=()
if [[ -n "$partition_id" ]]; then
    partition_args=(--auto-partition-id "$partition_id" --auto-partition-size 2)
fi

cd test
python3 run_suite.py \
    --hw ppu \
    --suite "$suite" \
    --nightly \
    --require-nonempty \
    --continue-on-error \
    "${partition_args[@]}" \
    --pytest \
    --junit-dir "$evidence_dir" \
    --timeout-per-file 1200 \
    --suite-timeout "$suite_timeout" \
    "${skip_args[@]}" \
    "${deselect_args[@]}" 2>&1 | tee "$evidence_dir/suite.log"
