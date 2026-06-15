#!/usr/bin/env bash
# POC: card-level LPT sharding for XPU CI.
#
# Detects how many XPU cards the current runner exposes, then launches one
# subprocess per card. Each subprocess is pinned to a single card via
# ZE_AFFINITY_MASK and runs its own LPT slice of the suite (auto-partition-id
# 0..N-1, auto-partition-size N). LPT inside run_suite.py balances test files
# across the N buckets by est_time.
#
# Usage:
#   bash scripts/ci/xpu/run_card_sharded.sh <suite> [timeout_per_file_seconds]
#
# Knobs (env):
#   SGLANG_XPU_MAX_CONCURRENCY  Cap shard count below physical card count.
#                                Useful when host RAM/CPU can't sustain N-way
#                                concurrency. Default: card count.

set -uo pipefail

SUITE="${1:?suite name required as first arg}"
TIMEOUT_PER_FILE="${2:-1800}"

N=$(python3 -c "import torch
ok = hasattr(torch, 'xpu') and torch.xpu.is_available()
print(torch.xpu.device_count() if ok else 0)")

if [ "$N" -eq 0 ]; then
  echo "::error::No XPU devices available on $(hostname)"
  exit 1
fi

MAX="${SGLANG_XPU_MAX_CONCURRENCY:-$N}"
if [ "$MAX" -lt "$N" ]; then
  echo "::warning::Capping shard count at $MAX (box has $N cards)"
  N="$MAX"
fi

echo "Launching $N parallel shards on $(hostname) for suite=$SUITE"

LOG_DIR="${RUNNER_TEMP:-/tmp}/shard_logs"
mkdir -p "$LOG_DIR"

pids=()
for i in $(seq 0 $((N-1))); do
  (
    export ZE_AFFINITY_MASK="$i"
    export HF_HOME="${RUNNER_TEMP:-/tmp}/hf_cache_$i"
    export TRITON_CACHE_DIR="${RUNNER_TEMP:-/tmp}/triton_cache_$i"
    export TORCHINDUCTOR_CACHE_DIR="${RUNNER_TEMP:-/tmp}/inductor_cache_$i"
    export MASTER_PORT=$((29500 + i))
    cd test
    python3 run_suite.py --hw xpu --suite "$SUITE" \
      --auto-partition-id "$i" --auto-partition-size "$N" \
      --timeout-per-file "$TIMEOUT_PER_FILE"
  ) > "$LOG_DIR/shard_$i.log" 2>&1 &
  pids+=($!)
done

fail=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "::error::Shard $idx FAILED"
    fail=1
  else
    echo "Shard $idx ok"
  fi
done

echo "===== Per-shard logs ====="
for i in $(seq 0 $((N-1))); do
  echo "::group::shard $i"
  cat "$LOG_DIR/shard_$i.log" || true
  echo "::endgroup::"
done

exit "$fail"
