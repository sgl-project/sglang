#!/usr/bin/env bash
# Run e2e cases back to back on one four-GPU slot of the devbox.
#   bash queue.sh <gpus> <case> [<case> ...]
# Each case is "run|file|class|tree|cache_name" (cache_name may be empty).
# A case waits for the slot's previous case; WAIT_FOR names a run whose
# test.log must show exit= before the first case starts.
set -uo pipefail
ROOT=${ROOT:-/hai-workspace/moe-prefill-cg}
GPUS=${1:?gpus}
shift
if [[ -n "${WAIT_FOR:-}" ]]; then
  until grep -q '^exit=' "$ROOT/logs/$WAIT_FOR/test.log" 2>/dev/null; do sleep 20; done
  # Let the previous server release its GPUs.
  sleep 30
fi
for case in "$@"; do
  IFS='|' read -r run file class tree cache <<< "$case"
  echo "$(date -u +%FT%TZ) start $run"
  TREE=$tree CACHE_NAME=$cache bash "$ROOT/tools/run_case.sh" "$GPUS" "$run" "$file" "$class"
  echo "$(date -u +%FT%TZ) done $run"
  sleep 30
done
