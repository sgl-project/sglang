#!/usr/bin/env bash
# Final-code e2e: two TP4 halves of the node each take the next job from the queue
# as they free up (half A only after the final test gate exists).
#   bash tools/e2e_queue.sh
set -uo pipefail

ROOT=/hai-workspace/dsv41-fi-topk
cd "$ROOT"
QUEUE=$ROOT/e2e/final-queue.txt
LOCK=$ROOT/e2e/final-queue.lock
GATE=$ROOT/logs/tests-final.done

# name|env assignments|server args
cat > "$QUEUE" <<'EOF'
r5-flashinfer-unfused|SGLANG_DSA_FUSE_TOPK=0|--dsa-topk-backend flashinfer
r5-sgl-kernel||
r5-flashinfer||--dsa-topk-backend flashinfer
r5-flashinfer-small-det|SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1|--dsa-topk-backend flashinfer
EOF

log() { echo "$(date -u +%FT%TZ) $*"; }

gpus_idle() {
  local used
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v want="$1" 'BEGIN { split(want, g, ",") } { for (i in g) if ($1 == g[i] && $2 > 1024) print $1 }')
  [[ -z $used ]]
}

next_job() {
  flock "$LOCK" bash -c 'head -1 "$0"; sed -i 1d "$0"' "$QUEUE"
}

worker() {
  local gpus=$1 port=$2 gate=${3:-} job name envs args
  if [[ -n $gate ]]; then
    until [[ -f $gate ]]; do sleep 30; done
  fi
  while job=$(next_job) && [[ -n $job ]]; do
    IFS='|' read -r name envs args <<< "$job"
    until gpus_idle "$gpus"; do sleep 10; done
    log "$name start gpus=$gpus"
    # shellcheck disable=SC2086
    bash tools/run_e2e.sh "$name" "$gpus" "$port" $envs -- $args > "e2e/$name.out" 2>&1
    log "$name finished: $(tail -2 "e2e/$name/run.txt" | tr '\n' ' ')"
  done
}

log "queue start"
worker 4,5,6,7 30001 &
worker 0,1,2,3 30000 "$GATE" &
wait
log "e2e finished"
