#!/bin/bash
# Compact one-line status of the th3 matrices plus hang indicators.

ROOT=${ROOT:-/scratch/sumpad_th3}
REPEAT_ROOT=${REPEAT_ROOT:-/scratch/sumpad_th3_rep}

NOW=$(date '+%H:%M:%S')
BENCH=$(ls "$ROOT"/bench/*/DONE 2>/dev/null | wc -l | tr -d ' ')
EV=$(ls "$ROOT"/evidence/*/DONE 2>/dev/null | wc -l | tr -d ' ')
PROF=$(ls "$ROOT"/profile/*/DONE 2>/dev/null | wc -l | tr -d ' ')
REP=$(ls "$REPEAT_ROOT"/bench/*/DONE 2>/dev/null | wc -l | tr -d ' ')

LATEST=$(ls -dt "$ROOT"/*/*/ "$REPEAT_ROOT"/*/*/ 2>/dev/null | head -1)
LOG="$LATEST/server.log"
AGE="na"
if [ -f "$LOG" ]; then
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
fi

UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd, -)
BADNESS=$(grep -lE "watchdog timeout|SIGQUIT|CUDA error|Traceback" "$ROOT"/*/*/server.log "$REPEAT_ROOT"/*/*/server.log 2>/dev/null | tr '\n' ' ')
BENCHPROC=$(pgrep -fc "bench_serving|one_batch_server" 2>/dev/null)

echo "[$NOW] bench=$BENCH/8 evidence=$EV/8 profile=$PROF/8 repeat=$REP/8 cur=$(basename "$LATEST") log_age=${AGE}s gpu_util=$UTIL bench_procs=$BENCHPROC bad_logs=[$BADNESS]"
