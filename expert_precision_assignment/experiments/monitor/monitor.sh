#!/usr/bin/env bash
# One-shot sweep progress snapshot.
# Usage:
#   bash monitor.sh                         # sharegpt (default)
#   DATASET=gsm8k bash monitor.sh           # different dataset
#   watch -n 30 bash monitor.sh             # live monitor (refresh 30s)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-sharegpt}"
R="$SCRIPT_DIR/results/$DATASET"

total=66
done=$(ls "$R"/*.jsonl 2>/dev/null | wc -l)
echo "════════════════════════════════════════════════════════════"
printf "  Sweep progress: %d / %d jsonl complete (%.0f%%)\n" \
    "$done" "$total" "$(awk "BEGIN{print 100*$done/$total}")"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

echo "  Per-mc completion:"
for mc in 8 16 32 64 128 256; do
    c=$(ls "$R"/mc${mc}_*.jsonl 2>/dev/null | wc -l)
    printf "    mc%-3d : %2d / 11\n" "$mc" "$c"
done

echo ""
echo "  Per-GPU status:"
for g in 4 5 6 7; do
    w="$R/gpu${g}_worker.log"
    [ -f "$w" ] || { echo "    gpu$g : <no log>"; continue; }
    ok=$(grep -c "server ready" "$w" 2>/dev/null)
    fail=$(grep -c "server died" "$w" 2>/dev/null)
    last=$(tail -1 "$w" 2>/dev/null | sed 's/.*\(\[gpu[0-9]\+.*\)/\1/' | cut -c1-80)
    printf "    gpu%d : %2d ok, %2d fail | %s\n" "$g" "$ok" "$fail" "$last"
done

echo ""
echo "  Bench in flight:"
for f in "$R"/*_bench.log; do
    [ -f "$f" ] || continue
    base=$(basename "$f" _bench.log)
    jsonl="$R/${base}_n1024.jsonl"
    if [ ! -f "$jsonl" ]; then
        last=$(tail -1 "$f" 2>/dev/null | cut -c1-80)
        printf "    %-22s %s\n" "$base" "$last"
    fi
done

echo ""
echo "  GPU util:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "    gpu%s : %s, util=%s\n", $1, $2, $3}'
