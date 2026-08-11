#!/bin/bash
# Collect a flamegraph of the rust TM threads on a live SGLANG_RUST_SERVER=1
# server: perf-attach to the named rust threads while driver.py generates load,
# then render SVG via the FlameGraph scripts (cloned to /tmp on first use).
#
# Usage: collect_flamegraph.sh [port] [model] [input_len] [num_requests] [out.svg]
set -euo pipefail
PORT=${1:-30800}
MODEL=${2:-Qwen/Qwen3.5-35B-A3B-FP8}
LEN=${3:-16384}
NREQ=${4:-64}
OUT=${5:-rust_tm_in${LEN}.svg}
HERE=$(dirname "$(readlink -f "$0")")
WORK=$(mktemp -d /tmp/rust_tm_fg.XXXX)   # perf -o MUST be local disk: a network
                                         # mount fails with EFAULT "Bad address"

# Resolve the scheduler owned by THIS port's server (several servers can
# coexist; shell wrappers embed the launch string, so match python processes
# only). Under TP>1 the rust TM lives in the TP0 scheduler.
SERVER=$(ps -eo pid,cmd | awk -v port="--port $PORT" \
  '$2 ~ /python/ && index($0, "sglang.launch_server") && index($0, port) {print $1; exit}')
[ -n "$SERVER" ] || { echo "no sglang.launch_server with --port $PORT" >&2; exit 1; }
SCHED=$(ps -eo pid,ppid,cmd | awk -v p="$SERVER" \
  '$2==p && ($3=="sglang::scheduler" || $3=="sglang::scheduler_TP0") {print $1; exit}')
[ -n "$SCHED" ] || { echo "no scheduler child of server pid $SERVER" >&2; exit 1; }
TIDS=$(ps -T -p "$SCHED" -o tid,comm | awk \
  '$2 ~ /tokenizer-|tm-ingress|tm-egress|detokenizer-|api-runtime|tokio-rt/ {print $1}' | paste -sd,)
echo "scheduler=$SCHED rust tids=$TIDS"

FG=/tmp/FlameGraph
[ -d "$FG" ] || git clone -q --depth 1 https://github.com/brendangregg/FlameGraph "$FG"

# fp unwinding requires the CARGO_PROFILE_RELEASE_FORCE_FRAME_POINTERS=true build.
perf record -F 999 -g --call-graph fp --no-bpf-event -t "$TIDS" -o "$WORK/perf.data" &
PERF=$!
sleep 1
python3 "$HERE/driver.py" --port "$PORT" --model "$MODEL" --input-len "$LEN" \
  --num-requests "$NREQ" --warmup 2 --out /dev/null | tail -1
kill -INT "$PERF"; wait "$PERF" 2>/dev/null || true

perf script -i "$WORK/perf.data" --no-inline > "$WORK/perf.script"
"$FG/stackcollapse-perf.pl" "$WORK/perf.script" > "$WORK/perf.folded"
"$FG/flamegraph.pl" --width 1400 \
  --title "rust TM threads, $MODEL, in=$LEN x$NREQ" "$WORK/perf.folded" > "$OUT"
echo "flamegraph: $OUT  (raw: $WORK/perf.data, folded: $WORK/perf.folded)"
