#!/usr/bin/env bash
# Runs INSIDE the ROCm SGLang container on every allocated node.
#
# Role is decided by NODE_RANK (= SLURM_PROCID), injected by pd_disagg.slurm:
#   rank < PREFILL_NODES            -> prefill server
#   rank >= PREFILL_NODES           -> decode server
#   rank 0 (first prefill node)     -> also runs the router + benchmark client
#
# Cross-node coordination is done with a marker file on the shared /results
# mount: workers idle until rank 0 finishes the benchmark and touches it.

set -uo pipefail

NODE_RANK="${NODE_RANK:?}"
NNODES="${NNODES:?}"
PREFILL_NODES="${PREFILL_NODES:?}"
RESULTS_DIR="${RESULTS_DIR:-/results}"
LOGS_DIR="${LOGS_DIR:-/logs}"
HOST="$(hostname)"
MARKER="$RESULTS_DIR/.bench_complete"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Resolve per-node IPs (rank order) → prefill/decode endpoints.
IFS=',' read -r -a IPS <<< "$IPADDRS"
PREFILL_IP="${IPS[0]}"
DECODE_IP="${IPS[$PREFILL_NODES]:-${IPS[0]}}"

log() { echo "[server_pd][rank $NODE_RANK][$HOST] $*"; }

wait_for_health() {
    local url="$1" name="$2" timeout="${3:-1800}" start
    start=$(date +%s)
    log "waiting for $name at $url (timeout ${timeout}s)"
    until curl -sf "$url" >/dev/null 2>&1; do
        if (( $(date +%s) - start >= timeout )); then
            log "ERROR: timed out waiting for $name ($url)"
            return 1
        fi
        sleep 5
    done
    log "$name is ready"
}

# ---------------------------------------------------------------------------
# Start the role-appropriate server (background).
# ---------------------------------------------------------------------------
if (( NODE_RANK < PREFILL_NODES )); then
    ROLE=prefill
    log "starting PREFILL server (tp=$PREFILL_TP) on port $PREFILL_SERVER_PORT"
    # shellcheck disable=SC2086
    python3 -m sglang.launch_server \
        --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
        --host 0.0.0.0 --port "$PREFILL_SERVER_PORT" --tp "$PREFILL_TP" \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port "$PREFILL_BOOTSTRAP_PORT" \
        $COMMON_FLAGS $PREFILL_FLAGS \
        > >(tee "$LOGS_DIR/prefill_${HOST}.log") 2>&1 &
    SERVER_PID=$!
    SELF_HEALTH="http://localhost:${PREFILL_SERVER_PORT}/health"
else
    ROLE=decode
    log "starting DECODE server (tp=$DECODE_TP) on port $DECODE_SERVER_PORT"
    # shellcheck disable=SC2086
    python3 -m sglang.launch_server \
        --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
        --host 0.0.0.0 --port "$DECODE_SERVER_PORT" --tp "$DECODE_TP" \
        --disaggregation-mode decode \
        --disaggregation-bootstrap-port "$DECODE_BOOTSTRAP_PORT" \
        $COMMON_FLAGS $DECODE_FLAGS \
        > >(tee "$LOGS_DIR/decode_${HOST}.log") 2>&1 &
    SERVER_PID=$!
    SELF_HEALTH="http://localhost:${DECODE_SERVER_PORT}/health"
fi

wait_for_health "$SELF_HEALTH" "$ROLE server" 2400 || { kill $SERVER_PID 2>/dev/null; exit 1; }

# ---------------------------------------------------------------------------
# Non-master ranks: idle until rank 0 signals the benchmark is done.
# ---------------------------------------------------------------------------
if (( NODE_RANK != 0 )); then
    log "$ROLE server up; waiting for benchmark completion marker"
    while [ ! -f "$MARKER" ]; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log "ERROR: $ROLE server died before benchmark completed"
            exit 1
        fi
        sleep 10
    done
    log "marker seen; shutting down $ROLE server"
    kill $SERVER_PID 2>/dev/null || true
    exit 0
fi

# ---------------------------------------------------------------------------
# Rank 0: wait for the decode server (on another node), start the router, run
# the benchmark sweep, write one normalized result JSON per concurrency.
# ---------------------------------------------------------------------------
wait_for_health "http://${DECODE_IP}:${DECODE_SERVER_PORT}/health" "decode server (remote)" 2400 \
    || { kill $SERVER_PID 2>/dev/null; exit 1; }

log "starting router on port $ROUTER_PORT (prefill=$PREFILL_IP decode=$DECODE_IP)"
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:${PREFILL_SERVER_PORT}" "$PREFILL_BOOTSTRAP_PORT" \
    --decode "http://${DECODE_IP}:${DECODE_SERVER_PORT}" \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    > >(tee "$LOGS_DIR/router_${HOST}.log") 2>&1 &
ROUTER_PID=$!

wait_for_health "http://localhost:${ROUTER_PORT}/health" "router" 600 \
    || { kill $SERVER_PID $ROUTER_PID 2>/dev/null; exit 1; }

BENCH_RC=0
for CONC in $CONCURRENCIES; do
    NP=$(( CONC * 5 ))
    (( NP < 50 )) && NP=50
    log "benchmark: concurrency=$CONC num_prompts=$NP isl=$ISL osl=$OSL"
    RAW="/tmp/bench_${CONC}.jsonl"
    rm -f "$RAW"
    if ! python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "$ROUTER_PORT" \
        --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name random \
        --random-input-len "$ISL" --random-output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --max-concurrency "$CONC" --num-prompts "$NP" --request-rate inf \
        --output-file "$RAW" 2>&1 | tee "$LOGS_DIR/bench_conc${CONC}_${HOST}.log"; then
        log "ERROR: bench_serving failed at concurrency $CONC"
        BENCH_RC=1
        continue
    fi
    # Normalize the last result line into the schema process_result.py expects.
    MODEL="$MODEL" CONC="$CONC" python3 - "$RAW" "$RESULTS_DIR/results_concurrency_${CONC}.json" <<'PY'
import json, os, sys
raw_path, out_path = sys.argv[1], sys.argv[2]
with open(raw_path) as f:
    lines = [l for l in f if l.strip()]
d = json.loads(lines[-1])
tot = d.get("total_token_throughput")
if tot is None:
    tot = float(d.get("input_throughput", 0)) + float(d.get("output_throughput", 0))
out = {
    "model_id": os.environ["MODEL"],
    "max_concurrency": int(d.get("max_concurrency", os.environ["CONC"])),
    "total_token_throughput": float(tot),
    "output_throughput": float(d["output_throughput"]),
    "median_ttft_ms": float(d["median_ttft_ms"]),
    "median_tpot_ms": float(d["median_tpot_ms"]),
    "median_e2el_ms": float(d["median_e2el_ms"]),
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {out_path}: {out}")
PY
done

log "benchmark sweep complete (rc=$BENCH_RC); signaling workers"
touch "$MARKER"
kill $SERVER_PID $ROUTER_PID 2>/dev/null || true
exit $BENCH_RC
