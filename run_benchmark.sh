#!/bin/bash
# LPLB vs Dynamic benchmark orchestrator
# Usage: run inside node 0 container (10.6.131.11)
set -euo pipefail

RESULTS_DIR="/raid/fei/lplb/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
BENCH_SCRIPT="/raid/fei/lplb/sglang/benchmark_lplb.py"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$RESULTS_DIR/benchmark.log"; }

log "============================================"
log "LPLB vs Dynamic Benchmark"
log "Results: $RESULTS_DIR"
log "============================================"

# Run order: dynamic first, then lp (Run A)
for DISPATCH in dynamic lp; do
    log ""
    log "======== DISPATCHER: $DISPATCH ========"

    # Wait for server to be ready (launched externally)
    log "Waiting for server..."
    for i in $(seq 1 60); do
        if curl -s -m 5 http://localhost:30000/health >/dev/null 2>&1; then
            log "Server ready"
            break
        fi
        if [ "$i" -eq 60 ]; then
            log "ERROR: Server timeout"
            exit 1
        fi
        sleep 5
    done

    # JIT warmup
    log "JIT warmup (1 request)..."
    curl -s -m 300 http://localhost:30000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"default","prompt":"JIT warmup","max_tokens":1}' > /dev/null 2>&1
    log "JIT done"

    # Benchmark MMLU
    log "--- Benchmark: $DISPATCH / mmlu ---"
    python3 "$BENCH_SCRIPT" "$DISPATCH" mmlu 1000 "$RESULTS_DIR" 2>&1 | tee -a "$RESULTS_DIR/benchmark.log"

    log "--- $DISPATCH done. Signal to stop server. ---"
    # Marker file for external orchestrator to know this dispatcher is done
    touch "$RESULTS_DIR/${DISPATCH}_done"

    # Wait for next server (if not last)
    if [ "$DISPATCH" = "dynamic" ]; then
        log "Waiting for LP server to start..."
        while [ ! -f "$RESULTS_DIR/lp_server_ready" ]; do sleep 2; done
        rm -f "$RESULTS_DIR/lp_server_ready"
    fi
done

# Summary
log ""
log "============================================"
log "SUMMARY"
log "============================================"
python3 -c "
import json, glob
results = {}
for f in sorted(glob.glob('$RESULTS_DIR/*_result.json')):
    with open(f) as fh:
        r = json.load(fh)
    results[(r['dispatch'], r['dataset'])] = r
print(f\"{'Dataset':<12} {'Dispatch':<10} {'p50':>8} {'p90':>8} {'tput':>10} {'err':>5}\")
print('-' * 55)
for ds in ['mmlu']:
    for disp in ['dynamic', 'lp']:
        r = results.get((disp, ds))
        if r:
            print(f\"{ds:<12} {disp:<10} {r.get('lat_p50','?'):>8} {r.get('lat_p90','?'):>8} {r.get('tput_input_tok_s','?'):>10} {r.get('n_errors',0):>5}\")
        else:
            print(f\"{ds:<12} {disp:<10} {'MISSING':>8}\")
" 2>&1 | tee -a "$RESULTS_DIR/benchmark.log"

log ""
log "Results saved: $RESULTS_DIR"
log "Done!"
