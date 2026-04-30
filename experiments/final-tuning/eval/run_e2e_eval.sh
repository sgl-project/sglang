#!/bin/bash
# E2E ShareGPT bench_serving comparison: baseline (pure INT4) vs treatment
# (heter-MoE with autotuned tiles + EfficiencyPromotionPolicy).
#
# Two servers in parallel:
#   baseline  on GPUs 0-3 (TP=4) port 31000
#   treatment on GPUs 4-7 (TP=4) port 32000
# bench_serving sequential per mc against each server.
set -o pipefail
# Note: set -u causes conda activate to fail on unset PS1; intentionally omitted

cd "$(dirname "$0")/../../.."

EVAL_DIR=experiments/final-tuning/eval
LOG_DIR=$EVAL_DIR/logs
RESULT_DIR=$EVAL_DIR/results
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Models + dataset
INT4_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"
BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
SHAREGPT="/data/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"

BASELINE_CFG=$EVAL_DIR/configs/baseline_int4_only.json
TREATMENT_CFG=$EVAL_DIR/configs/treatment_mc64.json

BASELINE_PORT=31000
TREATMENT_PORT=32000
# GPUs 0-3 reserved for another user; we run on 4 and 5.
# TP=1 each — TP=4 INT4-sharding bug (tensor 192 vs 768).
BASELINE_GPUS="2"
TREATMENT_GPUS="3"
TP_SIZE=1

NUM_PROMPTS=${NUM_PROMPTS:-300}
MC_LIST=(32 64 128)

eval "$(conda shell.bash hook)"
conda activate sglang

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/driver.log" >&2; }

# ---------------- launch + health check ----------------
launch_server() {
    local label=$1 cfg=$2 port=$3 gpus=$4 logfile=$5
    local extra=""
    # Treatment uses EfficiencyPromotionPolicy which has Python-side
    # cache mutation in dispatch — incompatible with piecewise CUDA
    # graph's runtime recompile path. Disable piecewise capture for
    # treatment; baseline keeps it on (no policy state mutation).
    if [[ "$label" == "treatment" ]]; then
        extra="--disable-piecewise-cuda-graph"
    fi
    log "launching $label on GPUs $gpus port $port $extra"
    CUDA_VISIBLE_DEVICES=$gpus python -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --tp "$TP_SIZE" \
        --host 127.0.0.1 --port "$port" \
        --trust-remote-code \
        --heter-precision-config "$cfg" \
        --mem-fraction-static 0.83 \
        $extra \
        > "$logfile" 2>&1 &
    echo $!
}

wait_ready() {
    # Treatment server (efficiency_promotion policy) takes ~18 min to
    # compile because dynamo retraces the dispatch path per dynamic shape.
    local port=$1 pid=$2 max_wait=1800 elapsed=0
    while ! curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "  ERROR: server pid $pid died (port $port)"
            return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            log "  ERROR: server pid $pid did not become ready in ${max_wait}s"
            return 1
        fi
    done
    log "  server ready (port $port) after ${elapsed}s"
}

bench_one() {
    local label=$1 port=$2 mc=$3
    local out=$RESULT_DIR/${label}_mc${mc}.jsonl
    local benchlog=$LOG_DIR/${label}_mc${mc}.log
    log "bench_serving $label mc=$mc → $out"
    curl -s -X POST "http://127.0.0.1:${port}/flush_cache" > /dev/null 2>&1 || true
    sleep 1
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${port}" \
        --dataset-name sharegpt \
        --dataset-path "$SHAREGPT" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$mc" \
        --output-file "$out" \
        > "$benchlog" 2>&1 || log "  bench failed (see $benchlog)"
}

# ---------------- main ----------------
START=$(date +%s)

BASELINE_PID=$(launch_server "baseline" "$BASELINE_CFG" \
    "$BASELINE_PORT" "$BASELINE_GPUS" "$LOG_DIR/server_baseline.log")
TREATMENT_PID=$(launch_server "treatment" "$TREATMENT_CFG" \
    "$TREATMENT_PORT" "$TREATMENT_GPUS" "$LOG_DIR/server_treatment.log")
log "baseline pid=$BASELINE_PID  treatment pid=$TREATMENT_PID"

trap 'kill -9 $BASELINE_PID $TREATMENT_PID 2>/dev/null; exit' INT TERM

wait_ready "$BASELINE_PORT" "$BASELINE_PID" || { log "baseline failed"; kill -9 $BASELINE_PID $TREATMENT_PID 2>/dev/null; exit 1; }
wait_ready "$TREATMENT_PORT" "$TREATMENT_PID" || { log "treatment failed"; kill -9 $BASELINE_PID $TREATMENT_PID 2>/dev/null; exit 1; }

for mc in "${MC_LIST[@]}"; do
    bench_one "baseline" "$BASELINE_PORT" "$mc"
    bench_one "treatment" "$TREATMENT_PORT" "$mc"
done

log "tearing down servers"
kill -INT "$BASELINE_PID" "$TREATMENT_PID" 2>/dev/null
sleep 5
kill -9 "$BASELINE_PID" "$TREATMENT_PID" 2>/dev/null

END=$(date +%s)
log "DONE in $((END - START))s"
log "  results: $RESULT_DIR/"

# Aggregate to side-by-side CSV
python - << PYEOF 2>&1 | tee -a "$LOG_DIR/driver.log"
import csv, glob, json, os
out_path = "$RESULT_DIR/sharegpt_e2e.csv"
rows = []
for f in sorted(glob.glob("$RESULT_DIR/*.jsonl")):
    name = os.path.basename(f).replace(".jsonl", "")
    label, mc = name.rsplit("_mc", 1)
    with open(f) as fp:
        for line in fp:
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows.append({
                "label": label, "mc": int(mc),
                "completed": d.get("completed"),
                "request_throughput": d.get("request_throughput"),
                "total_input_tokens": d.get("total_input_tokens"),
                "total_output_tokens": d.get("total_output_tokens"),
                "input_throughput": d.get("input_throughput"),
                "output_throughput": d.get("output_throughput"),
                "mean_ttft_ms": d.get("mean_ttft_ms"),
                "median_ttft_ms": d.get("median_ttft_ms"),
                "p99_ttft_ms": d.get("p99_ttft_ms"),
                "mean_itl_ms": d.get("mean_itl_ms"),
                "median_itl_ms": d.get("median_itl_ms"),
                "p99_itl_ms": d.get("p99_itl_ms"),
                "mean_e2el_ms": d.get("mean_e2el_ms"),
                "median_e2el_ms": d.get("median_e2el_ms"),
                "p99_e2el_ms": d.get("p99_e2el_ms"),
            })
if rows:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")
else:
    print("no results to aggregate")
PYEOF
