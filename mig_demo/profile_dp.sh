#!/usr/bin/env bash
# Profile the DP-like co-located servers under concurrent load.
# Traces land in /root/mig-demo/traces/{a,b}/ as Chrome trace .json.gz files.
set -uo pipefail

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE=/cluster-storage/models/hub
LOG_DIR=/root/mig-demo
TRACE_DIR=/root/mig-demo/traces

pkill -f sglang.launch_server 2>/dev/null
sleep 3
rm -rf "$TRACE_DIR"
mkdir -p "$TRACE_DIR/a" "$TRACE_DIR/b"

# Sequential startup: the mem fractions assume A allocates its pool before B
# measures free memory (pool = fraction x total - already_used).
for item in "30000:0.42:a" "30001:0.84:b"; do
    p=${item%%:*}; rest=${item#*:}; frac=${rest%%:*}; tag=${rest##*:}
    SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR/$tag" \
    nohup python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
        --host 127.0.0.1 --port $p --tp 1 --mem-fraction-static $frac \
        > "$LOG_DIR/prof_$p.log" 2>&1 &
    ok=0
    for i in $(seq 1 60); do
        curl -sf http://127.0.0.1:$p/health > /dev/null 2>&1 && { ok=1; break; }
        sleep 5
    done
    [ "$ok" = 1 ] || { echo "server on $p failed"; tail -5 "$LOG_DIR/prof_$p.log"; exit 1; }
    echo "server on $p up"
done
echo "both servers up"

# sanity generation on each
for p in 30000 30001; do
    curl -sf http://127.0.0.1:$p/generate -H 'Content-Type: application/json' \
        -d '{"text":"Hi","sampling_params":{"temperature":0,"max_new_tokens":8}}' \
        | head -c 120; echo
done

# start profiler on both (auto-stops after num_steps forward steps)
for p in 30000 30001; do
    curl -sf http://127.0.0.1:$p/start_profile -H 'Content-Type: application/json' \
        -d '{"num_steps": 60, "activities": ["CPU", "GPU"]}' && echo " <- profiling :$p"
done

# drive load on both concurrently
python /root/mig-demo/verify_and_bench.py --port 30000 --label prof-A --n 16 --max-new-tokens 128 --concurrency 8 &
BA=$!
python /root/mig-demo/verify_and_bench.py --port 30001 --label prof-B --n 16 --max-new-tokens 128 --concurrency 8 &
BB=$!
wait $BA $BB

sleep 5
for p in 30000 30001; do
    curl -sf -X POST http://127.0.0.1:$p/stop_profile && echo " <- stopped :$p"
done
sleep 10

echo "=== traces ==="
find "$TRACE_DIR" -name "*.trace.json.gz" -exec ls -la {} \;

pkill -f sglang.launch_server 2>/dev/null
