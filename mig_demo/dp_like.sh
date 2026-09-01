#!/usr/bin/env bash
# DP-like case: two INDEPENDENT TP=1 sglang servers sharing one H200.
# (With MIG these would be memory-isolated instances; here mem-fraction caps
# emulate the isolation.) No collectives involved. Benches both concurrently.
set -uo pipefail

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE=/cluster-storage/models/hub
LOG_DIR=/root/mig-demo

pkill -f sglang.launch_server 2>/dev/null
sleep 3

for item in "30000:${MEM_A:-0.42}" "30001:${MEM_B:-0.84}"; do
    p=${item%%:*}; frac=${item##*:}
    nohup python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
        --host 127.0.0.1 --port $p --tp 1 --mem-fraction-static $frac \
        > "$LOG_DIR/dp_$p.log" 2>&1 &
done

for p in 30000 30001; do
    ok=0
    for i in $(seq 1 60); do
        curl -sf http://127.0.0.1:$p/health > /dev/null 2>&1 && { ok=1; break; }
        sleep 5
    done
    [ "$ok" = 1 ] || { echo "server on $p failed"; tail -5 "$LOG_DIR/dp_$p.log"; exit 1; }
done
echo "both servers up; benching concurrently"
grep -h "max_total_num_tokens" "$LOG_DIR/dp_30000.log" "$LOG_DIR/dp_30001.log"

python /root/mig-demo/verify_and_bench.py --port 30000 --label dp-instance-A --n 32 --max-new-tokens 128 --concurrency 16 > /tmp/benchA.json 2>&1 &
BA=$!
python /root/mig-demo/verify_and_bench.py --port 30001 --label dp-instance-B --n 32 --max-new-tokens 128 --concurrency 16 > /tmp/benchB.json 2>&1 &
BB=$!
wait $BA $BB
cat /tmp/benchA.json /tmp/benchB.json

pkill -f sglang.launch_server 2>/dev/null
sleep 2
