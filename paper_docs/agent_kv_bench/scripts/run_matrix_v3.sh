#!/bin/bash
B=/data/xbw/conda_envs/sglang/bin/python
H=/home/xubowen/mxfp4/sglang/paper_docs/agent_kv_bench/harness.py
restart() {
  docker exec sglang-qwen3 bash -c "pkill -9 -f 'sglang' || true"
  sleep 6
  docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && nohup python3 -m sglang.launch_server --model-path /data/models/Qwen3-4B-Instruct-2507 --port 30000 --host 0.0.0.0 --tool-call-parser qwen25 --context-length 32768 $1 > $2 2>&1 & sleep 1; echo started"
  for i in $(seq 1 36); do
    sleep 5
    docker exec sglang-qwen3 grep -q "fired up" $2 2>/dev/null && return 0
    docker exec sglang-qwen3 grep -qE "Traceback" $2 2>/dev/null && { echo "ERR starting $2"; exit 1; }
  done
  echo "TIMEOUT $2"; exit 1
}
cd /data/xbw/kvbench
echo "=== V3 LRU ==="
restart "" /tmp/server_kv_bench_v3lru.log
for n in 16 32; do echo "--- v3_lru_n$n"; $B $H --concurrency $n --rounds 6 --seed 7 --out v3_lru_n$n.jsonl 2>&1 | tail -11; done
echo "=== V3 SCORE ==="
restart "--enable-tool-retention" /tmp/server_kv_bench_v3score.log
for n in 8 16 32; do echo "--- v3_score_n$n"; $B $H --concurrency $n --rounds 6 --seed 7 --out v3_score_n$n.jsonl 2>&1 | tail -11; done
echo "=== V3 LRU n8 (backfill) ==="
restart "" /tmp/server_kv_bench_v3lru.log
echo "--- v3_lru_n8"; $B $H --concurrency 8 --rounds 6 --seed 7 --out v3_lru_n8.jsonl 2>&1 | tail -11
echo "=== V3 DONE ==="
