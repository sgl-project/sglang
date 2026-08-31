#!/bin/bash
B=/data/xbw/conda_envs/sglang/bin/python
H=/home/xubowen/mxfp4/sglang/paper_docs/agent_kv_bench/harness.py
restart() {
  docker exec sglang-qwen3 bash -c "pkill -9 -f 'sglang' || true"
  sleep 6
  docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && nohup python3 -m sglang.launch_server --model-path /data/models/Qwen3-4B-Instruct-2507 --port 30000 --host 0.0.0.0 --tool-call-parser qwen25 --context-length 32768 $1 > $2 2>&1 & sleep 1; echo started"
  for i in $(seq 1 36); do sleep 5; docker exec sglang-qwen3 grep -q "fired up" $2 2>/dev/null && return 0; done
  echo "TIMEOUT $2"; exit 1
}
cd /data/xbw/kvbench
echo "=== FP4 LRU ==="
restart "--kv-cache-dtype fp4_mx_block32" /tmp/server_fp4_lru.log
docker exec sglang-qwen3 grep -oE "max_total_num_tokens=[0-9]+" /tmp/server_fp4_lru.log | head -1
for n in 32 48 64; do echo "--- fp4_lru_n$n"; $B $H --concurrency $n --rounds 6 --seed 7 --out fp4_lru_n$n.jsonl 2>&1 | tail -10; done
echo "=== FP4 SCORE ==="
restart "--kv-cache-dtype fp4_mx_block32 --enable-tool-retention" /tmp/server_fp4_score.log
for n in 32 48 64; do echo "--- fp4_score_n$n"; $B $H --concurrency $n --rounds 6 --seed 7 --out fp4_score_n$n.jsonl 2>&1 | tail -10; done
echo "=== FP4 DONE ==="
