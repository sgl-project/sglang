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
echo "=== D4a: bf16 envelope sweep (LRU, seed 7) ==="
restart "" /tmp/server_d4a.log
for n in 20 24 28; do echo "--- d4_bf16_lru_n$n"; $B $H --concurrency $n --rounds 6 --seed 7 --out d4_bf16_lru_n$n.jsonl 2>&1 | tail -3; done
echo "=== D4b: fp4 envelope sweep (LRU, seed 7) ==="
restart "--kv-cache-dtype fp4_mx_block32" /tmp/server_d4b.log
for n in 40 44 52 56; do echo "--- d4_fp4_lru_n$n"; $B $H --concurrency $n --rounds 6 --seed 7 --out d4_fp4_lru_n$n.jsonl 2>&1 | tail -3; done
echo "=== D4c: latency sensitivity (bf16 n16 + fp4 n32, LRU seed 7) ==="
restart "" /tmp/server_d4c.log
echo "--- d4_bf16_lru_n16_lat0.5"; $B $H --concurrency 16 --rounds 6 --seed 7 --latency-scale 0.5 --out d4_bf16_lru_n16_lat05.jsonl 2>&1 | tail -3
echo "--- d4_bf16_lru_n16_lat2"; $B $H --concurrency 16 --rounds 6 --seed 7 --latency-scale 2 --out d4_bf16_lru_n16_lat2.jsonl 2>&1 | tail -3
restart "--kv-cache-dtype fp4_mx_block32" /tmp/server_d4d.log
echo "--- d4_fp4_lru_n32_lat0.5"; $B $H --concurrency 32 --rounds 6 --seed 7 --latency-scale 0.5 --out d4_fp4_lru_n32_lat05.jsonl 2>&1 | tail -3
echo "--- d4_fp4_lru_n32_lat2"; $B $H --concurrency 32 --rounds 6 --seed 7 --latency-scale 2 --out d4_fp4_lru_n32_lat2.jsonl 2>&1 | tail -3
echo "=== D4d: seed 9 headline ==="
restart "" /tmp/server_d4e.log
echo "--- s9_bf16_lru_n16"; $B $H --concurrency 16 --rounds 6 --seed 9 --out s9_bf16_lru_n16.jsonl 2>&1 | tail -3
restart "--enable-tool-retention" /tmp/server_d4f.log
echo "--- s9_bf16_score_n16"; $B $H --concurrency 16 --rounds 6 --seed 9 --out s9_bf16_score_n16.jsonl 2>&1 | tail -3
restart "--kv-cache-dtype fp4_mx_block32" /tmp/server_d4g.log
echo "--- s9_fp4_lru_n32"; $B $H --concurrency 32 --rounds 6 --seed 9 --out s9_fp4_lru_n32.jsonl 2>&1 | tail -3
restart "--kv-cache-dtype fp4_mx_block32 --enable-tool-retention" /tmp/server_d4h.log
echo "--- s9_fp4_score_n32"; $B $H --concurrency 32 --rounds 6 --seed 9 --out s9_fp4_score_n32.jsonl 2>&1 | tail -3
echo "=== D4 DONE ==="
