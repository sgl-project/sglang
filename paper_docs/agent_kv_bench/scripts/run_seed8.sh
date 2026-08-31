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
restart "" /tmp/server_s8a.log
echo "--- s8_bf16_lru_n16"; $B $H --concurrency 16 --rounds 6 --seed 8 --out s8_bf16_lru_n16.jsonl 2>&1 | tail -3
echo "--- s8_bf16_score_n16"
restart "--enable-tool-retention" /tmp/server_s8b.log
$B $H --concurrency 16 --rounds 6 --seed 8 --out s8_bf16_score_n16.jsonl 2>&1 | tail -3
restart "" /tmp/server_s8c.log
echo "--- s8_bf16_lru_n32"; $B $H --concurrency 32 --rounds 6 --seed 8 --out s8_bf16_lru_n32.jsonl 2>&1 | tail -3
restart "--enable-tool-retention" /tmp/server_s8d.log
echo "--- s8_bf16_score_n32"; $B $H --concurrency 32 --rounds 6 --seed 8 --out s8_bf16_score_n32.jsonl 2>&1 | tail -3
restart "--kv-cache-dtype fp4_mx_block32" /tmp/server_s8e.log
echo "--- s8_fp4_lru_n32"; $B $H --concurrency 32 --rounds 6 --seed 8 --out s8_fp4_lru_n32.jsonl 2>&1 | tail -3
restart "--kv-cache-dtype fp4_mx_block32 --enable-tool-retention" /tmp/server_s8f.log
echo "--- s8_fp4_score_n32"; $B $H --concurrency 32 --rounds 6 --seed 8 --out s8_fp4_score_n32.jsonl 2>&1 | tail -3
echo "=== SEED8 DONE ==="
