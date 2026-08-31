#!/bin/bash
B=/data/xbw/conda_envs/sglang/bin/python
H=/home/xubowen/mxfp4/sglang/paper_docs/agent_kv_bench/harness.py
docker exec sglang-qwen3 bash -c "pkill -9 -f 'sglang' || true"
sleep 6
docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && nohup python3 -m sglang.launch_server --model-path /data/models/Qwen3-4B-Instruct-2507 --port 30000 --host 0.0.0.0 --tool-call-parser qwen25 --context-length 32768 --enable-tool-retention > /tmp/server_kv_bench_v5score.log 2>&1 & sleep 1; echo started"
for i in $(seq 1 36); do sleep 5; docker exec sglang-qwen3 grep -q "fired up" /tmp/server_kv_bench_v5score.log 2>/dev/null && break; done
cd /data/xbw/kvbench
for n in 16 32; do echo "--- v5_score_n$n"; $B $H --concurrency $n --rounds 6 --seed 7 --out v5_score_n$n.jsonl 2>&1 | tail -11; done
echo "=== V4 DONE ==="
