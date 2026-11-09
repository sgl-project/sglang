#!/bin/bash
# A/B the fused GDN decode unpack + Conv1D path (PR #47287) end to end.
# Usage: MODEL=<qwen3.5 path> TP=8 bash run_gdn_decode_fused_ab.sh
MODEL=${MODEL:?set MODEL}
TP=${TP:-8}
PORT=${PORT:-30000}

for FUSED in 0 1; do
  SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=$FUSED \
  python3 -m sglang.launch_server --model-path $MODEL --tp $TP --port $PORT \
    --trust-remote-code > server_fused${FUSED}.log 2>&1 &
  PID=$!
  until curl -s localhost:$PORT/health_generate > /dev/null; do sleep 5; done

  # accuracy
  python3 -m sglang.test.few_shot_gsm8k --port $PORT --num-questions 1319 \
    --parallel 128 --num-shots 5 2>&1 | tee gsm8k_fused${FUSED}.log

  # decode throughput / TPOT at several concurrencies
  for CONC in 1 4 16 64 128; do
    python3 -m sglang.bench_serving --backend sglang --port $PORT --model $MODEL \
      --dataset-name random --random-input-len 1024 --random-output-len 1024 \
      --max-concurrency $CONC --num-prompts $((CONC * 4)) --warmup-requests $CONC \
      2>&1 | tee bench_fused${FUSED}_c${CONC}.log
  done

  kill $PID; wait $PID 2>/dev/null
done

grep -H -E "Output token throughput|Mean TPOT" bench_fused*_c*.log
grep -H "Accuracy" gsm8k_fused*.log
