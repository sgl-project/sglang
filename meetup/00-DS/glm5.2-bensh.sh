#!/usr/bin/env bash
set -e

# 修改这里的 DP 数、压测并发数和服务地址。
DP_SIZE=1
CONCURRENCY=1
BENCH_HOST=127.0.0.1
BENCH_PORT=5599

MODEL_PATH=/home/weights/GLM-5.2-w4a8
DATASET_DIR=/home/datasets/completion-accept3-128k-prefix90-4096

# 保留 prefix cache，避免 SGLang 在 CI 模式下自动 flush。
export SGLANG_IS_IN_CI=false

# 第一阶段：服务端启用 prefix cache 和 round_robin，DP_SIZE 与实际 DP 数一致。
# 用 DP_SIZE * 2 条请求预热各 rank 的 KV cache，每条生成 4 token，丢弃预热结果。
echo "正在预热 prefix 到各个 DP rank：DP_SIZE=$DP_SIZE，请求数=$((DP_SIZE * 2))..."
python -m sglang.benchmark.serving \
  --backend sglang \
  --host "$BENCH_HOST" --port "$BENCH_PORT" \
  --model "$MODEL_PATH" \
  --served-model-name GLM-5.2-w4a8 \
  --tokenizer "$MODEL_PATH" \
  --dataset-name custom \
  --dataset-path "$DATASET_DIR/requests-64.jsonl" \
  --sharegpt-output-len 4 \
  --num-prompts "$((DP_SIZE * 2))" \
  --max-concurrency "$DP_SIZE" \
  --temperature 1.1 --top-p 0.95 \
  --seed 42 --warmup-requests 0 \
  --output-file /dev/null \
  > /dev/null 2>&1

# 第二阶段：不 flush cache，发送 CONCURRENCY * 4 条请求，输出正式结果。
echo "Prefix 预热完成，开始正式压测：并发=$CONCURRENCY，请求数=$((CONCURRENCY * 4))..."
python -m sglang.benchmark.serving \
  --backend sglang \
  --host "$BENCH_HOST" --port "$BENCH_PORT" \
  --model "$MODEL_PATH" \
  --served-model-name GLM-5.2-w4a8 \
  --tokenizer "$MODEL_PATH" \
  --dataset-name custom \
  --dataset-path "$DATASET_DIR/requests.jsonl" \
  --sharegpt-output-len 1024 \
  --num-prompts "$((CONCURRENCY * 4))" \
  --max-concurrency "$CONCURRENCY" \
  --temperature 1.1 --top-p 0.95 \
  --seed 42 --warmup-requests 0
