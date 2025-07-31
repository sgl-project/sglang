#!/bin/bash

# SGLang EAGLE Benchmark Script
# This script runs the offline throughput benchmark with EAGLE speculative decoding

python3 -m sglang.bench_offline_throughput \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-prompts 10 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --speculative-num-draft-tokens 64 \
  --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt \
  --mem-fraction 0.7 \
  --cuda-graph-max-bs 2 \
  --dtype float16 \
  --log-level info \
  --disable-cuda-graph \
  --log-cold-token-prob