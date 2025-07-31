#!/bin/bash

# Weights & Biases Launch Script for SGLang EAGLE Benchmark
# This script launches the benchmark with wandb tracking

wandb launch \
  --uri . \
  --project sglang-perf \
  --name "eagle-llama3-8b-benchmark" \
  --entry-point "bash run_eagle_benchmark.sh"