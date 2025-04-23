#!/bin/bash

NUM_GPUS=8
OUTPUT_FILE="benchmark_results.json"

torchrun --nproc_per_node=${NUM_GPUS} benchmark_fp8_cutlass_moe.py \
    --output-file ${OUTPUT_FILE} \
    "$@"

echo "All benchmarks completed. Analyzing results..."
python analyze_results.py --input-file ${OUTPUT_FILE} --merge 