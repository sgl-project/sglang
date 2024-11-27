#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

input_len_group=(3072 4096)
batch_group=(1 2 4 8 16)

for input_len in "${input_len_group[@]}"; do
    for batch in "${batch_group[@]}"; do
        python -m sglang.bench_latency --model-path /state/partition/ykchen/pretrain/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.6 --disable-cuda-graph --prefill-mode split --batch $batch --input-len $input_len --output-len 32
        echo $batch $input_len
    done
done
# echo quit | nvidia-cuda-mps-control
# echo quit | nvidia-cuda-mps-control
# echo quit | nvidia-cuda-mps-control