#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# nvidia-cuda-mps-control -d

mps_group=(60)
tp_group=(1)
#1 2 4 8 16 32 64 96 128 192 256 320 384 448 512 
input_len_group=(1024 1536 2048)
batch_group=(1 2 4 8 16 32)

for mps in "${mps_group[@]}"; do
    touch /home/ykchen/sglang/res/breakdown_mps_colocation_$mps.csv
    # echo quit | nvidia-cuda-mps-control
    # export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mps
    # nvidia-cuda-mps-control -d
    for tp in "${tp_group[@]}"; do
        for input_len in "${input_len_group[@]}"; do
            for batch in "${batch_group[@]}"; do
                python -m sglang.bench_latency --model-path /state/partition/ykchen/pretrain/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.6 --tp $tp --batch $batch --input-len $input_len --output-len 32 --mps $mps
                echo $mps $tp $batch $input_len
            done
        done
    done
done
# echo quit | nvidia-cuda-mps-control
# echo quit | nvidia-cuda-mps-control
# echo quit | nvidia-cuda-mps-control