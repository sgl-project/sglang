#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages/torch/lib
rm -rf nohup.out && \
nohup python3 -m sglang.launch_server \
    --attention-backend triton \
    --model-path /code/models/Qwen3-32B/ \
    --log-level info \
    --tp 4 --mem-frac 0.25 \
    --host 0.0.0.0 --port 33301 \
    --enable-metrics --enable-cache-report \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.5 --hicache-size 0 \
    --hicache-io-backend kernel \
    --hicache-mem-layout layer_first \
    --hicache-write-policy write_through \
    &

##################################################

export CONFIG_PATH=/tmp/bench_mix_config.json

# num_clients: Maximum number of concurrent client requests to be simulated
# round_ratios: Distribution of requests across rounds. Given sum(round_ratios) total requests,
#               round_ratios[i] denotes the number of requests that will execute for (i+1) rounds
echo '{
  "num_rounds": 10,
  "num_clients": 60,
  "round_ratios": [50, 25, 15, 15, 10, 10, 9, 8, 7, 6],
  "mean_new_tokens_per_round": [1000, 400, 350, 300, 280, 260, 240, 220, 210, 200],
  "mean_return_tokens_per_round": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
  "mean_inter_round_interval": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
}' > ${CONFIG_PATH}

rm -rf bench_mix.out && \
nohup python3 /sgl-workspace/sglang/benchmark/hicache/bench_mix.py \
    --model-path /code/models/Qwen3-32B/ \
    --dataset-path /code/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --port 33301 \
    --duration 600 \
> bench_mix.out &
