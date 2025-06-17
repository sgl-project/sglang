ps aux | grep "sglang" | grep -v grep | awk '{print $2}' | xargs kill -9

rm -rf nohup.out && \
env CUDA_VISIBLE_DEVICES=7 \
nohup python3 -m sglang.launch_server \
    --model-path /data/models/Qwen3-32B \
    --host 0.0.0.0 --port 33302 \
    \
    --page-size 64 \
    \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 50 \
    --hicache-write-policy write_through \
    \
    --hicache-use-disk \
    --hicache-disk-path /data/sglang_hicache_disk_path \
    --hicache-disk-ratio 4 --hicache-disk-size 200 \
&

python3 benchmark/hicache/bench_multiturn.py \
    --model-path /data/models/Qwen3-32B \
    --dataset-path /data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --port 33302 \
    --num-clients 256 --max-parallel 32

#cache_hit_rate: 0.6614896363279186

Performance metrics summary:
  Total requests: 1280 at 16 requests per second
  Average TTFT: 5.18
  P90 TTFT: 7.24
  Median TTFT: 5.87
  Average latency: 10.27
  P90 latency: 11.53
  Median latency: 10.33
  Throughput: 2.78 requests per second

################################################################################

ps aux | grep "sglang" | grep -v grep | awk '{print $2}' | xargs kill -9

rm -rf nohup.out && \
env CUDA_VISIBLE_DEVICES=7 \
nohup python3 -m sglang.launch_server \
    --model-path /data/models/Qwen3-32B \
    --host 0.0.0.0 --port 33302 \
    \
    --page-size 64 \
    \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 50 \
    --hicache-write-policy write_through \
&

python3 benchmark/hicache/bench_multiturn.py \
    --model-path /data/models/Qwen3-32B \
    --dataset-path /data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --port 33302 \
    --num-clients 256 --max-parallel 32

#cache_hit_rate: 0.381675786824275

Performance metrics summary:
  Total requests: 1280 at 16 requests per second
  Average TTFT: 9.04
  P90 TTFT: 17.46
  Median TTFT: 7.42
  Average latency: 17.25
  P90 latency: 25.27
  Median latency: 18.10
  Throughput: 1.73 requests per second
