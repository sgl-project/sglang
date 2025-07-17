SGLANG_HICACHE_HF3FS_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hf3fs/hf3fs.json \
python3 benchmark/hf3fs/bench_storage.py

rm -rf nohup.out && \
nohup python3 -m sglang.launch_server \
    --model-path /code/models/Qwen3-32B/ \
    --host 0.0.0.0 --port 33301 \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 0 \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs &

rm -rf bench_multiturn.out && \
nohup python3 benchmark/hicache/bench_multiturn.py \
    --model-path /code/models/Qwen3-32B \
    --dataset-path /code/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --port 33301 \
    --request-length 2048 --num-clients 512 --num-rounds 3 --max-parallel 8 \
    > bench_multiturn.out &

########## --hicache-storage-backend hf3fs ##########
#Input tokens: 3145728
#Output tokens: 98304
100%|██████████| 1536/1536 [40:33<00:00,  1.58s/it]
All requests completed
Performance metrics summary:
  Total requests: 1536 at 16 requests per second
  Average TTFT: 6.73
  P90 TTFT: 11.04
  Median TTFT: 6.90
  Average latency: 12.39
  P90 latency: 15.42
  Median latency: 12.06
  Throughput: 0.63 requests per second

########## --hicache-storage-backend none ##########
#Input tokens: 3145728
#Output tokens: 98304
100%|██████████| 1536/1536 [57:39<00:00,  2.25s/it]
All requests completed
Performance metrics summary:
  Total requests: 1536 at 16 requests per second
  Average TTFT: 10.62
  P90 TTFT: 18.13
  Median TTFT: 10.19
  Average latency: 17.71
  P90 latency: 23.70
  Median latency: 18.56
  Throughput: 0.45 requests per second
