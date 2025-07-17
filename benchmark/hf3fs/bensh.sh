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
