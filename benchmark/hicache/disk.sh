ps aux | grep "sglang" | grep -v grep | awk '{print $2}' | xargs kill -9

rm -rf nohup.out && \
nohup python3 -m sglang.launch_server \
    --model-path /data/models/Qwen3-8B \
    --host 0.0.0.0 --port 33301 \
    \
    --page-size 64 \
    \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 0 \
    --hicache-write-policy write_through \
    \
    --hicache-use-disk \
    --hicache-disk-path /data/sglang_hicache_disk_path \
    --hicache-disk-ratio 4 --hicache-disk-size 0 \
&

python3 benchmark/hicache/bench_multiturn.py \
    --model-path /data/models/Qwen3-8B \
    --dataset-path /data/models/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --port 33301
