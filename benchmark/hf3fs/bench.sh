SGLANG_HICACHE_HF3FS_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hf3fs/hf3fs.json \
python3 benchmark/hf3fs/bench_storage.py

####################################################################################################

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

####################################################################################################

rm -rf nohup.out && \
nohup python3 -m sglang.launch_server \
    --model-path /code/models/DeepSeek-R1/ \
    --tp 16 --nnodes 2 --node-rank 0 \
    --dist-init-addr 10.74.249.153:5000 \
    --host 0.0.0.0 --port 33301 \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 60 \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs &

rm -rf bench_multiturn.out && \
nohup python3 benchmark/hicache/bench_multiturn.py \
    --model-path /code/models/Qwen3-32B \
    --dataset-path /code/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --port 33301 \
    --request-length 2048 --num-clients 1024 --num-rounds 3 --max-parallel 8 \
    > bench_multiturn.out &

####################################################################################################

ps aux | grep "sglang.launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep "bench_multiturn.py" | grep -v grep | awk '{print $2}' | xargs kill -9
