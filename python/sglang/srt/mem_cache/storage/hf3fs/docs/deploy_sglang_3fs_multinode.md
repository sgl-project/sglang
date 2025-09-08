# 1. Startup 3fs metadata service
```bash
nohup python3 -m sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server > meta.out &
```


# 2. Startup sglang engine
## HF3fs configures
```bash
vim /sgl-workspace/sglang/benchmark/hf3fs/hf3fs_config.json
{
    "file_path_prefix": "/data/hicache",
    "file_size": 1099511627776,
    "numjobs": 16,
    "entries": 8,
    "metadata_server_url": "http://metaServerIp:18000"
}
```

## node1
```bash
export SGLANG_HICACHE_HF3FS_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hf3fs/hf3fs_config.json
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages
rm -rf instance1.out && \
nohup python3 -m sglang.launch_server \
    --model-path /code/models/Qwen3-32B/ \
    --host 0.0.0.0 --port 10000 \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 0 \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs --tp 2 > instance1.out &
```

## node2
```bash
export SGLANG_HICACHE_HF3FS_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hf3fs/hf3fs_config.json
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages
rm -rf instance2.out && \
nohup python3 -m sglang.launch_server \
    --model-path /code/models/Qwen3-32B/ \
    --host 0.0.0.0 --port 10000 \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 0 \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs --tp 2 > instance2.out &
```

# 3. Startup router
```bash
rm -rf router.out && \
nohup python -m sglang_router.launch_router --worker-urls http://node1:10000 http://node2:10000 > router.out &
```

# 4. Startup multiturn benchmark
```bash
rm -rf bench_multiturn.out && \
nohup python3 benchmark/hicache/bench_multiturn.py \
    --model-path /code/models/Qwen3-32B \
    --dataset-path /code/models/ShareGPT_V3_unfiltered_cleaned_split.json \
    --port 30000 \
    --request-length 2048 --num-clients 512 --num-rounds 5 --max-parallel 8 \
    > bench_multiturn.out &
```
