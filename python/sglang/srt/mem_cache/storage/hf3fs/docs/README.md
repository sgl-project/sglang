# HF3FS as L3 KV Cache

This document describes how to use deepseek-hf3fs as the L3 KV cache for SGLang.

## Step1: Install deepseek-3fs by 3fs-Operator (Coming Soon)

## Step2: Setup usrbio client

Please follow the document [setup_usrbio_client.md](setup_usrbio_client.md) to setup usrbio client.

## Step3: Deployment

### Single node deployment

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages
python3 -m sglang.launch_server \
    --model-path /code/models/Qwen3-32B/ \
    --host 0.0.0.0 --port 10000 \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 0 \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs
```

### Multi nodes deployment to share KV cache

Please follow the document [deploy_sglang_3fs_multinode.md](deploy_sglang_3fs_multinode.md) to deploy SGLang with 3FS on multiple nodes to share KV cache.
