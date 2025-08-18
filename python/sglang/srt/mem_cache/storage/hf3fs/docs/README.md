# Using HF3FS as L3 Global KV Cache

This document provides step-by-step instructions for setting up a k8s + 3FS + SGLang runtime environment from scratch, describing how to utilize deepseek-hf3fs as the L3 KV cache for SGLang.
The process consists of five main steps:

## Step 1: Install deepseek-3fs via 3fs-Operator
Refer to the [3fs-operator documentation](https://github.com/aliyun/kvc-3fs-operator/blob/main/README_en.md) to deploy 3FS components in your Kubernetes environment using the Operator with one-click deployment.

## Step 2: Launch SGLang Pod
Start your SGLang Pod while specifying 3FS-related labels in the YAML configuration. Follow the [fuse-client-creation guide](https://github.com/aliyun/kvc-3fs-operator/blob/main/README_en.md#fuse-client-creation).

## Step 3: Configure Usrbio Client in SGLang Pod
The Usrbio client is required for accessing 3FS. Install it in your SGLang Pod using either method below:

**Alternative 1 (Recommend):** Build from source (refer to [setup_usrbio_client.md](setup_usrbio_client.md))

**Alternative 2:** Run `pip3 install hf3fs-py-usrbio` (Follow https://pypi.org/project/hf3fs-py-usrbio/#files)

## Step 4: Deploy Model Serving

### Single Node Deployment
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages
python3 -m sglang.launch_server \
    --model-path /code/models/Qwen3-32B/ \
    --host 0.0.0.0 --port 10000 \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 0 \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs
```

### Multi-Node Deployment (Shared KV Cache)
Follow the [deploy_sglang_3fs_multinode.md](deploy_sglang_3fs_multinode.md) guide to deploy SGLang with 3FS across multiple nodes for shared KV caching.
