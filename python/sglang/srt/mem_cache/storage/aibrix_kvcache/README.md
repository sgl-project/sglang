# AIBrix KVCache as L3 KV Cache
This document provides brief instructions for setting up a AIBrixKVCache storage backend +  AIBrixKVCache + SGLang runtime environment from scratch, describing how to utilize AIBrixKVCache as the L3 KV cache for SGLang.
The process consists of three main steps:

## Step1:Install AIbrix KVCache
Refer to the [AIBrix KVCache documentation](https://github.com/vllm-project/aibrix/blob/main/python/aibrix_kvcache/README.md) to install  AIBrix KVCache.

## Step2: Deploy AIBrix Distributed KVCache Storage

AIBrix KVCache currently supports multiple distributed KVCache backends, including ByteDance's open-source Infinistore and the not-yet-open source PrisKV incubated by ByteDance's PrisDB & IAAS & DMI team.

For the Infinistore installation process, please refer to [this link](https://github.com/bytedance/InfiniStore).

PrisKV for AIBrix KVCache is currently in the open-source preparation stage, and no public documentation is available yet.


## Step3: Deploy Model Serving

For information on configuring a distributed KVCache backend for AIBrixKVCache, please refer to [this link](https://aibrix.readthedocs.io/latest/designs/aibrix-kvcache-offloading-framework.html)

Using PrisKV as an example, the startup command is as follows:
```bash
export AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED="0"
export AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND="PRIS"
export AIBRIX_KV_CACHE_OL_PRIS_REMOTE_ADDR="127.0.0.1"
export AIBRIX_KV_CACHE_OL_PRIS_REMOTE_PORT="6379"
export AIBRIX_KV_CACHE_OL_PRIS_PASSWORD="kvcache-redis"
MODEL_LENGTH=32768&&NCCL_MIN_NCHANNELS=24&&NCCL_IB_QPS_PER_CONNECTION=8&&NCCL_DEBUG=INFO \
python3 -m sglang.launch_server \
	--model-path /code/models/Qwen3-32B \
	--host 0.0.0.0 --port 8080 \
	--enable-hierarchical-cache \
	--hicache-storage-backend aibrix \
	--page-size 16 \
	--hicache-write-policy write_back \
	--enable-metrics --hicache-ratio=2
```
