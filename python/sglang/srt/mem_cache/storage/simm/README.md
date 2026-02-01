# SiMM as L3 KV Cache

This document describes how to use SiMM as the L3 KV cache for SGLang.

## About SiMM

SiMM(Scitix In-Memory Middleware) is a distributed, high-performance, elastic cache acceleration layer for all AI workloads.

For more details about SiMM, please refer to [SiMM project]() and [SiMM documents]().

### SiMM & SGLang HiCache

SiMM serves as a high-performance L3 storage backend for SGLang HiCache, enabling distributed KV cache storage across multiple servers with RDMA-baed transport. This integration addresses the capacity limitations of traditional GPU-only or GPU+CPU caching by providing virtually unlimited cache storage through a distributed memory pool.

When a cache miss occurs in L1 and L2, HiCache automatically fetches the required KV cache from SiMM's distributed memory pool. The system uses intelligent prefetching strategies to minimize latency, and utilize RDMA technology and zero-copy technique to ensure high-bandwidth, low-latency data transfer between SGLang instances and SiMM data servers.

## Install SiMM

**from source**

For more details, please refer to [SiMM official installation guide]().

## Deployment

**SiMM**

Before launch `SGLang server` with SiMM, you should launch SiMM `cluster manager service` and `data server service`.

You can visit [SiMM official deploy guide]() and deploy SiMM on your K8S cluster with RDMA network.

**Start the `SGLang server` with SiMM enabled:**

There are three ways to configure SiMM:

1. Via extra configuration passed through sglang parameters
2. Using JSON configuration files
3. Using environment variables

SiMM loads configuration in the following priority order:

1. If SiMM-specific options are provided in `--hicache-storage-backend-extra-config`, they are used first.
2. If not, SiMM checks whether the environment variable `DEFAULT_SIMM_CONFIG_PATH_ENV` is set, and loads the JSON config file from that path.
3. If neither of the above is provided, SiMM falls back to environment variables.

**HiCache Related Parameters for SGLang Server**

For a comprehensive overview of HiCache-related parameters, please refer to [this document](https://docs.sglang.io/advanced_features/hicache_design.html#related-parameters).


Note that, for `--hicache-mem-layout {layer_first,page_first,page_first_direct}`, which specifies the memory layout for the host memory pool, `page_first` or `page_first_direct` are required if use SiMM backend.

### Distributed Deployment

**Using extra-config of sglang arguments to configure SiMM**

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend simm \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config '{"manager_address": "127.0.0.1:30001"}'
```

**Using JSON file to configure SiMM**

SGLang server can load SiMM config from `SGLANG_HICACHE_SIMM_CONFIG_PATH`.

```bash
export SGLANG_HICACHE_SIMM_CONFIG_PATH=/sgl-workspace/sglang/benchmark/hicache/simm_config.json

echo '{
    "manager_address": "127.0.0.1:30001"
}' > ${SGLANG_HICACHE_SIMM_CONFIG_PATH}

python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend simm \
    --model-path [model_path]
```

**Using env variables to configure SiMM**

```bash
SIMM_CLUSTER_MANAGER="127.0.0.1:30001"
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend simm \
    --model-path [model_path]
```

## Test SiMM

This test is intended for developers to quickly verify that the SiMM class interfaces are functioning correctly.

First, start the `cluster manager service` and `data server service`. Then run the `test_hicache_simm.py`.

```bash
SIMM_CLUSTER_MANAGER="127.0.0.1:30001" \
python3 [path of test_hicache_simm.py]
```

If all tests pass, the message "✅ All tests passed" will be printed at the end.
