# SGLang HiCache Best Practices

## Why HiCache Matters

SGLang HiCache extends the traditional RadixAttention with a three-tier hierarchical KV caching system that dramatically improves performance for long-context and multi-turn conversation scenarios. By intelligently managing KV caches across GPU memory, host memory, and external storage backends, HiCache addresses the fundamental capacity bottleneck that limits cache hit rates in conventional systems.

### Real-World Performance Gains

Based on community feedback and benchmarks:

- **Coding Agent Scenarios**: In Qwen3-Coder-480B deployments with 25K+ token dialogues, integrating HiCache with DeepSeek 3FS achieved:
  - **56% reduction in TTFT** (Time to First Token)
  - **2× inference throughput improvement**
  - **Cache hit rate increased from 40% to 80%**

- **General QA Scenarios**: DeepSeek-R1-671B with Mooncake backend showed:
  - **84% reduction in TTFT** compared to full re-computation

## Architecture Overview

HiCache implements a three-tier memory hierarchy:

![HiCache Architecture Overview](https://lmsys.org/images/blog/hicache/hicache_overview.png)

*Figure 1: SGLang HiCache three-tier architecture with GPU memory (L1), host memory (L2), and storage backend (L3)*

### Key Components

1. **HiRadixTree**: Acts as a page table for referencing KV caches across all tiers
2. **Cache Controller**: Automatically manages data movement between tiers
3. **Storage Backends**: Pluggable interfaces for various storage solutions
4. **Optimized Data Plane**: GPU-assisted I/O kernels and zero-copy mechanisms

## Supported Storage Backends

HiCache supports multiple storage backends through a clean, generic interface:

### Production-Ready Backends

1. **DeepSeek 3FS (HF3FS)**: Kubernetes-native distributed storage solution with operator-based deployment

2. **Mooncake**: High-performance distributed KV cache with RDMA support and zero-copy transfers

3. **AIBrix KVCache**: Enterprise-grade distributed caching supporting multiple backends (Infinistore, PrisKV)

4. **NIXL**: Transfer library that bridges various storage backends including GPU-direct storage and cloud object storage

### Development/Testing Backend

5. **HiCacheFile**: Simple file-based storage for development, testing, and single-node deployments

## Configuration Guidelines

### Core HiCache Parameters

```bash
# Essential HiCache flags
--enable-hierarchical-cache         # Enable HiCache
--hicache-ratio 2                   # Host memory ratio (2x GPU memory)
--page-size 64                      # Page size for cache management
--hicache-storage-backend           # Storage backend (e.g., hf3fs, mooncake, etc.)
```

### Memory Layout Optimization

```bash
# Page-first: Optimized for I/O efficiency with zero-copy (recommended)
--hicache-mem-layout page_first

# Layer-first
--hicache-mem-layout layer_first
```

### Prefetch Policies

```bash
# Best-effort: Terminate prefetch when needed
--hicache-storage-prefetch-policy best_effort

# Wait-complete: Ensure complete prefetch, higher cache reuse
--hicache-storage-prefetch-policy wait_complete

# Timeout: Balance between completion and best-effort
--hicache-storage-prefetch-policy timeout
```

## Deployment Guidelines

### Deployment with HF3FS

Here is an example of deploying DeepSeek-R1 with HiCache-HF3FS. For more details, see the [HF3FS Documentation](../../python/sglang/srt/mem_cache/storage/hf3fs/docs/README.md).

```bash
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --log-level info \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
```

### Deployment with Mooncake

Here is an example of deploying Qwen3-235B-A22B-Instruct-2507 with Mooncake. For more details, see the [Mooncake Documentation](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md).

```bash
# Set Mooncake environment variables
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=127.0.0.1:50051

# Launch SGLang server with Mooncake backend
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp 8 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy timeout
```

### Integration with PD Disaggregation

HiCache works seamlessly with PD Disaggregation，You can choose between two configurations to deploy HICache and PD:

1. Enable HICache only between P nodes, which allows KVCache sharing among the P nodes.
2. Enable HICache on P nodes and asynchronous offloading of KVCache on D nodes. This configuration allows P nodes to share the KVCache from D nodes in multi-turn dialogue scenarios.

```bash
# Prefill node with HiCache enabled for cross-prefill sharing (ideal for SystemPrompt scenarios)
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake

# Enabling asynchronous offloading in Decode Node allows its KVCache to be contributed by Prefill
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-decode-enable-offload-kvcache \  # Enable async KV cache offloading in decode node
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake
```

## Custom Storage Backend Integration

To integrate a new storage backend:

1. **Implement three core methods:**
   - `get(key)`: Retrieve value by key
   - `exists(key)`: Check key existence  
   - `set(key, value)`: Store key-value pair

2. **Register your backend:** Add your storage backend to the HiCache [BackendFactory](../../python/sglang/srt/mem_cache/storage/backend_factory.py#L188)

The HiCache controller handles all scheduling and synchronization automatically.

## Community and Support

- **GitHub Issues**: Report bugs and feature requests
- **Slack Channel**: Join community discussions in #sgl-kv-cache-store
- **Documentation**: Refer to storage backend-specific guides

---

*This document will be continuously updated based on community feedback and new features. Contributions and suggestions are welcome!*