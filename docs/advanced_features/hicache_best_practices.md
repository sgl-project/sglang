# SGLang HiCache Best Practices

## Why HiCache Matters

SGLang HiCache extends the traditional RadixAttention with a three-tier hierarchical KV caching system that dramatically improves performance for long-context and multi-turn conversation scenarios. By intelligently managing KV caches across GPU memory, host memory, and external storage backends, HiCache addresses the fundamental capacity bottleneck that limits cache hit rates in conventional systems.

## Configuration Guidelines

## Core HiCache Parameters

```bash
# Essential HiCache flags
--page-size 64                        # Page size for cache management
--enable-hierarchical-cache           # Enable HiCache
--hicache-ratio 2                     # Host memory ratio (2x GPU memory)
--hicache-size 100                    # Host memory size in GBs, will override the above ratio
--hicache-io-backend kernel           # The I/O backend of moving data between CPU and GPU
--hicache-write-policy write_through  # Cache write policy from GPU to CPU
--hicache-storage-backend             # Optional storage backend (e.g., hf3fs, mooncake, etc.)
```

## Key Configurations with Storage Backends Enabled

### Memory Layout Optimization

```bash
# Page-first: Optimized for I/O efficiency with zero-copy (recommended with kernel backend)
--hicache-mem-layout page_first
# Page-first-direct: Optimized for direct I/O operations (Compatible with fa3 and same zero-copy performance as page_first)
--hicache-mem-layout page_first_direct
# Layer-first
--hicache-mem-layout layer_first
```
**Layout Compatibility:**
- `page_first`: Only compatible with `kernel` I/O backend, automatically switches to `layer_first` with `direct` backend
- `page_first_direct`: Specifically designed for `direct` I/O backend with optimized memory organization

### Prefetch Policies

```bash
# Best-effort: Terminate prefetch when needed
--hicache-storage-prefetch-policy best_effort
# Wait-complete: Ensure complete prefetch, higher cache reuse
--hicache-storage-prefetch-policy wait_complete
# Timeout: Balance between completion and best-effort
--hicache-storage-prefetch-policy timeout
```

### Integration with PD Disaggregation

HiCache works seamlessly with PD Disaggregation. You can choose between two configurations:

1. **Prefill-only HiCache**: Enable HiCache only on Prefill nodes, allowing KV cache sharing among Prefill instances
2. **Full HiCache with async offloading**: Enable HiCache on Prefill nodes and async KV cache offloading on Decode nodes, allowing Prefill nodes to reuse KV caches from Decode nodes in multi-turn dialogue scenarios

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
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake

# Decode node with async offloading enabled for KV cache reuse by Prefill (ideal for multi-turn conversations)
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
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-decode-enable-offload-kvcache \  # Enable async KV cache offloading in decode node
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake
```


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
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
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
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```


## Custom Storage Backend Integration

To integrate a new storage backend:

1. **Implement three core methods:**
   - `get(key)`: Retrieve value by key
   - `exists(key)`: Check key existence
   - `set(key, value)`: Store key-value pair

2. **Register your backend:** Add your storage backend to the HiCache [BackendFactory](../../python/sglang/srt/mem_cache/storage/backend_factory.py#L188)

The HiCache controller handles all scheduling and synchronization automatically.

### Dynamic Backend Loading

Alternatively, you can use dynamic loading to avoid hard-coding your backend in the repository:

```bash
python3 -m sglang.launch_server \
  --model-path your-model \
  --enable-hierarchical-cache \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{"backend_name":"custom_backend_name", "module_path": "your_module_path", "class_name": "YourHiCacheClassName"}'
```

**Configuration Parameters:**
- `--hicache-storage-backend`: Set to `dynamic`
- `--hicache-storage-backend-extra-config`: JSON configuration with:
  - `backend_name`: Custom backend identifier
  - `module_path`: Python module path to your implementation
  - `class_name`: Your HiCache implementation class name


## Community and Support

- **GitHub Issues**: Report bugs and feature requests
- **Slack Channel**: Join community discussions in #sgl-kv-cache-store
- **Documentation**: Refer to storage backend-specific guides

---

*This document will be continuously updated based on community feedback and new features. Contributions and suggestions are welcome!*
