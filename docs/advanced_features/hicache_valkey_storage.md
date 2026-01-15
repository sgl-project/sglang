# Valkey Storage Backend for HiCache

The Valkey storage backend provides a high-performance, distributed alternative to the default file-based storage for SGLang's HiCache system. Valkey is an open-source, in-memory data store that offers fast cache operations and scalability across multiple instances.

## Features

- **High Performance**: Leverages Valkey's in-memory storage for fast cache operations
- **Scalability**: Supports distributed caching across multiple Valkey instances
- **Full Compatibility**: Implements the complete HiCacheStorage interface
- **Batch Operations**: Optimized batch get/set operations using Valkey pipelines
- **Flexible Configuration**: Configure via environment variables or parameters

## Installation

Install the required Valkey Python client:

```bash
pip install valkey
```

## Configuration

### Environment Variables

Configure the Valkey backend using environment variables:

```bash
export VALKEY_HOST=localhost
export VALKEY_PORT=6379
export VALKEY_DB=0
export VALKEY_PASSWORD=your_password  # Optional
```

### Server Arguments

When launching SGLang server, specify the Valkey backend:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --hicache-storage-backend valkey
```

For more server arguments, see [Server Arguments](server_arguments.md).

## Usage

### Command-Line Usage

Basic server launch with Valkey backend:

```bash
# Set environment variables
export VALKEY_HOST=your-valkey-host
export VALKEY_PORT=6379

# Launch server
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --hicache-storage-backend valkey
```

### Programmatic Usage

Create a Valkey storage backend programmatically:

```python
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

# Create storage configuration
config = HiCacheStorageConfig(
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    is_mla_model=False,
    is_page_first_layout=True,
    model_name="my_model"
)

# Create Valkey storage backend
storage = StorageBackendFactory.create_backend(
    "valkey",
    config,
    mem_pool_host=None,
    host="localhost",
    port=6379,
    db=0
)
```

## Key Features

### Automatic Key Prefixing

Keys are automatically prefixed with model and tensor parallel configuration to avoid conflicts:

- Standard format: `hicache:{model_name}:{tp_rank}:{tp_size}:{key}`
- MLA models: `hicache:{model_name}:{key}`

This ensures that different models and parallel configurations don't interfere with each other's cached data.

### Batch Operations

The Valkey backend uses pipelines for efficient batch operations:

- **`batch_get()`**: Retrieve multiple tensors in a single round-trip
- **`batch_set()`**: Store multiple tensors efficiently
- **`batch_exists()`**: Check existence of multiple keys at once

Batch operations significantly reduce network overhead when working with multiple cache entries.

### Statistics and Monitoring

Built-in statistics reporting includes:

- Key count for current model
- Memory usage (bytes and human-readable)
- Cache hit/miss ratios
- Connected clients

Access statistics through the storage interface to monitor cache performance.

## Performance Considerations

### Memory Usage

Valkey stores all data in memory. Ensure your Valkey instance has sufficient RAM for your cache size:

- Monitor memory usage with `INFO memory` command
- Configure appropriate memory policies (e.g., `maxmemory-policy`)
- Consider using Valkey clustering for large datasets

For more details, see the [Valkey Memory Optimization Guide](https://valkey.io/docs/topics/memory-optimization/).

### Network Latency

For remote Valkey instances:

- Co-locate Valkey with SGLang workers when possible
- Use batch operations to minimize round-trips
- Monitor network latency with Valkey's [latency monitoring](https://valkey.io/topics/latency-monitor/)

### Batch Operations

Always prefer batch operations when working with multiple keys:

```python
# Good: Single batch operation
storage.batch_set(keys, values)

# Avoid: Multiple single operations
for key, value in zip(keys, values):
    storage.set(key, value)
```

### Connection Pooling

The Valkey client automatically handles connection pooling. No additional configuration is needed for basic use cases.

## Troubleshooting

### Connection Issues

If you cannot connect to Valkey:

1. Verify Valkey server is running: `valkey-cli ping`
2. Check firewall settings and network connectivity
3. Validate authentication credentials if using password protection
4. Ensure the correct host and port are configured

For detailed troubleshooting, see the [Valkey Connection Guide](https://valkey.io/docs/topics/clients/).

### Memory Issues

If Valkey runs out of memory:

1. Monitor memory usage: `valkey-cli INFO memory`
2. Configure `maxmemory` and `maxmemory-policy` in Valkey configuration
3. Consider using Valkey clustering for horizontal scaling
4. Review cache eviction policies

For comprehensive memory management, see the [Valkey Memory Optimization Guide](https://valkey.io/docs/topics/memory-optimization/).

### Performance Issues

If experiencing slow cache operations:

1. Use batch operations for multiple keys
2. Monitor network latency to Valkey server
3. Consider co-locating Valkey with SGLang workers
4. Profile with [Valkey's CPU profiling tools](https://valkey.io/topics/performance-on-cpu/)
5. Benchmark with the [Valkey Benchmarking Tool](https://valkey.io/topics/benchmark/)

## Comparison with Other Backends

| Feature | Valkey | File-based |
|---------|--------|------------|
| Performance | High (in-memory) | Moderate (disk I/O) |
| Scalability | Distributed | Single node |
| Persistence | Optional | Built-in |
| Setup Complexity | Requires Valkey server | None |
| Memory Usage | High | Low |

Choose Valkey when:
- You need high-performance caching
- You're running distributed deployments
- You have sufficient memory resources
- You need to share cache across multiple SGLang instances

Choose file-based storage when:
- You're running on a single node
- Memory is limited
- Setup simplicity is important
- Persistent cache is required

## Related Documentation

- [HiCache Overview](hicache_design.md)
- [HiCache Best Practices](hicache_best_practices.md)
- [Server Arguments](server_arguments.md)
