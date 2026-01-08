# HiCacheValkeyStorage - Valkey Backend for SGLang

This implementation provides a Valkey-based storage backend for SGLang's HiCache system, serving as an alternative to the default file-based storage.

## Features

- **High Performance**: Leverages Valkey's in-memory storage for fast cache operations
- **Scalability**: Supports distributed caching across multiple Valkey instances
- **Compatibility**: Implements the full HiCacheStorage interface
- **Batch Operations**: Optimized batch get/set operations using Valkey pipelines
- **Configuration**: Flexible configuration via environment variables or parameters

## Installation

Install the required Valkey Python client:

```bash
pip install valkey
```

## Configuration

The Valkey backend can be configured through environment variables:

```bash
export VALKEY_HOST=localhost
export VALKEY_PORT=6379
export VALKEY_DB=0
export VALKEY_PASSWORD=your_password  # Optional
```

Or pass parameters directly when creating the storage instance.

## Usage

### Basic Usage

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

### Using with SGLang Server

To use Valkey as the HiCache storage backend when running SGLang server:

```bash
# Set environment variables
export VALKEY_HOST=your-valkey-host
export VALKEY_PORT=6379

# Run SGLang server with Valkey backend
python -m sglang.launch_server \
    --model-path your-model \
    --hicache-backend valkey
```

## Key Features

### Automatic Key Prefixing
Keys are automatically prefixed with model and tensor parallel configuration to avoid conflicts:
- Format: `hicache:{model_name}:{tp_rank}:{tp_size}:{key}`
- For MLA models: `hicache:{model_name}:{key}`

### Batch Operations
Optimized batch operations using Valkey pipelines for better performance:
- `batch_get()`: Retrieve multiple tensors in a single round-trip
- `batch_set()`: Store multiple tensors efficiently
- `batch_exists()`: Check existence of multiple keys

### Error Handling
Comprehensive error handling with detailed logging for debugging and monitoring.

### Statistics
Built-in statistics reporting including:
- Key count for current model
- Memory usage
- Cache hit/miss ratios
- Connected clients

## Performance Considerations

1. **Memory Usage**: Valkey stores data in memory, ensure sufficient RAM
2. **Network Latency**: For remote Valkey instances, consider network latency
3. **Batch Operations**: Use batch operations when possible for better throughput
4. **Connection Pooling**: Valkey client handles connection pooling automatically

## Troubleshooting

### Connection Issues
For detailed connection troubleshooting, see the [Valkey Connection Guide](https://valkey.io/docs/topics/clients/).

- Verify Valkey server is running and accessible
- Check firewall settings and network connectivity
- Validate authentication credentials

### Memory Issues
For comprehensive memory management information, see the [Valkey Memory Optimization Guide](https://valkey.io/docs/topics/memory-optimization/).

- Monitor Valkey memory usage with `INFO memory`
- Configure appropriate memory policies in Valkey
- Consider using Valkey clustering for large datasets

### Performance Issues
For performance benchmarking, see the [Valkey Benchmarking Tool](https://valkey.io/topics/benchmark/)

- Use batch operations for multiple keys
- [Monitor network latency](https://valkey.io/topics/latency-monitor/) to Valkey server
- Consider co-locating Valkey with SGLang workers
- Use [CPU profiling](https://valkey.io/topics/performance-on-cpu/) to find bottlenecks

## Testing

The implementation includes a comprehensive test using Python's unittest framework:

### Run Unit Tests
```bash
# Run the unittest suite
cd /path/to/sglang/python
VALKEY_HOST=your-valkey-host python sglang/srt/mem_cache/storage/valkey/test_hicache_valkey_storage.py
```

### Test Coverage
The unit tests cover:
- Basic set/get operations
- Key existence checks
- Batch operations (set, get, exists)
- Statistics reporting
- Storage clearing
- Error handling
- MLA model key prefixing
- Edge cases (empty inputs, nonexistent keys)

All tests verify functionality against a live Valkey server.
