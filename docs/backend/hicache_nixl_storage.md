# HiCache NIXL Storage Connector

The HiCache NIXL Storage Connector provides high-performance storage capabilities for SGLang's hierarchical cache system using NIXL (Network Interface for eXchange and Learning) storage endpoints.

## Overview

The `HiCacheNixl` class implements the `HiCacheStorage` interface and leverages NIXL's efficient data transfer capabilities for storing and retrieving KV cache data. This connector enables high-performance, distributed storage for hierarchical cache operations with optimized batch operations and pre-opened files.

## Features

- **High-performance storage**: Leverages NIXL's optimized data transfer protocols
- **Batch operations**: Efficient batch get/set operations with minimal overhead
- **Pre-opened files**: Files are pre-opened at initialization for better performance
- **Smart tensor registration**: Only registers unregistered tensors to avoid redundancy
- **GPU/CPU memory detection**: Automatically detects and registers GPU/CPU tensors appropriately
- **File plugin selection**: Support for different NIXL file plugins (GDS_MT, 3FS, POSIX)
- **Error handling**: Graceful handling of network and storage failures

## Installation

### Prerequisites

1. Install NIXL following the official instructions:
   ```bash
   pip install nixl
   ```

   Or build from source:
   ```bash
   git clone https://github.com/ai-dynamo/nixl.git
   cd nixl
   pip install -e .
   ```

2. Ensure you have a NIXL storage endpoint running (or configure one)

## Usage

### Basic Usage

To use the NIXL storage backend with SGLang, launch the server with the following arguments:

```bash
python -m sglang.launch_server \
    --model-path your-model-path \
    --enable-hierarchical-cache \
    --hicache-storage-backend nixl \
    --hicache-size 100 \
    --page-size 64
```

### Configuration Options

The NIXL storage connector supports the following configuration options:

- `file_path`: Path to the storage directory for NIXL file operations (default: "/tmp/hicache_nixl")
- `file_plugin`: Plugin to use ("auto", "GDS_MT", "3FS", "POSIX"). "auto" selects the best available plugin automatically.

### Example Configuration

```python
from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl

# Basic initialization with auto plugin selection
storage = HiCacheNixl()

# With custom file path
storage = HiCacheNixl(file_path="/tmp/hicache_nixl_custom")

# With specific plugin
storage = HiCacheNixl(file_plugin="GDS_MT")

# With both custom file path and plugin
storage = HiCacheNixl(file_path="/tmp/hicache_nixl_custom", file_plugin="3FS")
```

## API Reference

### HiCacheNixl Class

#### Constructor

```python
HiCacheNixl(file_path: str = "/tmp/hicache_nixl", file_plugin: str = "auto")
```

**Parameters:**
- `file_path`: Path to the storage directory for NIXL file operations
- `file_plugin`: Plugin to use ("auto", "GDS_MT", "3FS", "POSIX"). "auto" selects the best available plugin automatically.

#### Methods

##### `get(key: str, dst_tensor: torch.Tensor) -> torch.Tensor | None`

Retrieve the value associated with the given key from NIXL storage.

**Parameters:**
- `key`: Storage key
- `dst_tensor`: Destination tensor buffer for the transfer. Required for pure NIXL operation.

**Returns:**
- `torch.Tensor | None`: Retrieved tensor or None if key doesn't exist

**Note:** The `dst_tensor` parameter is required for pure NIXL datapath operations. The tensor data will be transferred directly into this buffer.

##### `batch_get(keys: List[str], dst_tensors: List[torch.Tensor]) -> List[torch.Tensor | None]`

Retrieve values for multiple keys from NIXL storage using batch operations.

**Parameters:**
- `keys`: List of storage keys
- `dst_tensors`: List of destination tensor buffers for the transfers

**Returns:**
- `List[torch.Tensor | None]`: List of retrieved tensors or None for each key

**Note:** The `dst_tensors` parameter is required. Tensor data will be transferred directly into these buffers.

##### `set(key: str, value: torch.Tensor, overwrite: bool = False) -> bool`

Store the value associated with the given key in NIXL storage.

**Parameters:**
- `key`: Storage key
- `value`: Tensor to store
- `overwrite`: If True, overwrite existing key. If False, skip if key exists.

**Returns:**
- `bool`: True if operation was successful, False otherwise

##### `batch_set(keys: List[str], values: List[torch.Tensor], overwrite: bool = False) -> bool`

Store multiple key-value pairs in NIXL storage using batch operations.

**Parameters:**
- `keys`: List of storage keys
- `values`: List of tensors to store
- `overwrite`: If True, overwrite existing keys. If False, skip if key exists.

**Returns:**
- `bool`: True if all operations were successful, False otherwise

##### `exists(key: str) -> bool`

Check if the key exists in NIXL storage.

**Parameters:**
- `key`: Storage key

**Returns:**
- `bool`: True if key exists, False otherwise

##### `delete(key: str) -> None`

Delete the key from NIXL storage.

**Parameters:**
- `key`: Storage key to delete

##### `clear() -> None`

Clear all entries in NIXL storage.

##### `register(tensor: torch.Tensor) -> bool`

Register a tensor with NIXL for optimized operations. Automatically detects if the tensor is on GPU memory and registers accordingly.

**Parameters:**
- `tensor`: Tensor to register with NIXL

**Returns:**
- `bool`: True if registration was successful, False otherwise

##### `deregister(tensor: torch.Tensor) -> bool`

Deregister a tensor from NIXL.

**Parameters:**
- `tensor`: Tensor to deregister from NIXL

**Returns:**
- `bool`: True if deregistration was successful, False otherwise

## File Plugin Selection

The HiCacheNixl connector provides automatic file plugin selection with the following priority order:

1. **GDS_MT**: Multi-threaded GPU Direct Storage (highest priority)
2. **3FS**: Third-party File System
3. **POSIX**: Standard file system (fallback)

```python
# Auto-select best available plugin
storage = HiCacheNixl(file_plugin="auto")

# Use specific plugin
storage = HiCacheNixl(file_plugin="GDS_MT")
storage = HiCacheNixl(file_plugin="3FS")
storage = HiCacheNixl(file_plugin="POSIX")
```

## GPU Memory Detection

The connector automatically detects GPU tensors and registers them appropriately:

```python
# CPU tensor - registered with DRAM
cpu_tensor = torch.randn(10, 10)
storage.register(cpu_tensor)  # Uses DRAM registration

# GPU tensor - registered with VRAM
gpu_tensor = torch.randn(10, 10, device="cuda")
storage.register(gpu_tensor)  # Uses VRAM registration
```

## Performance Optimizations

### Pre-Opened Files

Files are pre-opened at initialization for better performance:

```python
# At initialization, all existing .bin files are pre-opened
storage = HiCacheNixl(file_path="/tmp/hicache_nixl")
# Files are automatically opened and registered with NIXL
```

### Smart Tensor Registration

Only unregistered tensors are registered to avoid redundancy:

```python
# First operation: Register tensor
storage.get("key", tensor)  # Tensor gets registered

# Second operation: Skip registration (tensor already registered)
storage.get("key", tensor)  # No registration overhead
```

### Batch Operations

Batch operations are optimized for maximum efficiency:

```python
# Efficient batch operations
keys = ["key1", "key2", "key3"]
dst_tensors = [torch.zeros(10, 10) for _ in range(3)]

# Single batch transfer operation
retrieved = storage.batch_get(keys, dst_tensors)
```

## Integration with SGLang

The NIXL storage connector is automatically integrated into SGLang's hierarchical cache system when you specify `--hicache-storage-backend nixl` when launching the server.

### Server Launch Example

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --enable-hierarchical-cache \
    --hicache-storage-backend nixl \
    --hicache-size 200 \
    --page-size 128 \
    --hicache-write-policy write_through_selective \
    --hicache-io-backend direct
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--hicache-storage-backend` | Storage backend type | `None` |
| `--hicache-size` | Size of host KV cache memory pool in GB | `0` |
| `--page-size` | Page size for KV cache operations | `1` |
| `--hicache-write-policy` | Write policy for hierarchical cache | `write_through_selective` |
| `--hicache-io-backend` | IO backend for KV cache transfer | `""` |

## Performance Considerations

### Network Latency

The NIXL storage connector relies on network communication with storage endpoints. Consider the following:

- **Network latency**: Higher latency can impact cache performance
- **Bandwidth**: Ensure sufficient bandwidth for large tensor transfers
- **Reliability**: Network failures can affect cache operations

### Storage Endpoint Configuration

For optimal performance:

1. **Co-locate storage**: Place storage endpoints close to compute nodes
2. **High-bandwidth network**: Use high-bandwidth interconnects (InfiniBand, etc.)
3. **Redundant endpoints**: Configure multiple storage endpoints for reliability

### Memory Management

The connector handles tensor serialization/deserialization:

- **Serialization overhead**: Consider the cost of tensor serialization
- **Memory usage**: Monitor memory usage during large transfers
- **Batch operations**: Use batch operations when possible for better efficiency

## Error Handling

The NIXL storage connector includes comprehensive error handling:

### Network Errors

- Connection failures are logged and handled gracefully
- Operations return appropriate error values (None, False) on failure
- Automatic retry mechanisms for transient failures

### Storage Errors

- Storage endpoint failures are detected and reported
- Graceful degradation when storage is unavailable
- Clear error messages for debugging

### Example Error Handling

```python
try:
    storage = HiCacheNixl(file_path="/tmp/hicache_nixl")
    
    # Get operation with dst_tensor
    dst_tensor = torch.zeros(10, 10)
    retrieved = storage.get("key", dst_tensor)
    if retrieved is None:
        print("Key not found or transfer failed")
        
    # Set operation
    success = storage.set("key", tensor)
    if not success:
        print("Storage operation failed")
        
except ImportError:
    print("NIXL not available")
except Exception as e:
    print(f"Storage error: {e}")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure NIXL is properly installed
   ```bash
   pip install nixl
   ```

2. **Missing dst_tensor**: The `get()` method requires a `dst_tensor` parameter
   ```python
   # Correct usage
   dst_tensor = torch.zeros(10, 10)
   retrieved = storage.get("key", dst_tensor)
   
   # Incorrect usage (will return None)
   retrieved = storage.get("key", None)
   ```

3. **Performance Issues**: Monitor network and storage performance
   ```bash
   # Check network bandwidth
   iperf3 -c storage-endpoint-ip
   ```

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('sglang.srt.mem_cache.hicache_storage').setLevel(logging.DEBUG)
```

### Monitoring

Monitor storage operations using the provided logging:

- Storage operation success/failure rates
- Transfer completion times
- Error patterns and frequencies

## Future Enhancements

Planned improvements for the NIXL storage connector:

1. **Compression**: Add tensor compression for reduced network traffic
2. **Caching**: Implement local caching for frequently accessed data
3. **Load balancing**: Support for multiple storage endpoints with load balancing
4. **Metrics**: Enhanced monitoring and metrics collection
5. **Security**: Add authentication and encryption support

## Contributing

To contribute to the NIXL storage connector:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility when possible

## Related Documentation

- [Hierarchical Cache Documentation](./hierarchical_cache.md)
- [NIXL Project Documentation](https://github.com/ai-dynamo/nixl)
- [SGLang Storage Backends](./storage_backends.md) 