# NIXL Integration for HiCache

This directory contains the **NIXL (NVIDIA Inference Xfer Library)** integration for **HiCache**, enabling high-performance storage across multiple backends.

NIXL provides a unified API for accessing various storage plugins, including but not limited to:

- **Deepseek's 3FS APIs** for high-throughput file operations
- **GPU Direct Storage (GDS)** for direct data movement between storage and GPU memory, bypassing CPU memory copies
- **Amazon S3-compatible object storage** for key-value access patterns

Additional backend integrations are planned for future releases.

## NIXL Resources

- **Project Repository**: [NIXL on GitHub](https://github.com/ai-dynamo/nixl)
- **Documentation**: [NIXL Documentation](https://github.com/ai-dynamo/nixl/tree/main/docs)

## Overview

The NIXL integration consists of two main files:

- **`hicache_nixl.py`** - Main HiCache storage connector using NIXL
- **`nixl_utils.py`** - Utility classes for backend selection, registration, and file management

## Components

### HiCacheNixl
The main storage connector that provides:
- Single and batch tensor set/get operations
- Automatic backend selection (3FS > POSIX > GDS_MT > GDS > OBJ)
- High-performance file-based (or) object based storage access using NIXL

### NixlUtils
Consolidated utility classes:
- **NixlBackendSelection** - Handles backend selection and creation
- **NixlRegistration** - Manages memory registration for tensors, files and objects
- **NixlFileManager** - Handles file system operations and NIXL tuple creation

## Running Unit Tests

### Prerequisites
- NIXL library installed and available (latest main required for supporting object query)
- PyTorch installed
- Python 3.8+

### Unit tests from Project root
Navigate to the project root directory (`/path/to/sglang`) and run:

#### Run all NIXL tests:
```bash
PYTHONPATH=python python -m sglang.srt.mem_cache.nixl.tests.test_nixl_unified
```

#### Run with verbose output:
```bash
PYTHONPATH=python python -m unittest sglang.srt.mem_cache.nixl.tests.test_nixl_unified.TestNixlUnified -v
```

#### Run a specific test:
```bash
PYTHONPATH=python python -m unittest sglang.srt.mem_cache.nixl.tests.test_nixl_unified.TestNixlUnified.test_single_set_get -v
```

#### Run all tests in the nixl directory:
```bash
PYTHONPATH=python python -m unittest discover sglang.srt.mem_cache.nixl.tests -p "test_*.py" -v
```

### From Tests Directory
Navigate to the tests directory and run:

```bash
cd python/sglang/srt/mem_cache/nixl/tests
PYTHONPATH=../../../../.. python test_nixl_unified.py
```

## Test Coverage

The unified test suite (`test_nixl_unified.py`) covers:

### HiCache Integration Tests (4 tests)
- Single tensor set/get operations
- Batch tensor set/get operations
- Mixed single and batch operations
- Error handling in set/get operations

### File Management Tests (5 tests)
- Basic file operations
- NIXL tuple creation
- Error handling in file operations
- File descriptor cleanup on failure

### Registration Tests (2 tests)
- Tensor registration with memory type detection
- File registration using NIXL tuples

### Backend Selection Tests (1 test)
- NixlBackendSelection class testing
- Priority order verification (3FS > POSIX > GDS_MT > GDS)

## Expected Output

When tests run successfully, you should see:
- NIXL agent initialization messages
- Backend selection messages (e.g., "Backend POSIX was instantiated")
- Test results with "ok" for passed tests
- Summary showing "Ran X tests in Y seconds" and "OK"

## Troubleshooting

### Import Errors
If you encounter `ModuleNotFoundError`, ensure:
- You're running from the correct directory
- `PYTHONPATH` is set correctly
- NIXL library is properly installed

### NIXL Errors
If NIXL operations fail:
- Check that NIXL is properly installed
- Verify that required plugins are available
- Ensure file permissions are correct for test directories

## File Structure

```
python/sglang/srt/mem_cache/nixl/
├── hicache_nixl.py          # Main HiCache storage connector
├── nixl_utils.py            # All NIXL utility classes
├── README.md                # This file
└── tests/
    └── test_nixl_unified.py # All tests in one file
```

## Dependencies

- **NIXL**: NVIDIA I/O eXchange Library (version 1.0 or later)
  - Required plugins: POSIX (minimum), 3FS/GDS (optional for better performance)
  - See [NIXL Installation Guide](https://github.com/ai-dynamo/nixl/blob/main/README.md#installation)
- **PyTorch**: For tensor operations (version 1.8 or later)
- **Python 3.8+**: For type hints and modern features

## Supported Features

### Memory Types
- **FILE**: Standard file-based storage (all backends)
  - Supports all numeric tensor types (int32, int64, float32, float64)
  - Supports multi-dimensional tensors
  - File descriptor management and cleanup
- **OBJ**: Object-based storage (OBJ backend only)
  - Key-based storage and retrieval
  - Direct tensor-to-object mapping
  - No file system overhead

### Tensor Support
- **Data Types**:
  - Integer: int32, int64
  - Floating Point: float32, float64
  - Note: boolean, empty tensors, and special values (inf, nan) are not currently supported
- **Shapes**: Supports all tensor shapes (1D, 2D, 3D, and higher dimensions)
- **Devices**: CPU tensors (GPU tensor support depends on backend)

### Backend Priority

The NIXL backend selection follows this priority order:
1. **3FS** - Highest performance (if available)
    - Best for high-throughput file operations using Deepseek 3FS APIs
2. **POSIX** - Standard file I/O (fallback)
    - Universal compatibility
    - Good for development and testing - Levearges both libaio/liburing
3. **GDS_MT** - Multi-threaded GDS (if available)
    - Optimized for concurrent operations
    - Supports GPU Direct storage with multiple light weight threads
4. **GDS** - GPU Direct Storage (if available)
    - Direct GPU-storage data path
    - Best for filesystems benefiting from batch operations and smaller IOs.
5. **OBJ** - Amazon S3 based Object Storage
    - Key-value based storage
The system automatically selects the best available backend, with POSIX as the default fallback.

## Note

This is v0 of the NIXL connector. Future versions will focus on further performance optimizations such as memory pre-registration (pre-allocating and registering memory buffers to reduce registration overhead during transfers) and block merging (combining related blocks as offsets within the same file to reduce file operations and improve throughput).

These optimizations require changes at a higher layer, as the current HiCache API doesn't expose information like block relationships or hash patterns that would enable these optimizations.
