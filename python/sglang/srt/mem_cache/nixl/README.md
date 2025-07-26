# NIXL Integration for HiCache

This directory contains the NIXL (NVIDIA I/O eXchange Library) integration for HiCache, providing high-performance storage using NIXL file plugins.

## Overview

The NIXL integration consists of two main files:

- **`hicache_nixl.py`** - Main HiCache storage connector using NIXL
- **`nixl_utils.py`** - Utility classes for backend selection, registration, and file management

## Components

### HiCacheNixl
The main storage connector that provides:
- Single and batch tensor set/get operations
- Automatic backend selection (3FS > POSIX > GDS_MT > GDS)
- High-performance file-based storage using NIXL

### NixlUtils
Consolidated utility classes:
- **NixlBackendSelection** - Handles backend selection and creation
- **NixlRegistration** - Manages memory registration for tensors and files
- **NixlFileManager** - Handles file system operations and NIXL tuple creation

## Running Unit Tests

### Prerequisites
- NIXL library installed and available
- PyTorch installed
- Python 3.8+

### From Project Root
Navigate to the project root directory (`/path/to/sglang-venkat`) and run:

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
- NIXL tuple creation with proper offsets and lengths
- Conversion of 4-element tuples to 3-element tuples for transfer
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

- **NIXL**: NVIDIA I/O eXchange Library
- **PyTorch**: For tensor operations
- **Python 3.8+**: For type hints and modern features

## Backend Priority

The NIXL backend selection follows this priority order:
1. **3FS** - Highest performance (if available)
2. **POSIX** - Standard file I/O (fallback)
3. **GDS_MT** - Multi-threaded GDS (if available)
4. **GDS** - GPU Direct Storage (if available)

The system automatically selects the best available backend, with POSIX as the default fallback. 