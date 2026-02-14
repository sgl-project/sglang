# HiCache Storage Performance Comparison Test

## Overview

`test_hicache_storage_performance_comparison.py` is a performance comparison testing tool that compares two interface implementations of the HiCacheFile storage backend:

- **Old Interface**: `batch_set` / `batch_get` - Traditional batch read/write interface
- **New Interface**: `batch_set_v1` / `batch_get_v1` - Zero-copy batch read/write interface (optimized version)

This testing tool helps evaluate the performance improvements of the new interface compared to the old interface, especially when handling different batch sizes.

## Features

### 1. Performance Comparison Testing
- Compare write and read performance between old and new interfaces
- Support testing with different batch sizes (small, medium, large)
- Provide multiple runs for statistically more accurate results

### 2. Zero-Copy Optimization Verification
- Verify that the zero-copy mechanism of the new interface works correctly
- Test the behavior of `get_page_buffer_meta` method under MHA and MLA models

### 3. Integration Testing
- Verify that the integration between the new interface and storage backend is correct
- Check that file creation and sizes meet expectations

## Usage

### Basic Usage

```bash
# Run all tests (default: compare both old and new interfaces)
python3 test/registered/hicache/test_hicache_storage_performance_comparison.py

# Test only the old interface
python3 test/registered/hicache/test_hicache_storage_performance_comparison.py old

# Test only the new interface
python3 test/registered/hicache/test_hicache_storage_performance_comparison.py new

# Compare both old and new interfaces (default behavior)
python3 test/registered/hicache/test_hicache_storage_performance_comparison.py both
```

## Test Case Descriptions

### 1. `test_get_page_buffer_meta_mha`
Tests the `get_page_buffer_meta` method under MHA (Multi-Head Attention) model:
- Verifies the number of returned pointers (each page should return K and V pointers)
- Verifies pointer sizes and validity
- Verifies the correct relationship between K and V pointers

### 2. `test_get_page_buffer_meta_mla`
Tests the `get_page_buffer_meta` method under MLA (Multi-Query Latent Attention) model:
- Verifies the number of returned pointers (one pointer per page)
- Verifies pointer sizes and validity

### 3. `test_get_page_buffer_meta_integration`
Integration test that verifies the collaboration between `get_page_buffer_meta` and `batch_set_v1`:
- Verifies that batch write operations succeed
- Verifies that generated file sizes are correct

### 4. `test_performance_comparison_small_batch`
Small batch performance comparison (10 keys):
- Measures write and read times
- Calculates performance improvement multiplier

### 5. `test_performance_comparison_medium_batch`
Medium batch performance comparison (50 keys)

### 6. `test_performance_comparison_large_batch`
Large batch performance comparison (100 keys)

### 7. `test_performance_comparison_multiple_runs`
Multiple runs test (10 keys, 5 runs):
- Provides more accurate performance statistics
- Calculates average performance improvement

## Output Example

```
================================================================================
Performance Comparison: Small Batch (10 keys)
================================================================================

[Old Interface] batch_set: 15.234 ms
[Old Interface] batch_get: 12.456 ms

[New Interface] batch_set_v1: 8.123 ms
[New Interface] batch_get_v1: 6.789 ms

================================================================================
Write Speedup: 1.88x (faster)
Read Speedup:  1.84x (faster)
================================================================================
```

## Technical Details

### MockHostKVCache
The test uses the `MockHostKVCache` class to simulate a real HostKVCache:
- Supports both MHA and MLA model types
- Provides buffer metadata required for zero-copy operations
- Simulates key methods such as `get_data_page`, `set_from_flat_data_page`, etc.

### Performance Measurement Method
- Uses `time.perf_counter()` for high-precision time measurement
- Includes a warm-up phase to eliminate cold start effects
- Measurement time includes all related operations (such as `get_data_page`, `set_from_flat_data_page`, etc.)

### File Format
- Old interface: Uses `.bin` files
- New interface: Uses `.batch.bin` files

## Notes

1. **Temporary Files**: Tests create files in temporary directories, which are automatically cleaned up after tests complete
2. **Memory Allocation**: Tests use pre-allocated memory pools to avoid repeated allocations affecting performance measurements
3. **Model Type**: Default tests use MHA model, can test MLA model by modifying the `is_mla_model` parameter
4. **Page Size**: Default page size is 64 tokens, can be adjusted via the `page_size` parameter

## Requirements

- Python 3.x
- PyTorch
- sglang related modules (`sglang.srt.mem_cache.hicache_storage`)

## Troubleshooting

### Test Failures
- Check if there is sufficient disk space for temporary files
- Confirm that PyTorch and related dependencies are correctly installed
- Review detailed error messages in test output

### Abnormal Performance Results
- Ensure system load is low to avoid other processes affecting test results
- Run tests multiple times to obtain more stable averages
- Check if background processes are consuming I/O resources

## Contributing

If you find issues in the tests or have improvement suggestions, please submit an Issue or Pull Request.
