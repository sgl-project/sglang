# DeepSeek Refactoring Test Scripts

This directory contains test scripts for validating the DeepSeek utilities refactoring.

## Quick Start

Run the smoke test (works on any machine, including MacBook):

```bash
cd test/manual
python test_deepseek_smoke.py
```

## Test Scripts

### 1. `test_deepseek_smoke.py` - Quick Smoke Test
**Requirements:** Python, PyTorch (CPU/GPU)  
**Duration:** < 5 seconds

Validates:
- All imports work
- Utility functions are callable
- Constants are accessible

```bash
python test_deepseek_smoke.py
```

### 2. `test_deepseek_utils_refactoring.py` - Unit Tests
**Requirements:** Python, PyTorch (CPU/GPU)  
**Duration:** < 10 seconds

Comprehensive unit tests for:
- `yarn_get_mscale()`
- `_get_llama_4_scaling()`
- `enable_nextn_moe_bf16_cast_to_fp8()`
- `add_forward_absorb_core_attention_backend()`
- Constants validation

```bash
python test_deepseek_utils_refactoring.py
```

### 3. `test_deepseek_model_loading.py` - Model Loading Test
**Requirements:** NVIDIA GPU, transformers library  
**Duration:** ~30 seconds

Tests:
- Model class instantiation
- Config loading from HuggingFace
- Utility accessibility from model context

```bash
python test_deepseek_model_loading.py
```

## Full Documentation

See [Testing Guide: DeepSeek Utils Refactoring](../../docs/developer_guide/testing_deepseek_refactoring.md) for:
- Complete testing strategy (6 levels)
- Server integration tests
- Benchmark regression tests
- Troubleshooting guide

## Expected Results

All tests should output:
```
✅ ALL TESTS PASSED!
```

If any test fails, check the detailed documentation for troubleshooting steps.

## Changes Validated

These tests validate the refactoring that moved utilities from:
- `python/sglang/srt/models/deepseek_v2.py`

To:
- `python/sglang/srt/models/deepseek_common/utils.py`

Extracted items:
- 4 utility functions
- 3 constants
- No behavior changes (pure code movement)
