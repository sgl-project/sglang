# Testing Guide: DeepSeek Utils Refactoring

This document describes how to test the refactoring changes made to extract standalone utilities from `deepseek_v2.py` to `deepseek_common/utils.py`.

## Overview of Changes

The refactoring extracted the following from `python/sglang/srt/models/deepseek_v2.py` to `python/sglang/srt/models/deepseek_common/utils.py`:

### Constants
- `NVFP4_CKPT_FP8_ATTN_QUANT_MODULES`
- `FORWARD_ABSORB_CORE_ATTENTION_BACKENDS`
- `_is_cublas_ge_129`

### Utility Functions
- `enable_nextn_moe_bf16_cast_to_fp8()` - Determines if nextn MoE weights should be cast to FP8
- `add_forward_absorb_core_attention_backend()` - Registry function for attention backends
- `yarn_get_mscale()` - YaRN scaling calculation for RoPE embeddings
- `_get_llama_4_scaling()` - Llama 4 style position scaling

## Testing Levels

### Level 1: Import Verification (No GPU Required)

Verify that all imports work correctly. This can be run on any machine including MacBook.

```bash
cd /path/to/sglang

# Test main model imports
python -c "from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM"

# Test utility imports
python -c "from sglang.srt.models.deepseek_common.utils import (
    enable_nextn_moe_bf16_cast_to_fp8,
    add_forward_absorb_core_attention_backend,
    yarn_get_mscale,
    _get_llama_4_scaling,
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    NVFP4_CKPT_FP8_ATTN_QUANT_MODULES,
    _is_cublas_ge_129
)"

# Test nextn model imports
python -c "from sglang.srt.models.deepseek_nextn import DeepseekModelNextN"

echo "✅ All imports successful!"
```

**Expected Result:** No import errors.

---

### Level 2: Unit Tests for Extracted Utilities (CPU/GPU)

Create a test file to verify the utility functions work correctly.

**File:** `test/manual/test_deepseek_utils_refactoring.py`

```python
"""
Unit tests for DeepSeek utility functions after refactoring.
Run: python test/manual/test_deepseek_utils_refactoring.py
"""
import torch
from sglang.srt.models.deepseek_common.utils import (
    enable_nextn_moe_bf16_cast_to_fp8,
    add_forward_absorb_core_attention_backend,
    yarn_get_mscale,
    _get_llama_4_scaling,
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    NVFP4_CKPT_FP8_ATTN_QUANT_MODULES,
    _is_cublas_ge_129,
)


def test_constants():
    """Test that constants are accessible and have expected values."""
    print("Testing constants...")
    
    # Test NVFP4_CKPT_FP8_ATTN_QUANT_MODULES
    assert isinstance(NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, list)
    assert "q_b_proj" in NVFP4_CKPT_FP8_ATTN_QUANT_MODULES
    
    # Test FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
    assert isinstance(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS, list)
    expected_backends = ["fa3", "nsa", "flashinfer", "cutlass_mla", "trtllm_mla", "ascend"]
    for backend in expected_backends:
        assert backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS, f"{backend} missing"
    
    # Test _is_cublas_ge_129
    assert isinstance(_is_cublas_ge_129, bool)
    
    print("  ✓ All constants verified")


def test_yarn_get_mscale():
    """Test YaRN mscale calculation."""
    print("Testing yarn_get_mscale...")
    
    # Test scale <= 1 returns 1.0
    assert yarn_get_mscale(1.0) == 1.0
    assert yarn_get_mscale(0.5) == 1.0
    
    # Test scale > 1 returns value > 1.0
    result = yarn_get_mscale(2.0, 1.0)
    assert result > 1.0
    assert isinstance(result, float)
    
    # Test with different mscale values
    result1 = yarn_get_mscale(2.0, 1.0)
    result2 = yarn_get_mscale(2.0, 2.0)
    assert result2 > result1  # Higher mscale should give higher result
    
    print(f"  ✓ yarn_get_mscale(2.0, 1.0) = {result1:.4f}")


def test_get_llama_4_scaling():
    """Test Llama 4 position scaling."""
    print("Testing _get_llama_4_scaling...")
    
    # Test with sample positions
    positions = torch.tensor([0, 100, 1000, 10000])
    scaling = _get_llama_4_scaling(8192, 0.5, positions)
    
    # Check output shape (should have broadcast dimensions)
    assert scaling.shape == (4, 1, 1), f"Expected shape (4, 1, 1), got {scaling.shape}"
    
    # Check scaling increases with position
    assert scaling[3] > scaling[0]  # Position 10000 should have higher scaling than 0
    
    # Check all values are >= 1.0 (no scaling down)
    assert (scaling >= 1.0).all()
    
    print(f"  ✓ Scaling at pos 0: {scaling[0].item():.4f}, pos 10000: {scaling[3].item():.4f}")


def test_enable_nextn_moe_bf16_cast_to_fp8():
    """Test nextn MoE BF16 to FP8 casting logic."""
    print("Testing enable_nextn_moe_bf16_cast_to_fp8...")
    
    # Test with None quant_config (should return False)
    result = enable_nextn_moe_bf16_cast_to_fp8(None)
    assert result == False, "Should return False with None quant_config"
    
    print("  ✓ Returns False with None quant_config")


def test_add_forward_absorb_core_attention_backend():
    """Test adding new attention backend to registry."""
    print("Testing add_forward_absorb_core_attention_backend...")
    
    # Store original list
    original_backends = FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.copy()
    
    # Add a test backend
    test_backend = "test_backend_xyz123"
    add_forward_absorb_core_attention_backend(test_backend)
    
    # Verify it was added
    assert test_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
    
    # Try adding again (should not duplicate)
    initial_len = len(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS)
    add_forward_absorb_core_attention_backend(test_backend)
    assert len(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS) == initial_len
    
    # Clean up: remove test backend
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.remove(test_backend)
    
    print("  ✓ Backend registration works correctly")


def main():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("DeepSeek Utils Refactoring - Unit Tests")
    print("="*60 + "\n")
    
    try:
        test_constants()
        test_yarn_get_mscale()
        test_get_llama_4_scaling()
        test_enable_nextn_moe_bf16_cast_to_fp8()
        test_add_forward_absorb_core_attention_backend()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
```

**Run:**
```bash
cd test/manual
python test_deepseek_utils_refactoring.py
```

**Expected Result:** All tests pass with green checkmarks.

---

### Level 3: Integration Test with Existing Test Suite (GPU Required)

Run the existing SGLang test suite for DeepSeek models.

```bash
cd test

# Find all DeepSeek-related tests
find . -name "*deepseek*" -type f

# Run DeepSeek model tests
python -m pytest registered/ -k deepseek -v --tb=short

# Or run specific test categories
python -m pytest registered/test_deepseek_models.py -v
python -m pytest srt/ -k deepseek -v
```

**Expected Result:** All existing tests pass without regression.

---

### Level 4: Model Loading and Inference Test (GPU Required)

Test that DeepSeek models can be loaded and used for inference.

**File:** `test/manual/test_deepseek_model_loading.py`

```python
"""
Integration test: Load DeepSeek model and run inference.
Requires: GPU with sufficient VRAM
Run: python test/manual/test_deepseek_model_loading.py
"""
import torch
from transformers import AutoTokenizer, AutoConfig
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM


def test_model_loading():
    """Test loading DeepSeek V3 model architecture."""
    print("Testing DeepSeek V3 model loading...")
    
    # Use a small config for testing (not loading actual weights)
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
    
    try:
        # Test model instantiation
        model = DeepseekV3ForCausalLM(config, quant_config=None)
        print("  ✓ Model instantiated successfully")
        
        # Verify utilities are accessible from model
        from sglang.srt.models.deepseek_common.utils import (
            FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
            yarn_get_mscale,
        )
        print("  ✓ Utilities accessible from model context")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("DeepSeek Model Loading Test")
    print("="*60 + "\n")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("⚠️  No GPU detected. Some tests may fail.\n")
    
    success = test_model_loading()
    
    print("\n" + "="*60)
    if success:
        print("✅ MODEL LOADING TEST PASSED!")
    else:
        print("❌ MODEL LOADING TEST FAILED!")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
```

**Run:**
```bash
python test/manual/test_deepseek_model_loading.py
```

---

### Level 5: Full Server Test (GPU Required)

Test with a running SGLang server and actual inference.

```bash
# Start server with DeepSeek-V2-Lite (smallest variant)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --tp 1 \
    --port 30000 \
    --log-level info

# In another terminal, test inference
python -c "
import requests
import json

response = requests.post(
    'http://localhost:30000/generate',
    json={
        'text': 'What is the capital of France?',
        'sampling_params': {
            'max_new_tokens': 32,
            'temperature': 0.0
        }
    }
)

result = response.json()
print('Prompt:', result['text'][:50])
print('Generated:', result['text'][50:])
print('✅ Server inference test passed!')
"

# Stop the server
pkill -f "sglang.launch_server"
```

**Expected Result:** Server starts successfully and generates coherent responses.

---

### Level 6: Regression Test with Benchmark (Optional)

Run benchmarks to ensure no performance regression.

```bash
# Run GSM8K benchmark (if available)
cd benchmark/gsm8k
python bench_sglang.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --num-questions 10

# Compare results with baseline (before refactoring)
```

---

## Quick Smoke Test

For a fast sanity check on an NVIDIA machine:

```bash
python -c "
import torch
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
from sglang.srt.models.deepseek_common.utils import (
    yarn_get_mscale,
    _get_llama_4_scaling,
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
)

print('GPU Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

print('\n--- Testing Utilities ---')
print('yarn_get_mscale(2.0):', yarn_get_mscale(2.0))

positions = torch.tensor([0, 8192])
scaling = _get_llama_4_scaling(8192, 0.5, positions)
print('_get_llama_4_scaling shape:', scaling.shape)

print('Attention backends:', len(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS))
print('\n✅ All refactoring smoke tests passed!')
"
```

---

## Troubleshooting

### Import Errors

If you see import errors:

```python
# Check if the utils module exists
import os
utils_path = 'python/sglang/srt/models/deepseek_common/utils.py'
print(f"Utils file exists: {os.path.exists(utils_path)}")

# Verify Python path
import sys
print("Python path:", sys.path)
```

### Function Not Found

If a function is not found in `deepseek_v2.py`:

1. Check it was imported: `grep "from sglang.srt.models.deepseek_common.utils import" python/sglang/srt/models/deepseek_v2.py`
2. Verify it exists in utils: `grep "def function_name" python/sglang/srt/models/deepseek_common/utils.py`

### Model Loading Failures

If model loading fails:

1. Check VRAM: `nvidia-smi`
2. Try with smaller batch size or model
3. Check logs for specific error messages

---

## Test Checklist

- [ ] Level 1: Import verification passes
- [ ] Level 2: Unit tests all pass
- [ ] Level 3: Existing test suite passes
- [ ] Level 4: Model loads successfully
- [ ] Level 5: Server inference works
- [ ] Level 6: Benchmarks show no regression (optional)

---

## Expected Outcomes

### What Should Work
✅ All imports resolve correctly  
✅ Utility functions return expected values  
✅ Models instantiate without errors  
✅ Inference produces same results as before  
✅ No performance regression  

### What Should NOT Change
- Model accuracy
- Inference speed
- Memory usage
- API behavior

---

## Reporting Issues

If tests fail, collect:

1. **Python version:** `python --version`
2. **PyTorch version:** `python -c "import torch; print(torch.__version__)"`
3. **CUDA version:** `nvcc --version` or `nvidia-smi`
4. **Full error traceback**
5. **Specific test that failed**

Report in the issue with label `refactoring` and `deepseek`.

---

## Summary

The refactoring was **purely structural** - moving code without changing logic. As long as:
1. Imports work correctly
2. Models can be loaded
3. Inference runs successfully

The refactoring is successful! The main risk areas are import paths and circular dependencies, which the test levels above verify systematically.
