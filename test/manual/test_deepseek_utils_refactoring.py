"""
Unit tests for DeepSeek utility functions after refactoring.
Run: python test/manual/test_deepseek_utils_refactoring.py
"""
import torch
import traceback
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
    
    print("All constants verified")


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
    
    print(f"yarn_get_mscale(2.0, 1.0) = {result1:.4f}")


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
    
    print(f"Scaling at pos 0: {scaling[0].item():.4f}, pos 10000: {scaling[3].item():.4f}")


def test_enable_nextn_moe_bf16_cast_to_fp8():
    """Test nextn MoE BF16 to FP8 casting logic."""
    print("Testing enable_nextn_moe_bf16_cast_to_fp8...")
    
    # Test with None quant_config (should return False)
    result = enable_nextn_moe_bf16_cast_to_fp8(None)
    assert result == False, "Should return False with None quant_config"
    
    print("Returns False with None quant_config")


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
    
    print("Backend registration works correctly")


def main():
    """Run all unit tests."""
    print("DeepSeek Utils Refactoring - Unit Tests")

    try:
        test_constants()
        test_yarn_get_mscale()
        test_get_llama_4_scaling()
        test_enable_nextn_moe_bf16_cast_to_fp8()
        test_add_forward_absorb_core_attention_backend()
        
        print("ALL TESTS PASSED!")
        return 0
        
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}\n")

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
