"""
Unit tests for DeepSeek utility functions.
This test validates the refactored utility functions in deepseek_common/utils.py
"""

import unittest

import torch

from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    NVFP4_CKPT_FP8_ATTN_QUANT_MODULES,
    _get_llama_4_scaling,
    _is_cublas_ge_129,
    add_forward_absorb_core_attention_backend,
    enable_nextn_moe_bf16_cast_to_fp8,
    yarn_get_mscale,
)
from sglang.test.ci.ci_register import register_cpu_ci

# Register this test for CPU CI with estimated time of 1 second
register_cpu_ci(est_time=1, suite="default")


class TestDeepSeekUtils(unittest.TestCase):
    """Test suite for DeepSeek utility functions."""

    def test_constants_exist(self):
        """Test that constants are accessible and have expected values."""
        # Test NVFP4_CKPT_FP8_ATTN_QUANT_MODULES
        self.assertIsInstance(NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, list)
        self.assertIn("q_b_proj", NVFP4_CKPT_FP8_ATTN_QUANT_MODULES)

        # Test FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        self.assertIsInstance(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS, list)
        expected_backends = [
            "fa3",
            "nsa",
            "flashinfer",
            "cutlass_mla",
            "trtllm_mla",
            "ascend",
        ]
        for backend in expected_backends:
            self.assertIn(
                backend,
                FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
                f"{backend} missing from FORWARD_ABSORB_CORE_ATTENTION_BACKENDS",
            )

        # Test _is_cublas_ge_129
        self.assertIsInstance(_is_cublas_ge_129, bool)

    def test_yarn_get_mscale(self):
        """Test YaRN mscale calculation."""
        # Test scale <= 1 returns 1.0
        self.assertEqual(yarn_get_mscale(1.0), 1.0)
        self.assertEqual(yarn_get_mscale(0.5), 1.0)

        # Test scale > 1 returns value > 1.0
        result = yarn_get_mscale(2.0, 1.0)
        self.assertGreater(result, 1.0)
        self.assertIsInstance(result, float)

        # Test with different mscale values
        result1 = yarn_get_mscale(2.0, 1.0)
        result2 = yarn_get_mscale(2.0, 2.0)
        self.assertGreater(result2, result1, "Higher mscale should give higher result")

    def test_get_llama_4_scaling(self):
        """Test Llama 4 position scaling."""
        # Test with sample positions
        positions = torch.tensor([0, 100, 1000, 10000])
        scaling = _get_llama_4_scaling(8192, 0.5, positions)

        # Check output shape (should have broadcast dimensions)
        self.assertEqual(
            scaling.shape,
            (4, 1, 1),
            f"Expected shape (4, 1, 1), got {scaling.shape}",
        )

        # Check scaling increases with position
        self.assertGreater(
            scaling[3].item(),
            scaling[0].item(),
            "Position 10000 should have higher scaling than 0",
        )

        # Check all values are >= 1.0 (no scaling down)
        self.assertTrue((scaling >= 1.0).all())

    def test_enable_nextn_moe_bf16_cast_to_fp8(self):
        """Test nextn MoE BF16 to FP8 casting logic."""
        # Test with None quant_config (should return False)
        result = enable_nextn_moe_bf16_cast_to_fp8(None)
        self.assertFalse(result, "Should return False with None quant_config")

    def test_add_forward_absorb_core_attention_backend(self):
        """Test adding new attention backend to registry."""
        # Store original list
        original_backends = FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.copy()

        # Add a test backend
        test_backend = "test_backend_xyz123"
        add_forward_absorb_core_attention_backend(test_backend)

        # Verify it was added
        self.assertIn(test_backend, FORWARD_ABSORB_CORE_ATTENTION_BACKENDS)

        # Try adding again (should not duplicate)
        initial_len = len(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS)
        add_forward_absorb_core_attention_backend(test_backend)
        self.assertEqual(len(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS), initial_len)

        # Clean up: remove test backend
        FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.remove(test_backend)

        # Verify cleanup
        self.assertNotIn(test_backend, FORWARD_ABSORB_CORE_ATTENTION_BACKENDS)


if __name__ == "__main__":
    unittest.main()
