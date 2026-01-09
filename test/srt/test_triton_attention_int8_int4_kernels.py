"""
Unit tests for Triton attention kernels with int8/int4 quantized KV cache.
Tests decode attention with quantized KV cache against non-quantized baseline.
"""

import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_normal,
    decode_attention_fwd_normal_quant,
)
from sglang.srt.mem_cache.kv_quant_kernels import (
    quantized_set_kv_int4_triton,
    quantized_set_kv_int8_triton,
)
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


class TestTritonAttentionInt8Int4Kernels(CustomTestCase):
    """Test decode attention with int8/int4 quantized KV cache."""

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def _naive_dequantize_int8(self, q, scale, zero):
        """Naive int8 dequantization for reference."""
        return (q.to(torch.float32) - zero) * scale

    def _naive_dequantize_int4(self, packed, scale, zero):
        """Naive int4 dequantization for reference."""
        q1 = (packed & 0x0F).to(torch.float32)
        q2 = ((packed >> 4) & 0x0F).to(torch.float32)
        deq1 = (q1 - zero) * scale
        deq2 = (q2 - zero) * scale
        return torch.cat([deq1, deq2], dim=-1)

    def _test_decode_attention_quant_once(
        self, B, H_Q, H_KV, D, S, kv_dtype, cache_size=None
    ):
        """
        Test decode attention with quantized KV cache against non-quantized baseline.

        Args:
            B: Batch size
            H_Q: Number of query heads
            H_KV: Number of KV heads
            D: Head dimension
            S: Sequence length (number of tokens in KV cache)
            kv_dtype: "int4" or "int8"
            cache_size: Size of cache buffer (defaults to total_tokens)
        """
        device = get_device()
        dtype = torch.bfloat16
        total_tokens = B * S
        if cache_size is None:
            cache_size = total_tokens
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device=device)

        # Create query (one per batch item)
        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

        # Create non-quantized KV cache (baseline)
        k_buffer_fp = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer_fp = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)

        # Setup KV indices
        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        kv_indptr[1 : B + 1] = torch.cumsum(
            torch.full((B,), S, dtype=torch.int32, device=device), dim=0
        )
        kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)

        # Run non-quantized baseline
        o_baseline = torch.zeros(B, H_Q, D, dtype=dtype, device=device)
        attn_logits_baseline = torch.empty(
            (B, H_Q, max_kv_splits, D), dtype=torch.float32, device=device
        )
        attn_lse_baseline = torch.empty(
            (B, H_Q, max_kv_splits), dtype=torch.float32, device=device
        )

        decode_attention_fwd_normal(
            q,
            k_buffer_fp,
            v_buffer_fp,
            o_baseline,
            kv_indptr,
            kv_indices,
            attn_logits_baseline,
            attn_lse_baseline,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
        )

        # Create quantized KV cache
        if kv_dtype == "int4":
            head_dim_stored = D // 2
            assert D % 2 == 0, "head_dim must be even for int4"
        else:
            head_dim_stored = D

        # Create cache buffers
        k_cache_buffer = torch.zeros(
            cache_size, H_KV, head_dim_stored, device=device, dtype=torch.uint8
        )
        v_cache_buffer = torch.zeros(
            cache_size, H_KV, head_dim_stored, device=device, dtype=torch.uint8
        )
        k_scales_zeros = torch.zeros(
            cache_size, H_KV, 2, device=device, dtype=torch.float32
        )
        v_scales_zeros = torch.zeros(
            cache_size, H_KV, 2, device=device, dtype=torch.float32
        )

        # Quantize KV cache
        cache_loc = kv_indices.clone()
        if kv_dtype == "int4":
            quantized_set_kv_int4_triton(
                k_buffer_fp,
                v_buffer_fp,
                cache_loc,
                k_cache_buffer,
                v_cache_buffer,
                k_scales_zeros,
                v_scales_zeros,
            )
        else:
            quantized_set_kv_int8_triton(
                k_buffer_fp,
                v_buffer_fp,
                cache_loc,
                k_cache_buffer,
                v_cache_buffer,
                k_scales_zeros,
                v_scales_zeros,
            )

        # Run quantized decode attention
        o_quant = torch.zeros(B, H_Q, D, dtype=dtype, device=device)
        attn_logits_quant = torch.empty(
            (B, H_Q, max_kv_splits, D), dtype=torch.float32, device=device
        )
        attn_lse_quant = torch.empty(
            (B, H_Q, max_kv_splits), dtype=torch.float32, device=device
        )

        decode_attention_fwd_normal_quant(
            q,
            k_cache_buffer,
            v_cache_buffer,
            k_scales_zeros,
            v_scales_zeros,
            o_quant,
            kv_indptr,
            kv_indices,
            attn_logits_quant,
            attn_lse_quant,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            kv_dtype=kv_dtype,
        )

        # Compare outputs
        diff = o_quant - o_baseline
        norm_diff = torch.norm(diff).item()
        norm_baseline = torch.norm(o_baseline).item()
        relative_error = norm_diff / (norm_baseline + 1e-8)

        # Calculate max absolute difference
        max_diff = torch.abs(diff).max().item()

        # Print error metrics for debugging
        print(
            f"\nConfig: B={B}, H_Q={H_Q}, H_KV={H_KV}, D={D}, S={S}, dtype={kv_dtype}"
        )
        print(f"Relative error: {relative_error:.6f}")
        print(f"Max absolute diff: {max_diff:.6f}")
        print(f"Norm(baseline): {norm_baseline:.6f}, Norm(diff): {norm_diff:.6f}")

        # For int4, quantization error is larger, so we use more lenient tolerance
        if kv_dtype == "int4":
            # Allow up to 20% relative error for int4
            # Int4 quantization has larger inherent errors due to only 16 quantization levels
            max_rel_error_threshold = 0.2

            self.assertTrue(
                relative_error < max_rel_error_threshold,
                f"Relative error {relative_error:.6f} exceeds {max_rel_error_threshold} for int4",
            )
            self.assertTrue(
                torch.allclose(o_quant, o_baseline, atol=0.4, rtol=0.2),
                f"Outputs not close enough. Max diff: {max_diff:.6f}",
            )
        else:
            # For int8, quantization error is smaller
            self.assertTrue(
                relative_error < 0.02,
                f"Relative error {relative_error:.6f} exceeds 0.02 for int8",
            )
            self.assertTrue(
                torch.allclose(o_quant, o_baseline, atol=0.05, rtol=0.05),
                f"Outputs not close enough. Max diff: {max_diff:.6f}",
            )

    def test_decode_attention_int8_basic(self):
        """Test basic int8 decode attention."""
        self._test_decode_attention_quant_once(
            B=2, H_Q=4, H_KV=4, D=64, S=10, kv_dtype="int8"
        )

    def test_decode_attention_int4_basic(self):
        """Test basic int4 decode attention."""
        self._test_decode_attention_quant_once(
            B=2, H_Q=4, H_KV=4, D=64, S=10, kv_dtype="int4"
        )

    def test_decode_attention_int8_grouped(self):
        """Test int8 decode attention with grouped attention (GQA)."""
        self._test_decode_attention_quant_once(
            B=2, H_Q=16, H_KV=4, D=128, S=20, kv_dtype="int8"
        )

    def test_decode_attention_int4_grouped(self):
        """Test int4 decode attention with grouped attention (GQA)."""
        self._test_decode_attention_quant_once(
            B=2, H_Q=16, H_KV=4, D=128, S=20, kv_dtype="int4"
        )

    def test_decode_attention_int8_different_head_dims(self):
        """Test int8 decode attention with different head dimensions."""
        for head_dim in [64, 128, 256]:
            self._test_decode_attention_quant_once(
                B=2, H_Q=8, H_KV=8, D=head_dim, S=10, kv_dtype="int8"
            )

    def test_decode_attention_int4_different_head_dims(self):
        """Test int4 decode attention with different head dimensions."""
        for head_dim in [64, 128, 256]:
            self._test_decode_attention_quant_once(
                B=2, H_Q=8, H_KV=8, D=head_dim, S=10, kv_dtype="int4"
            )

    def test_decode_attention_int8_different_seq_lens(self):
        """Test int8 decode attention with different sequence lengths."""
        for seq_len in [5, 20, 50, 100]:
            self._test_decode_attention_quant_once(
                B=2, H_Q=4, H_KV=4, D=64, S=seq_len, kv_dtype="int8"
            )

    def test_decode_attention_int4_different_seq_lens(self):
        """Test int4 decode attention with different sequence lengths."""
        for seq_len in [5, 20, 50, 100]:
            self._test_decode_attention_quant_once(
                B=2, H_Q=4, H_KV=4, D=64, S=seq_len, kv_dtype="int4"
            )

    def test_decode_attention_int8_large_batch(self):
        """Test int8 decode attention with large batch."""
        self._test_decode_attention_quant_once(
            B=8, H_Q=16, H_KV=16, D=128, S=50, kv_dtype="int8"
        )

    def test_decode_attention_int4_large_batch(self):
        """Test int4 decode attention with large batch."""
        self._test_decode_attention_quant_once(
            B=8, H_Q=16, H_KV=16, D=128, S=50, kv_dtype="int4"
        )

    def test_decode_attention_int8_non_standard_head_dim(self):
        """Test int8 decode attention with non-standard head dimension."""
        self._test_decode_attention_quant_once(
            B=2, H_Q=4, H_KV=4, D=80, S=10, kv_dtype="int8"
        )

    def test_decode_attention_int4_non_standard_head_dim(self):
        """Test int4 decode attention with non-standard head dimension."""
        self._test_decode_attention_quant_once(
            B=2, H_Q=4, H_KV=4, D=80, S=10, kv_dtype="int4"
        )


if __name__ == "__main__":
    unittest.main()
