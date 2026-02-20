"""
Unit tests for TRTLLM int8/int4 KV cache fusion kernel.
1. flatten KV cache
2. int4/int8 setkvcache

"""

import unittest

import torch

from sglang.srt.layers.attention.kernels.flatten_kv_cache import flatten_kv_cache_sglang
from sglang.srt.mem_cache.kv_quant_kernels import (
    quantized_set_kv_int4_triton,
    quantized_set_kv_int8_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=90, suite="stage-b-test-small-1-gpu-amd")


class TestTRTLLMInt8Int4KVKernel(CustomTestCase):
    """Test fused int8/int4 KV cache write kernel correctness."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def _naive_quantize_int8(self, x, scale, zero):
        """Naive int8 quantization for reference."""
        q = torch.clamp((x / scale + zero + 0.5).to(torch.uint8), 0, 255)
        return q

    def _naive_dequantize_int8(self, q, scale, zero):
        """Naive int8 dequantization for reference."""
        return (q.to(torch.float32) - zero) * scale

    def _naive_quantize_int4(self, x, scale, zero):
        """Naive int4 quantization for reference (packed)."""
        q1 = torch.clamp(
            (x[:, : x.shape[1] // 2] / scale + zero + 0.5).to(torch.uint8), 0, 15
        )
        q2 = torch.clamp(
            (x[:, x.shape[1] // 2 :] / scale + zero + 0.5).to(torch.uint8), 0, 15
        )
        packed = q1 | (q2 << 4)
        return packed

    def _naive_dequantize_int4(self, packed, scale, zero):
        """Naive int4 dequantization for reference."""
        # packed shape: [..., head_dim//2]
        # Extract lower and upper nibbles
        q1 = (packed & 0x0F).to(torch.float32)  # Lower 4 bits
        q2 = ((packed >> 4) & 0x0F).to(torch.float32)  # Upper 4 bits
        # Dequantize
        deq1 = (q1 - zero) * scale
        deq2 = (q2 - zero) * scale
        # Concatenate along the last dimension
        return torch.cat([deq1, deq2], dim=-1)

    def _test_set_kv_int8_correctness(
        self,
        num_tokens,
        num_heads,
        head_dim,
        cache_size,
    ):
        """Test int8 set KV cache correctness."""
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Create input tensors
        cache_k = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        cache_v = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )

        # Create cache buffers
        k_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim, device=device, dtype=torch.uint8
        )
        v_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim, device=device, dtype=torch.uint8
        )
        k_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )
        v_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )

        # Create cache locations
        cache_loc = torch.randperm(cache_size, device=device, dtype=torch.int32)[
            :num_tokens
        ]

        # Run Triton kernel
        quantized_set_kv_int8_triton(
            cache_k.clone(),
            cache_v.clone(),
            cache_loc,
            k_cache_buffer,
            v_cache_buffer,
            k_scales_zeros,
            v_scales_zeros,
        )

        # Verify correctness by dequantizing and comparing
        for i, loc in enumerate(cache_loc):
            for h in range(num_heads):
                scale_k = k_scales_zeros[loc, h, 0].item()
                zero_k = k_scales_zeros[loc, h, 1].item()
                scale_v = v_scales_zeros[loc, h, 0].item()
                zero_v = v_scales_zeros[loc, h, 1].item()

                # Dequantize
                k_deq = self._naive_dequantize_int8(
                    k_cache_buffer[loc, h : h + 1, :], scale_k, zero_k
                )
                v_deq = self._naive_dequantize_int8(
                    v_cache_buffer[loc, h : h + 1, :], scale_v, zero_v
                )

                # Compare with original (allow some quantization error)
                k_orig = cache_k[i, h, :].to(torch.float32)
                v_orig = cache_v[i, h, :].to(torch.float32)

                torch.testing.assert_close(
                    k_deq.squeeze(),
                    k_orig,
                    atol=0.1,
                    rtol=0.1,
                    msg=f"K cache mismatch at loc {loc}, head {h}",
                )
                torch.testing.assert_close(
                    v_deq.squeeze(),
                    v_orig,
                    atol=0.1,
                    rtol=0.1,
                    msg=f"V cache mismatch at loc {loc}, head {h}",
                )

    def _test_set_kv_int4_correctness(
        self,
        num_tokens,
        num_heads,
        head_dim,
        cache_size,
    ):
        """Test int4 set KV cache correctness."""
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # head_dim must be even for int4
        assert head_dim % 2 == 0, "head_dim must be even for int4"

        # Create input tensors
        cache_k = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        cache_v = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )

        # Create cache buffers (packed, half dimension)
        k_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim // 2, device=device, dtype=torch.uint8
        )
        v_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim // 2, device=device, dtype=torch.uint8
        )
        k_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )
        v_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )

        # Create cache locations
        cache_loc = torch.randperm(cache_size, device=device, dtype=torch.int32)[
            :num_tokens
        ]

        # Run Triton kernel
        quantized_set_kv_int4_triton(
            cache_k.clone(),
            cache_v.clone(),
            cache_loc,
            k_cache_buffer,
            v_cache_buffer,
            k_scales_zeros,
            v_scales_zeros,
        )

        # Verify correctness by dequantizing and comparing
        for i, loc in enumerate(cache_loc):
            for h in range(num_heads):
                scale_k = k_scales_zeros[loc, h, 0].item()
                zero_k = k_scales_zeros[loc, h, 1].item()
                scale_v = v_scales_zeros[loc, h, 0].item()
                zero_v = v_scales_zeros[loc, h, 1].item()

                # Dequantize
                k_packed = k_cache_buffer[loc, h : h + 1, :]
                v_packed = v_cache_buffer[loc, h : h + 1, :]
                k_deq = self._naive_dequantize_int4(k_packed, scale_k, zero_k)
                v_deq = self._naive_dequantize_int4(v_packed, scale_v, zero_v)

                # Compare with original (allow some quantization error)
                k_orig = cache_k[i, h, :].to(torch.float32)
                v_orig = cache_v[i, h, :].to(torch.float32)

                k_deq_squeezed = k_deq.squeeze()
                v_deq_squeezed = v_deq.squeeze()

                # Calculate relative error for better diagnostics
                k_diff = (k_deq_squeezed - k_orig).abs()
                k_rel_error = (k_diff / (k_orig.abs() + 1e-8)).max().item()
                k_max_diff = k_diff.max().item()

                v_diff = (v_deq_squeezed - v_orig).abs()
                v_rel_error = (v_diff / (v_orig.abs() + 1e-8)).max().item()
                v_max_diff = v_diff.max().item()

                # For int4, quantization error can be significant, use more lenient tolerance
                # Allow up to 30% relative error or 0.3 absolute error
                self.assertTrue(
                    k_max_diff < 0.3 or k_rel_error < 0.3,
                    f"K cache mismatch at loc {loc}, head {h}: max_diff={k_max_diff:.6f}, rel_error={k_rel_error:.6f}",
                )
                self.assertTrue(
                    v_max_diff < 0.3 or v_rel_error < 0.3,
                    f"V cache mismatch at loc {loc}, head {h}: max_diff={v_max_diff:.6f}, rel_error={v_rel_error:.6f}",
                )

    def _test_flatten_kv_cache_correctness(
        self,
        batch_size,
        num_heads,
        head_dim,
        page_size,
        quant_policy,
    ):
        """Test flatten KV cache correctness."""
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Create sequence lengths
        cache_seqlens = torch.randint(
            1, page_size * 4, (batch_size,), device=device, dtype=torch.int32
        )
        max_seq_len = cache_seqlens.max().item()
        total_tokens = cache_seqlens.sum().item()

        # Create cumulative sequence lengths
        cu_seqlens_k = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens_k[1:] = torch.cumsum(cache_seqlens, dim=0)

        # Create page table
        max_pages = (max_seq_len + page_size - 1) // page_size
        page_table = torch.zeros(
            batch_size, max_pages, device=device, dtype=torch.int32
        )
        total_slots = batch_size * max_pages * page_size

        # Fill page table with sequential page indices
        for b in range(batch_size):
            num_pages = (cache_seqlens[b].item() + page_size - 1) // page_size
            page_table[b, :num_pages] = torch.arange(
                b * max_pages,
                b * max_pages + num_pages,
                device=device,
                dtype=torch.int32,
            )

        # Create quantized cache
        if quant_policy == 4:
            head_dim_stored = head_dim // 2
        else:
            head_dim_stored = head_dim

        # Generate random uint8 values (can't use randn with uint8)
        k_cache = torch.randint(
            0,
            256,
            (total_slots, num_heads, head_dim_stored),
            device=device,
            dtype=torch.uint8,
        )
        v_cache = torch.randint(
            0,
            256,
            (total_slots, num_heads, head_dim_stored),
            device=device,
            dtype=torch.uint8,
        )
        k_scales_zeros = torch.randn(
            total_slots, num_heads, 2, device=device, dtype=torch.float32
        )
        v_scales_zeros = torch.randn(
            total_slots, num_heads, 2, device=device, dtype=torch.float32
        )

        # Ensure scales are positive
        k_scales_zeros[:, :, 0] = torch.abs(k_scales_zeros[:, :, 0]) + 0.01
        v_scales_zeros[:, :, 0] = torch.abs(v_scales_zeros[:, :, 0]) + 0.01

        # Run flatten kernel
        k_flattened, v_flattened = flatten_kv_cache_sglang(
            k_cache,
            v_cache,
            k_scales_zeros,
            v_scales_zeros,
            page_table,
            cache_seqlens,
            cu_seqlens_k,
            page_size,
            num_heads,
            head_dim,
            head_dim,
            quant_policy,
            dtype,
            max_seq_len,
            total_tokens,
        )

        # Verify output shapes (flatten_kv_cache_sglang returns [total_tokens, num_heads, head_dim])
        self.assertEqual(k_flattened.shape, (total_tokens, num_heads, head_dim))
        self.assertEqual(v_flattened.shape, (total_tokens, num_heads, head_dim))

        # Verify that output is dequantized (not uint8)
        self.assertEqual(k_flattened.dtype, dtype)
        self.assertEqual(v_flattened.dtype, dtype)

        # Verify that output contains reasonable values (not all zeros)
        self.assertGreater(k_flattened.abs().max().item(), 0.0)
        self.assertGreater(v_flattened.abs().max().item(), 0.0)

    def _test_set_and_flatten_kv_cache(
        self,
        batch_size,
        num_heads,
        head_dim,
        page_size,
        kv_dtype,
    ):
        """
        Test: set_kv -> flatten_kv_cache -> compare with original.

        This test verifies the entire pipeline:
        1. Create random bf16 KV cache
        2. Quantize and store using set_kv
        3. Flatten using flatten_kv_cache
        4. Compare flattened output with original
        """
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Create sequence lengths for each batch item
        cache_seqlens = torch.randint(
            1, page_size * 4, (batch_size,), device=device, dtype=torch.int32
        )
        max_seq_len = cache_seqlens.max().item()
        total_tokens = cache_seqlens.sum().item()

        # Create original bf16 KV cache [total_tokens, num_heads, head_dim]
        k_original = torch.randn(
            total_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        v_original = torch.randn(
            total_tokens, num_heads, head_dim, device=device, dtype=dtype
        )

        # Create cumulative sequence lengths
        cu_seqlens_k = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens_k[1:] = torch.cumsum(cache_seqlens, dim=0)

        # Create page table and cache locations
        max_pages = (max_seq_len + page_size - 1) // page_size
        page_table = torch.zeros(
            batch_size, max_pages, device=device, dtype=torch.int32
        )
        total_slots = batch_size * max_pages * page_size

        # Build page table and cache location mapping
        # Map each token to a slot index consistently with how flatten_kv_cache will read
        cache_loc_list = []

        for b in range(batch_size):
            num_pages = (cache_seqlens[b].item() + page_size - 1) // page_size
            # Assign page indices (each batch gets a contiguous range of pages)
            page_table[b, :num_pages] = torch.arange(
                b * max_pages,
                b * max_pages + num_pages,
                device=device,
                dtype=torch.int32,
            )

            # Map tokens to slots for this batch
            # This must match how flatten_kv_cache computes slot indices
            seq_start = cu_seqlens_k[b].item()
            seq_end = cu_seqlens_k[b + 1].item()
            for token_idx in range(seq_start, seq_end):
                token_in_seq = token_idx - seq_start
                page_idx = token_in_seq // page_size
                offset_in_page = token_in_seq % page_size
                # Get the page ID from page table
                page_id = page_table[b, page_idx].item()
                # Compute slot index: page_id * page_size + offset_in_page
                # This matches the formula in flatten_kv_cache: slot = page_index * PAGE_SIZE + offset
                slot_id = page_id * page_size + offset_in_page
                cache_loc_list.append(slot_id)

        cache_loc = torch.tensor(cache_loc_list, device=device, dtype=torch.int32)

        # Create quantized cache buffers
        if kv_dtype == "int4":
            head_dim_stored = head_dim // 2
            assert head_dim % 2 == 0, "head_dim must be even for int4"
        else:
            head_dim_stored = head_dim

        k_cache_buffer = torch.zeros(
            total_slots, num_heads, head_dim_stored, device=device, dtype=torch.uint8
        )
        v_cache_buffer = torch.zeros(
            total_slots, num_heads, head_dim_stored, device=device, dtype=torch.uint8
        )
        k_scales_zeros = torch.zeros(
            total_slots, num_heads, 2, device=device, dtype=torch.float32
        )
        v_scales_zeros = torch.zeros(
            total_slots, num_heads, 2, device=device, dtype=torch.float32
        )

        # Step 1: Quantize and store using set_kv
        if kv_dtype == "int4":
            quantized_set_kv_int4_triton(
                k_original,
                v_original,
                cache_loc,
                k_cache_buffer,
                v_cache_buffer,
                k_scales_zeros,
                v_scales_zeros,
            )
        else:
            quantized_set_kv_int8_triton(
                k_original,
                v_original,
                cache_loc,
                k_cache_buffer,
                v_cache_buffer,
                k_scales_zeros,
                v_scales_zeros,
            )

        # Step 2: Flatten using flatten_kv_cache
        k_flattened, v_flattened = flatten_kv_cache_sglang(
            k_cache_buffer,
            v_cache_buffer,
            k_scales_zeros,
            v_scales_zeros,
            page_table,
            cache_seqlens,
            cu_seqlens_k,
            page_size,
            num_heads,
            head_dim,
            head_dim,
            4 if kv_dtype == "int4" else 8,
            dtype,
            max_seq_len,
            total_tokens,
        )

        # Step 3: Compare flattened output with original
        # flattened output is [total_tokens, num_heads, head_dim]
        # original is [total_tokens, num_heads, head_dim]
        # They should match (allowing for quantization error)

        k_diff = (k_flattened - k_original).abs()
        v_diff = (v_flattened - v_original).abs()

        k_norm_diff = torch.norm(k_diff).item()
        k_norm_orig = torch.norm(k_original).item()
        k_rel_error = k_norm_diff / (k_norm_orig + 1e-8)

        v_norm_diff = torch.norm(v_diff).item()
        v_norm_orig = torch.norm(v_original).item()
        v_rel_error = v_norm_diff / (v_norm_orig + 1e-8)

        k_max_diff = k_diff.max().item()
        v_max_diff = v_diff.max().item()

        # Print error metrics
        print(f"\n=== Test: set_kv + flatten_kv_cache ({kv_dtype}) ===")
        print(
            f"Config: batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}, page_size={page_size}"
        )
        print(f"Total tokens: {total_tokens}, Max seq len: {max_seq_len}")
        print(
            f"K cache - Relative error: {k_rel_error:.6f}, Max diff: {k_max_diff:.6f}"
        )
        print(
            f"V cache - Relative error: {v_rel_error:.6f}, Max diff: {v_max_diff:.6f}"
        )
        print(
            f"K cache - Norm(original): {k_norm_orig:.6f}, Norm(diff): {k_norm_diff:.6f}"
        )
        print(
            f"V cache - Norm(original): {v_norm_orig:.6f}, Norm(diff): {v_norm_diff:.6f}"
        )

        # Check if errors are too large
        if kv_dtype == "int4":
            # For int4, allow up to 10% relative error or 0.3 absolute error
            # Int4 quantization has larger inherent errors due to only 16 quantization levels
            max_rel_error_threshold = 0.10
            max_abs_error_threshold = 0.3
        else:
            # For int8, allow up to 2% relative error or 0.1 absolute error
            max_rel_error_threshold = 0.02
            max_abs_error_threshold = 0.1

        if (
            k_rel_error > max_rel_error_threshold
            or k_max_diff > max_abs_error_threshold
        ):
            print(f"WARNING: K cache error exceeds threshold!")
            print(f"  Relative error: {k_rel_error:.6f} > {max_rel_error_threshold}")
            print(f"  Max absolute diff: {k_max_diff:.6f} > {max_abs_error_threshold}")
            # Find worst case
            worst_k_idx = k_diff.argmax()
            worst_k_pos = torch.unravel_index(worst_k_idx, k_diff.shape)
            # torch.unravel_index returns a tuple of tensors, convert to tuple of ints
            worst_k_pos_tuple = tuple(p.item() for p in worst_k_pos)
            print(
                f"  Worst case at position {worst_k_pos_tuple}: orig={k_original[worst_k_pos_tuple].item():.6f}, "
                f"flattened={k_flattened[worst_k_pos_tuple].item():.6f}, diff={k_diff[worst_k_pos_tuple].item():.6f}"
            )

        if (
            v_rel_error > max_rel_error_threshold
            or v_max_diff > max_abs_error_threshold
        ):
            print(f"WARNING: V cache error exceeds threshold!")
            print(f"  Relative error: {v_rel_error:.6f} > {max_rel_error_threshold}")
            print(f"  Max absolute diff: {v_max_diff:.6f} > {max_abs_error_threshold}")
            # Find worst case
            worst_v_idx = v_diff.argmax()
            worst_v_pos = torch.unravel_index(worst_v_idx, v_diff.shape)
            # torch.unravel_index returns a tuple of tensors, convert to tuple of ints
            worst_v_pos_tuple = tuple(p.item() for p in worst_v_pos)
            print(
                f"  Worst case at position {worst_v_pos_tuple}: orig={v_original[worst_v_pos_tuple].item():.6f}, "
                f"flattened={v_flattened[worst_v_pos_tuple].item():.6f}, diff={v_diff[worst_v_pos_tuple].item():.6f}"
            )

        # Assertions - use OR logic: pass if either relative error OR absolute error is within threshold
        # This is more lenient for int4 which can have larger quantization errors
        k_error_ok = (
            k_rel_error < max_rel_error_threshold
            or k_max_diff < max_abs_error_threshold
        )
        v_error_ok = (
            v_rel_error < max_rel_error_threshold
            or v_max_diff < max_abs_error_threshold
        )

        self.assertTrue(
            k_error_ok,
            f"K cache error too large: rel_error={k_rel_error:.6f} (threshold={max_rel_error_threshold}), "
            f"max_diff={k_max_diff:.6f} (threshold={max_abs_error_threshold})",
        )
        self.assertTrue(
            v_error_ok,
            f"V cache error too large: rel_error={v_rel_error:.6f} (threshold={max_rel_error_threshold}), "
            f"max_diff={v_max_diff:.6f} (threshold={max_abs_error_threshold})",
        )

    # ========== Test cases for set_kv_int8 ==========

    def test_set_kv_int8_basic(self):
        """Test basic int8 set KV cache."""
        self._test_set_kv_int8_correctness(
            num_tokens=16,
            num_heads=8,
            head_dim=128,
            cache_size=128,
        )

    def test_set_kv_int8_large_batch(self):
        """Test int8 set KV cache with large batch."""
        self._test_set_kv_int8_correctness(
            num_tokens=128,
            num_heads=16,
            head_dim=64,
            cache_size=256,
        )

    def test_set_kv_int8_different_head_dims(self):
        """Test int8 set KV cache with different head dimensions."""
        for head_dim in [64, 128, 256]:
            self._test_set_kv_int8_correctness(
                num_tokens=32,
                num_heads=8,
                head_dim=head_dim,
                cache_size=128,
            )

    # ========== Test cases for set_kv_int4 ==========

    def test_set_kv_int4_basic(self):
        """Test basic int4 set KV cache."""
        self._test_set_kv_int4_correctness(
            num_tokens=16,
            num_heads=8,
            head_dim=128,
            cache_size=128,
        )

    def test_set_kv_int4_large_batch(self):
        """Test int4 set KV cache with large batch."""
        self._test_set_kv_int4_correctness(
            num_tokens=128,
            num_heads=16,
            head_dim=64,
            cache_size=256,
        )

    def test_set_kv_int4_different_head_dims(self):
        """Test int4 set KV cache with different head dimensions."""
        for head_dim in [64, 128, 256]:
            self._test_set_kv_int4_correctness(
                num_tokens=32,
                num_heads=8,
                head_dim=head_dim,
                cache_size=128,
            )

    # ========== Test cases for flatten_kv_cache ==========

    def test_flatten_kv_cache_int8_basic(self):
        """Test basic int8 flatten KV cache."""
        self._test_flatten_kv_cache_correctness(
            batch_size=4,
            num_heads=8,
            head_dim=128,
            page_size=16,
            quant_policy=8,
        )

    def test_flatten_kv_cache_int4_basic(self):
        """Test basic int4 flatten KV cache."""
        self._test_flatten_kv_cache_correctness(
            batch_size=4,
            num_heads=8,
            head_dim=128,
            page_size=16,
            quant_policy=4,
        )

    def test_flatten_kv_cache_int8_large_batch(self):
        """Test int8 flatten KV cache with large batch."""
        self._test_flatten_kv_cache_correctness(
            batch_size=16,
            num_heads=16,
            head_dim=64,
            page_size=32,
            quant_policy=8,
        )

    def test_flatten_kv_cache_int4_large_batch(self):
        """Test int4 flatten KV cache with large batch."""
        self._test_flatten_kv_cache_correctness(
            batch_size=16,
            num_heads=16,
            head_dim=64,
            page_size=32,
            quant_policy=4,
        )

    # ========== Tests (set_kv + flatten_kv_cache) ==========

    def test_set_and_flatten_int8_basic_rel_error_check(self):
        """Test: int8 set_kv + flatten_kv_cache with relative error check."""
        self._test_set_and_flatten_kv_cache(
            batch_size=4,
            num_heads=8,
            head_dim=128,
            page_size=16,
            kv_dtype="int8",
        )

    def test_set_and_flatten_int4_basic_rel_error_check(self):
        """Test: int4 set_kv + flatten_kv_cache with relative error check."""
        self._test_set_and_flatten_kv_cache(
            batch_size=4,
            num_heads=8,
            head_dim=128,
            page_size=16,
            kv_dtype="int4",
        )

    def test_set_and_flatten_int8_large_batch_rel_error_check(self):
        """Test: int8 set_kv + flatten_kv_cache with large batch and relative error check."""
        self._test_set_and_flatten_kv_cache(
            batch_size=16,
            num_heads=16,
            head_dim=64,
            page_size=32,
            kv_dtype="int8",
        )

    def test_set_and_flatten_int4_large_batch_rel_error_check(self):
        """Test: int4 set_kv + flatten_kv_cache with large batch and relative error check."""
        self._test_set_and_flatten_kv_cache(
            batch_size=16,
            num_heads=16,
            head_dim=64,
            page_size=32,
            kv_dtype="int4",
        )

    def test_set_and_flatten_int8_different_head_dims_rel_error_check(self):
        """Test: int8 with different head dimensions and relative error check."""
        for head_dim in [64, 128, 256]:
            self._test_set_and_flatten_kv_cache(
                batch_size=4,
                num_heads=8,
                head_dim=head_dim,
                page_size=16,
                kv_dtype="int8",
            )

    def test_set_and_flatten_int4_different_head_dims_rel_error_check(self):
        """Test: int4 with different head dimensions and relative error check."""
        for head_dim in [64, 128, 256]:
            self._test_set_and_flatten_kv_cache(
                batch_size=4,
                num_heads=8,
                head_dim=head_dim,
                page_size=16,
                kv_dtype="int4",
            )

    # ========== Edge cases ==========

    def test_set_kv_int8_single_token(self):
        """Test int8 set KV cache with single token."""
        self._test_set_kv_int8_correctness(
            num_tokens=1,
            num_heads=8,
            head_dim=128,
            cache_size=128,
        )

    def test_set_kv_int4_single_token(self):
        """Test int4 set KV cache with single token."""
        self._test_set_kv_int4_correctness(
            num_tokens=1,
            num_heads=8,
            head_dim=128,
            cache_size=128,
        )

    def test_set_kv_int8_empty_input(self):
        """Test int8 set KV cache with empty input."""
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_tokens = 0
        num_heads = 8
        head_dim = 128
        cache_size = 128

        cache_k = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        cache_v = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        k_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim, device=device, dtype=torch.uint8
        )
        v_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim, device=device, dtype=torch.uint8
        )
        k_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )
        v_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )
        cache_loc = torch.empty(num_tokens, device=device, dtype=torch.int32)

        # Should not crash
        quantized_set_kv_int8_triton(
            cache_k,
            cache_v,
            cache_loc,
            k_cache_buffer,
            v_cache_buffer,
            k_scales_zeros,
            v_scales_zeros,
        )

    def test_set_kv_int4_empty_input(self):
        """Test int4 set KV cache with empty input."""
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_tokens = 0
        num_heads = 8
        head_dim = 128
        cache_size = 128

        cache_k = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        cache_v = torch.randn(
            num_tokens, num_heads, head_dim, device=device, dtype=dtype
        )
        k_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim // 2, device=device, dtype=torch.uint8
        )
        v_cache_buffer = torch.zeros(
            cache_size, num_heads, head_dim // 2, device=device, dtype=torch.uint8
        )
        k_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )
        v_scales_zeros = torch.zeros(
            cache_size, num_heads, 2, device=device, dtype=torch.float32
        )
        cache_loc = torch.empty(num_tokens, device=device, dtype=torch.int32)

        # Should not crash
        quantized_set_kv_int4_triton(
            cache_k,
            cache_v,
            cache_loc,
            k_cache_buffer,
            v_cache_buffer,
            k_scales_zeros,
            v_scales_zeros,
        )


if __name__ == "__main__":
    unittest.main()
