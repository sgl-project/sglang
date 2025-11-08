# -*- coding: utf-8 -*-
"""
PyTest: correctness of fused MLA RoPE + FP8 quantization + KV write.
Tests the mla_rope_quantize_fp8_fused kernel from sgl_kernel extension.

Tests both:
1. Baseline path (kernel writes to k_nope_out/k_rope_out)
2. Fused path (kernel directly writes to KV cache buffer)
"""
import itertools

import pytest
import torch
from sgl_kernel import mla_rope_quantize_fp8_fused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "nnz,num_heads,Dn,Dr,dtype",
    list(
        itertools.product(
            [64, 256],  # nnz: number of tokens
            [1, 8],  # num_heads: 1 for 2D Q, 8 for 3D Q
            [512],  # Dn: nope dimension
            [64],  # Dr: rope dimension
            [torch.float16, torch.bfloat16],  # dtypes
        )
    ),
)
def test_fused_matches_baseline(nnz, num_heads, Dn, Dr, dtype):
    """Test that fused KV write produces same results as baseline."""
    device = "cuda"
    torch.manual_seed(42)

    # Create inputs based on whether we're testing 2D or 3D Q
    if num_heads == 1:
        # 2D case: [nnz, dim]
        q_nope = torch.randn(nnz, Dn, device=device, dtype=dtype)
        q_rope = torch.randn(nnz, Dr, device=device, dtype=dtype)
        q_out_shape = (nnz, Dn + Dr)
    else:
        # 3D case: [nnz, num_heads, dim]
        q_nope = torch.randn(nnz, num_heads, Dn, device=device, dtype=dtype)
        q_rope = torch.randn(nnz, num_heads, Dr, device=device, dtype=dtype)
        q_out_shape = (nnz, num_heads, Dn + Dr)

    # K is always 2D regardless of Q shape
    k_nope = torch.randn(nnz, Dn, device=device, dtype=dtype)
    k_rope = torch.randn(nnz, Dr, device=device, dtype=dtype)

    # Create cos/sin cache
    max_seq = max(2048, nnz)
    t = torch.linspace(0, 1, steps=max_seq, device=device, dtype=torch.float32)[:, None]
    idx = torch.arange(Dr, device=device, dtype=torch.float32)[None, :]
    freqs = 0.1 * (idx + 1.0)  # Small frequencies to avoid overflow
    cos = torch.cos(t * freqs)
    sin = torch.sin(t * freqs)
    cos_sin = torch.cat([cos, sin], dim=1)  # [max_seq, 2*Dr]

    # Random position IDs
    pos_ids = torch.randint(
        low=0, high=max_seq, size=(nnz,), device=device, dtype=torch.long
    )

    # ========================================================================
    # BASELINE PATH: Write to k_nope_out/k_rope_out, then concat manually
    # ========================================================================
    q_out_base = torch.empty(q_out_shape, device=device, dtype=torch.uint8)
    k_nope_out = torch.empty(nnz, Dn, device=device, dtype=torch.uint8)
    k_rope_out = torch.empty(nnz, Dr, device=device, dtype=torch.uint8)

    mla_rope_quantize_fp8_fused(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        cos_sin,
        pos_ids,
        False,  # is_neox
        q_out_base,
        k_nope_out,
        k_rope_out,
        None,  # kv_buffer
        None,  # kv_cache_loc
    )

    # Manually concat K parts into KV buffer (simulating set_mla_kv_buffer)
    slots = nnz + 8  # Add some extra slots
    kv_base = torch.zeros(slots, Dn + Dr, device=device, dtype=torch.uint8)  # 2D format
    loc = torch.arange(nnz, device=device, dtype=torch.long)
    kv_base[loc, :Dn] = k_nope_out
    kv_base[loc, Dn:] = k_rope_out

    # ========================================================================
    # FUSED PATH: Direct KV write, skip separate K outputs
    # ========================================================================
    q_out_fused = torch.empty_like(q_out_base)
    kv_fused = torch.zeros_like(kv_base)

    mla_rope_quantize_fp8_fused(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        cos_sin,
        pos_ids,
        False,  # is_neox
        q_out_fused,
        None,  # k_nope_out
        None,  # k_rope_out
        kv_fused,  # Direct write to KV buffer
        loc,  # kv_cache_loc
    )

    # ========================================================================
    # ASSERTIONS
    # ========================================================================
    # For FP8 quantized outputs, we can't expect perfect match due to
    # rounding, but they should be very close when converted back to float
    torch.testing.assert_close(
        q_out_base.float(),
        q_out_fused.float(),
        rtol=1e-3,
        atol=1e-3,
        msg=f"q_out mismatch for {dtype=}, {num_heads=}",
    )

    torch.testing.assert_close(
        kv_base.float(),
        kv_fused.float(),
        rtol=1e-3,
        atol=1e-3,
        msg=f"KV buffer mismatch for {dtype=}, {num_heads=}",
    )

    # For stricter check on used slots (should be exact)
    assert torch.equal(
        kv_base[loc], kv_fused[loc]
    ), f"Used KV slots must match exactly for {dtype=}, {num_heads=}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("nnz,Dn,Dr", [(128, 512, 64), (1024, 512, 64)])
def test_baseline_only_path(nnz, Dn, Dr):
    """Test that baseline path (without KV buffer) works correctly."""
    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(42)

    # 2D inputs
    q_nope = torch.randn(nnz, Dn, device=device, dtype=dtype)
    q_rope = torch.randn(nnz, Dr, device=device, dtype=dtype)
    k_nope = torch.randn(nnz, Dn, device=device, dtype=dtype)
    k_rope = torch.randn(nnz, Dr, device=device, dtype=dtype)

    # Create cos/sin cache
    max_seq = 2048
    t = torch.linspace(0, 1, steps=max_seq, device=device, dtype=torch.float32)[:, None]
    idx = torch.arange(Dr, device=device, dtype=torch.float32)[None, :]
    freqs = 0.1 * (idx + 1.0)
    cos = torch.cos(t * freqs)
    sin = torch.sin(t * freqs)
    cos_sin = torch.cat([cos, sin], dim=1)

    pos_ids = torch.randint(
        low=0, high=max_seq, size=(nnz,), device=device, dtype=torch.long
    )

    # Allocate outputs
    q_out = torch.empty(nnz, Dn + Dr, device=device, dtype=torch.uint8)
    k_nope_out = torch.empty(nnz, Dn, device=device, dtype=torch.uint8)
    k_rope_out = torch.empty(nnz, Dr, device=device, dtype=torch.uint8)

    # Call kernel (baseline path only)
    mla_rope_quantize_fp8_fused(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        cos_sin,
        pos_ids,
        False,
        q_out,
        k_nope_out,
        k_rope_out,
        None,  # No KV buffer
        None,  # No kv_cache_loc
    )

    # Basic sanity checks
    assert q_out.shape == (nnz, Dn + Dr)
    assert k_nope_out.shape == (nnz, Dn)
    assert k_rope_out.shape == (nnz, Dr)

    # Check that outputs are not all zeros (actual quantization happened)
    assert q_out.abs().sum() > 0, "q_out should not be all zeros"
    assert k_nope_out.abs().sum() > 0, "k_nope_out should not be all zeros"
    assert k_rope_out.abs().sum() > 0, "k_rope_out should not be all zeros"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_only_path():
    """Test that fused path (only KV buffer, no separate K outputs) works."""
    device = "cuda"
    dtype = torch.float16
    nnz, Dn, Dr = 128, 512, 64
    torch.manual_seed(42)

    # 2D inputs
    q_nope = torch.randn(nnz, Dn, device=device, dtype=dtype)
    q_rope = torch.randn(nnz, Dr, device=device, dtype=dtype)
    k_nope = torch.randn(nnz, Dn, device=device, dtype=dtype)
    k_rope = torch.randn(nnz, Dr, device=device, dtype=dtype)

    # Create cos/sin cache
    max_seq = 2048
    t = torch.linspace(0, 1, steps=max_seq, device=device, dtype=torch.float32)[:, None]
    idx = torch.arange(Dr, device=device, dtype=torch.float32)[None, :]
    freqs = 0.1 * (idx + 1.0)
    cos = torch.cos(t * freqs)
    sin = torch.sin(t * freqs)
    cos_sin = torch.cat([cos, sin], dim=1)

    pos_ids = torch.randint(
        low=0, high=max_seq, size=(nnz,), device=device, dtype=torch.long
    )

    # Allocate only Q output and KV buffer
    q_out = torch.empty(nnz, Dn + Dr, device=device, dtype=torch.uint8)
    slots = nnz + 16
    kv_buffer = torch.zeros(
        slots, Dn + Dr, device=device, dtype=torch.uint8
    )  # 2D format
    loc = torch.arange(nnz, device=device, dtype=torch.long)

    # Call kernel (fused path only)
    mla_rope_quantize_fp8_fused(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        cos_sin,
        pos_ids,
        False,
        q_out,
        None,  # No k_nope_out
        None,  # No k_rope_out
        kv_buffer,  # Direct KV write
        loc,
    )

    # Check that KV buffer was written
    assert kv_buffer[loc].abs().sum() > 0, "KV buffer should have been written"
    # Check unused slots are still zero
    unused = torch.ones(slots, dtype=torch.bool, device=device)
    unused[loc] = False
    assert kv_buffer[unused].abs().sum() == 0, "Unused KV slots should remain zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
