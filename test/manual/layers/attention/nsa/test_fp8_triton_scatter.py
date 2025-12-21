"""
Unit tests for Triton FP8 scatter kernel used in MLA KV cache writes.
Tests correctness against PyTorch index_put_ baseline.
"""

import torch

from sglang.srt.mem_cache.utils import set_mla_kv_buffer_fp8_triton


def test_triton_scatter_2d():
    """Test 2D scatter operation (basic case)."""
    device = "cuda"
    size, dim = 1000, 576
    n = 100

    kv_buffer = torch.zeros(size, dim, dtype=torch.uint8, device=device)
    loc = torch.randperm(size, dtype=torch.int32, device=device)[:n]  # Unique indices
    cache_k = torch.randint(0, 256, (n, dim), dtype=torch.uint8, device=device)

    # Baseline: PyTorch index_put_
    kv_buffer_baseline = kv_buffer.clone()
    kv_buffer_baseline[loc] = cache_k

    # Test: Triton scatter
    kv_buffer_triton = kv_buffer.clone()
    set_mla_kv_buffer_fp8_triton(kv_buffer_triton, loc, cache_k)

    assert torch.equal(
        kv_buffer_baseline, kv_buffer_triton
    ), "Triton scatter result does not match index_put_ baseline"


def test_triton_scatter_3d():
    """Test 3D scatter operation (with page_size=1 dimension, as used in MLA)."""
    device = "cuda"
    size, page_size, dim = 1000, 1, 576
    n = 100

    kv_buffer = torch.zeros(size, page_size, dim, dtype=torch.uint8, device=device)
    loc = torch.randperm(size, dtype=torch.int32, device=device)[:n]  # Unique indices
    cache_k = torch.randint(
        0, 256, (n, page_size, dim), dtype=torch.uint8, device=device
    )

    kv_buffer_baseline = kv_buffer.clone()
    kv_buffer_baseline[loc] = cache_k

    kv_buffer_triton = kv_buffer.clone()
    set_mla_kv_buffer_fp8_triton(kv_buffer_triton, loc, cache_k)

    assert torch.equal(
        kv_buffer_baseline, kv_buffer_triton
    ), "3D scatter result does not match baseline"


def test_triton_scatter_large_scale():
    """Stress test with realistic MLA dimensions (DeepSeek-V3.2 scale)."""
    device = "cuda"
    size, dim = 10000, 1152
    n = 512

    kv_buffer = torch.zeros(size, dim, dtype=torch.uint8, device=device)
    loc = torch.randperm(size, dtype=torch.int32, device=device)[:n]  # Unique indices
    cache_k = torch.randint(0, 256, (n, dim), dtype=torch.uint8, device=device)

    kv_buffer_baseline = kv_buffer.clone()
    kv_buffer_baseline[loc] = cache_k

    kv_buffer_triton = kv_buffer.clone()
    set_mla_kv_buffer_fp8_triton(kv_buffer_triton, loc, cache_k)

    assert torch.equal(
        kv_buffer_baseline, kv_buffer_triton
    ), "Large-scale scatter does not match baseline"


if __name__ == "__main__":
    test_triton_scatter_2d()
    test_triton_scatter_3d()
    test_triton_scatter_large_scale()
    print("All tests passed!")
