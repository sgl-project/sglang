"""MXFP4 codec (quantize / dequant) correctness.

Verifies:
  1. Triton kernel roundtrip quality (E2M1 is lossy, cos ≈ 0.99)
  2. Triton matches the PyTorch reference (E8M0 rounding may differ slightly)
  3. Empty batches, out-of-range indices, boundary cases
  4. Pre-allocated output buffer reuse
"""

from __future__ import annotations

import torch
from sglang.test.ci.ci_register import register_cuda_ci

from sglang.srt.layers.attention.dsv4.mxfp4_k_cache import (
    MXFP4_BYTES_PER_TOKEN,
    MXFP4_TOTAL_DIM,
    dequantize_dsv4_mxfp4_k_cache_paged,
    quantize_dsv4_mxfp4_k_cache_into,
)

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Tolerance: E2M1 (4-bit) roundtrip typically gives cos ∈ [0.98, 0.998]
# depending on data distribution.
_COS_MIN = 0.97
_ERR_MAX = 5.0


def _pool(page_size: int, num_pages: int = 4) -> torch.Tensor:
    return torch.zeros(
        num_pages,
        page_size * MXFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device="cuda",
    )


def test_roundtrip_quality():
    """Triton quantize → dequant roundtrip preserves signal."""
    torch.manual_seed(1)
    page_size, num_tokens = 128, 32
    dev = torch.device("cuda")

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)
    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)

    cos = torch.nn.functional.cosine_similarity(
        k.float().flatten(), out[:, 0, :].float().flatten(), dim=0
    ).item()
    max_err = (k.float() - out[:, 0, :].float()).abs().max().item()

    assert cos > _COS_MIN, f"cos {cos} below threshold {_COS_MIN}"
    assert max_err < _ERR_MAX, f"max_err {max_err} above threshold {_ERR_MAX}"


def test_triton_vs_reference():
    """Triton output matches PyTorch reference (CPU fallback)."""
    torch.manual_seed(2)
    page_size, num_tokens = 64, 8
    dev = torch.device("cuda")

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)

    # Triton path
    pool_triton = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)
    quantize_dsv4_mxfp4_k_cache_into(k, pool_triton, loc, page_size)
    out_triton = dequantize_dsv4_mxfp4_k_cache_paged(pool_triton, loc, page_size)

    # CPU reference path (move data to CPU, quantize, move back)
    k_cpu = k.cpu()
    pool_cpu = torch.zeros(
        1,
        page_size * MXFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
    )
    loc_cpu = torch.arange(num_tokens, dtype=torch.int32)
    quantize_dsv4_mxfp4_k_cache_into(k_cpu, pool_cpu, loc_cpu, page_size)
    out_ref = dequantize_dsv4_mxfp4_k_cache_paged(pool_cpu, loc_cpu, page_size)

    out_cpu = out_ref[:, 0, :].to(dev).float()
    out_gpu = out_triton[:, 0, :].float()

    diff = (out_cpu - out_gpu).abs()
    max_diff = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(
        out_cpu.flatten(), out_gpu.flatten(), dim=0
    ).item()

    # Quantization decisions (E8M0 rounding) may differ slightly between
    # PyTorch and Triton due to FP32 order-of-operations.
    assert cos > 0.999, f"cos {cos} too low"
    assert max_diff < 1e-3, f"max_diff {max_diff} too high"


def test_empty_batch():
    """Zero-token batch produces no-op."""
    page_size = 128
    k = torch.empty(0, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device="cuda")
    pool = _pool(page_size)
    loc = torch.empty(0, dtype=torch.int32, device="cuda")

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)
    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)

    assert out.shape == (0, 1, MXFP4_TOTAL_DIM)


def test_oob_indices():
    """Out-of-bounds indices yield zero rows (padded graph batch).

    The output is pre-filled with non-zero values so stale memory (or a
    reused pre-allocated buffer) cannot mask a missing zero write.
    """
    torch.manual_seed(3)
    page_size, num_tokens = 32, 16
    dev = torch.device("cuda")
    num_rows = 4 * page_size

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev) - 4  # [-4 .. 11]
    loc[0] = num_rows + 17  # above capacity
    loc[1] = -100  # negative
    assert (loc < 0).any() and (loc >= num_rows).any()

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)

    out = torch.full(
        (num_tokens, 1, MXFP4_TOTAL_DIM), 7.5, dtype=torch.bfloat16, device=dev
    )
    dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size, out=out)

    oob = (loc < 0) | (loc >= num_rows)
    assert (out[oob] == 0).all(), (
        f"OOB rows should be zero, got {(out[oob] != 0).sum().item()} non-zero"
    )
    # In-range rows still dequantize (nonzero signal).
    assert (out[~oob] != 0).any()


def test_pre_allocated_output():
    """Using a pre-allocated output tensor (no allocation in dequant)."""
    torch.manual_seed(4)
    page_size, num_tokens = 64, 8
    dev = torch.device("cuda")

    k = torch.randn(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)

    out1 = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)
    out2 = torch.empty(num_tokens, 1, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    out3 = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size, out=out2)

    assert out3 is out2, "Should return the passed-in tensor"
    assert (out1 == out2).all(), "Pre-allocated output mismatches default-allocated"
