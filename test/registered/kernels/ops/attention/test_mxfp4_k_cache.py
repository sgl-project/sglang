"""MXFP4 codec (quantize / dequant) correctness.

Verifies:
  1. Triton kernel roundtrip quality (E2M1 is lossy, cos ≈ 0.99)
  2. Triton matches the PyTorch reference (E8M0 rounding may differ slightly)
  3. Empty batches, out-of-range indices, boundary cases
  4. Pre-allocated output buffer reuse
"""

from __future__ import annotations

import math

import torch

from sglang.kernels.ops.attention.dsv4.mxfp4_k_cache import (
    MXFP4_BYTES_PER_TOKEN,
    MXFP4_GROUP_SIZE,
    MXFP4_NOPE_DIM,
    MXFP4_NUM_GROUPS,
    MXFP4_TOTAL_DIM,
    dequantize_dsv4_mxfp4_k_cache_paged,
    quantize_dsv4_mxfp4_k_cache_into,
)
from sglang.test.ci.ci_register import register_cuda_ci

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


def test_scale_is_ceil_never_saturates_group_amax():
    """The E8M0 byte must follow ceil(log2(amax / 6)) (MX convention).

    The original round(log2(amax / 6)) picked a scale one octave too small
    for every group whose amax lands in (6·2^k, 6·2^k·2^0.5]: the max
    element then saturated to the largest E2M1 code (6), returning up to
    29% short after dequantization (amax 8 decoded as 6).  ceil guarantees
    the group amax stays representable.
    """
    dev = torch.device("cuda")
    page_size = 128
    # All values are exact in bf16.  Entries whose log2(amax/6) is an
    # integer sit exactly on a rounding boundary — the kernel's fp32
    # log2(amax) - log2(6) may land a ulp on either side, so either
    # adjacent byte is accepted there (both keep the amax representable).
    # The other entries pin the exact ceil byte, including the (6, 8.49]
    # band the old round scale saturated.
    amaxes = [6.0, 12.0, 0.75, 1.5, 3.0, 7.0, 8.0, 15.0, 4.0, 0.5]
    k = torch.zeros(1, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    for group, amax in enumerate(amaxes):
        k[0, group * 32 : (group + 1) * 32] = amax / 4.0
        k[0, group * 32] = amax  # the group's max element
    pool = _pool(page_size, num_pages=1)
    loc = torch.zeros(1, dtype=torch.int32, device=dev)

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)
    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)[:, 0, :].float()

    grid = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for group, amax in enumerate(amaxes):
        byte = int(pool[0, 224 + group])
        ratio = math.log2(amax / 6.0)
        s_low = math.ceil(ratio)
        on_boundary = ratio == math.floor(ratio)
        if on_boundary:
            assert byte in (s_low + 127, s_low + 128), (
                f"group {group} amax {amax}: byte {byte} escaped the "
                f"({s_low + 127}, {s_low + 128}) boundary pair"
            )
        else:
            assert (
                byte == s_low + 127
            ), f"group {group} amax {amax}: byte {byte} != {s_low + 127}"
        s = byte - 127
        recovered = out[0, group * 32].item()
        scaled = amax / 2.0**s
        if scaled in grid:
            # amax/2^s on the E2M1 grid: the max element roundtrips exactly.
            assert recovered == amax, f"group {group}: {amax} -> {recovered}"
        else:
            # scaled ∈ (3, 6) off-grid (7/2 = 3.5 tie, 15/4 = 3.75): RNE
            # lands on the 4 code — never on the 6·2^(s-1) clamp the round
            # scale produced.
            assert (
                recovered == 4.0 * 2.0**s
            ), f"group {group}: amax {amax} recovered as {recovered}"


def test_e8m0_byte_zero_decodes_as_2_pow_minus_127():
    """E8M0 has no zero encoding: byte 0 is 2^-127 (bytes 0..254 = 2^-127
    .. 2^127, 255 reserved NaN), matching the fused decode kernel's
    e8m0_bits_to_float.  The dequantizer must apply that exact exponent
    rather than treating the byte as a zero scale."""
    dev = torch.device("cuda")
    page_size = 128
    k = torch.zeros(1, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size, num_pages=1)
    loc = torch.zeros(1, dtype=torch.int32, device=dev)
    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)
    # Hand-craft the first group: E2M1 code 0b101 (=3.0) in every nibble
    # with scale byte 0 — the byte lives at row offset 224.
    pool[0, 0:16] = torch.full((16,), 0x55, dtype=torch.uint8, device=dev)
    pool[0, 224] = 0
    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)[:, 0, :].float()
    expected = torch.full((32,), 3.0 * 2.0**-127, device=dev)
    assert torch.equal(out[0, :32], expected)


def test_all_zero_block_quantizes_to_zero_codes():
    """An all-zero block must quantize to E2M1 code 0 with scale byte 0 and
    round-trip to exactly zero (regression: the quantizer derived the RNE
    denominator from tl.math.exp2, which flushes the subnormal 2^-127 to
    zero; the `magnitude >= denominator * midpoint` rungs then all held at
    magnitude 0 and zero inputs were encoded as code 3 — nonzero packed
    nibbles that also diverged from the PyTorch reference).  All-zero rows
    occur in production: idle dummy stores write them into the pool."""
    dev = torch.device("cuda")
    page_size, num_tokens = 128, 4
    k = torch.zeros(num_tokens, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    pool = _pool(page_size)
    loc = torch.arange(num_tokens, dtype=torch.int32, device=dev)

    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)

    rows = pool.view(-1, MXFP4_BYTES_PER_TOKEN)[:num_tokens]
    packed = rows[:, : MXFP4_NOPE_DIM // 2]
    scales = rows[:, MXFP4_NOPE_DIM // 2 : MXFP4_NOPE_DIM // 2 + MXFP4_NUM_GROUPS]
    assert torch.equal(packed, torch.zeros_like(packed))
    assert torch.equal(scales, torch.zeros_like(scales))

    out = dequantize_dsv4_mxfp4_k_cache_paged(pool, loc, page_size)
    nope = out[:, 0, :MXFP4_NOPE_DIM].float()
    assert torch.equal(nope, torch.zeros_like(nope))

    # The same must hold for a zero group inside an otherwise healthy row:
    # its packed bytes and its scale byte stay zero while the row's other
    # groups encode normally.
    torch.manual_seed(3)
    k = torch.randn(1, MXFP4_TOTAL_DIM, dtype=torch.bfloat16, device=dev)
    k[0, :MXFP4_GROUP_SIZE] = 0
    pool = _pool(page_size)
    loc = torch.zeros(1, dtype=torch.int32, device=dev)
    quantize_dsv4_mxfp4_k_cache_into(k, pool, loc, page_size)
    row = pool.view(-1, MXFP4_BYTES_PER_TOKEN)[0]
    group_bytes = MXFP4_GROUP_SIZE // 2
    assert torch.equal(
        row[:group_bytes], torch.zeros(group_bytes, dtype=torch.uint8, device=dev)
    )
    assert row[MXFP4_NOPE_DIM // 2] == 0
    assert not torch.equal(
        row[group_bytes : MXFP4_NOPE_DIM // 2],
        torch.zeros(MXFP4_NOPE_DIM // 2 - group_bytes, dtype=torch.uint8, device=dev),
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
    assert (
        out[oob] == 0
    ).all(), f"OOB rows should be zero, got {(out[oob] != 0).sum().item()} non-zero"
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
