"""Request-local FP8 index K preparation for SM90 prefill scoring."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import _e2m1_decode


@triton.jit
def _gather_dequant_fp8(
    Cache,
    Slots,
    Out,
    N: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    SS: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)
    slot = tl.load(Slots + row * SS, row < N, other=0).to(tl.int64)
    page, offset = slot // PAGE, slot % PAGE
    byte = tl.load(
        Cache + page[:, None] * STRIDE + offset[:, None] * 64 + d[None, :] // 2,
        row[:, None] < N,
        other=0,
    ).to(tl.uint8)
    code = (byte >> ((d[None, :] % 2) * 4)) & 15
    exponent = (
        tl.load(
            Cache
            + page[:, None] * STRIDE
            + PAGE * 64
            + offset[:, None] * 4
            + d[None, :] // 32,
            row[:, None] < N,
            other=127,
        )
        .to(tl.uint8)
        .to(tl.float32)
    )
    value = _e2m1_decode(code) * tl.exp2(exponent - 127.0)
    tl.store(Out + row[:, None] * 128 + d[None, :], value, row[:, None] < N)


def gather_index_k_fp8(cache, slots, page_size):
    """Read valid slots from fused [pages, page_size*68] FP4 storage.

    Slot validity is owned by the request table, as in the existing readback.
    Returns shared [len(slots),128] FP8 K with unit scale, without BF16 scratch.
    """
    if (
        cache.ndim != 2
        or cache.element_size() != 1
        or cache.stride(1) != 1
        or page_size <= 0
        or cache.shape[1] < page_size * 68
    ):
        raise ValueError("expected fused contiguous-within-page FP4 cache")
    if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots must be a one-dimensional integer tensor")
    if not cache.is_cuda or slots.device != cache.device:
        raise ValueError("cache and slots must share a CUDA device")
    out = torch.empty(
        (slots.numel(), 128), device=cache.device, dtype=torch.float8_e4m3fn
    )
    if slots.numel():
        _gather_dequant_fp8[(triton.cdiv(slots.numel(), 4),)](
            cache,
            slots,
            out,
            slots.numel(),
            page_size,
            cache.stride(0),
            slots.stride(0),
            ROWS=4,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
