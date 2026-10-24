"""Normalize/rotate KV and store directly into the TRT uniform-FP8 cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_norm_rope_fp8_store(
    X,
    W,
    F,
    POS,
    LOC,
    CACHE,
    EPS: tl.constexpr,
    SX: tl.constexpr,
    SF: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * SX
    d = tl.arange(0, 512)
    x = tl.load(X + base + d).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 512 + EPS)
    y = x * inv * w

    # Match _fused_norm_rope_kernel's reduction and RoPE operation order.
    pair = tl.arange(0, 32)
    even = tl.load(X + base + 448 + pair * 2).to(tl.float32)
    odd = tl.load(X + base + 449 + pair * 2).to(tl.float32)
    we = tl.load(W + 448 + pair * 2).to(tl.float32)
    wo = tl.load(W + 449 + pair * 2).to(tl.float32)
    even = even * inv * we
    odd = odd * inv * wo
    pos = tl.load(POS + row).to(tl.int64)
    c = tl.load(F + pos * SF + pair * 2).to(tl.float32)
    s = tl.load(F + pos * SF + pair * 2 + 1).to(tl.float32)
    re = even * c - odd * s
    im = even * s + odd * c

    slot = tl.load(LOC + row).to(tl.int64)
    # Preserve PyTorch indexed assignment's negative-index interpretation.
    slot = tl.where(slot < 0, slot + CACHE_ROWS, slot)
    dst = CACHE + slot * 512
    # The old path materializes BF16/FP16 before torch's FP8 conversion.
    # Keep that rounding boundary without writing the intermediate to HBM.
    y = y.to(X.dtype.element_ty).to(tl.float32)
    re = re.to(X.dtype.element_ty).to(tl.float32)
    im = im.to(X.dtype.element_ty).to(tl.float32)
    tl.store(dst + d, y, d < 448)
    tl.store(dst + 448 + pair * 2, re)
    tl.store(dst + 449 + pair * 2, im)


def fused_k_norm_rope_uniform_fp8(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    cache: torch.Tensor,
) -> None:
    """Store norm+RoPE KV as unit-scale E4M3, without modifying ``kv``.

    ``loc`` addresses contiguous 512-byte rows in the uniform-FP8 pool.
    Active rows must have valid, distinct destinations, as with index_put.
    """
    assert kv.ndim == 2 and kv.shape[1] == 512 and kv.stride(1) == 1
    assert kv.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert weight.shape == (512,) and weight.is_contiguous()
    assert freqs_cis.dtype == torch.complex64 and freqs_cis.is_contiguous()
    assert freqs_cis.ndim == 2 and freqs_cis.shape[1] == 32
    assert positions.shape == loc.shape == (kv.shape[0],)
    assert positions.dtype in (torch.int32, torch.int64)
    assert loc.dtype in (torch.int32, torch.int64)
    assert positions.is_contiguous() and loc.is_contiguous()
    assert cache.dtype == torch.float8_e4m3fn and cache.is_contiguous()
    assert cache.numel() % 512 == 0
    assert all(
        t.device == kv.device for t in (weight, freqs_cis, positions, loc, cache)
    )
    if kv.shape[0] == 0:
        return
    _kv_norm_rope_fp8_store[(kv.shape[0],)](
        kv,
        weight,
        torch.view_as_real(freqs_cis),
        positions,
        loc,
        cache,
        EPS=eps,
        SX=kv.stride(0),
        SF=freqs_cis.stride(0) * 2,
        CACHE_ROWS=cache.numel() // 512,
        num_warps=4,
    )
