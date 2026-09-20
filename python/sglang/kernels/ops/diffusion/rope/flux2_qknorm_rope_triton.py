"""Packed FLUX.2 Q/K RMSNorm and interleaved RoPE with native rounding."""

import torch
import triton
import triton.language as tl


def can_use_flux2_strided_qknorm_rope(q, k, q_weight, k_weight, cache) -> bool:
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or torch.version.hip is not None
        or not q.is_cuda
        or torch.cuda.get_device_capability(q.device) != (9, 0)
        or q.ndim != 4
        or q.shape[-2:] != (24, 128)
        or q.numel() == 0
        or q.is_contiguous()
        or k.shape != q.shape
        or k.stride() != q.stride()
        or q.stride(-1) != 1
        or q.stride(-2) != 128
        or q.stride(0) != q.shape[1] * q.stride(1)
        or q.stride(1) < 24 * 128
    ):
        return False
    if any(
        t.device != q.device or t.dtype != torch.bfloat16
        for t in (q, k, q_weight, k_weight)
    ):
        return False
    if any(t.shape != (128,) or not t.is_contiguous() for t in (q_weight, k_weight)):
        return False
    return (
        isinstance(cache, torch.Tensor)
        and cache.device == q.device
        and cache.dtype == torch.float32
        and cache.ndim == 2
        and cache.shape[0] >= q.shape[1]
        and cache.shape[1] == 128
        and cache.is_contiguous()
    )


@triton.jit
def _flux2_strided_qknorm_rope_kernel(
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    cache_ptr,
    qo_ptr,
    ko_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    dims = tl.arange(0, 128)[None, :]
    if tl.program_id(1) == 0:
        x_ptr, w_ptr, y_ptr = q_ptr, qw_ptr, qo_ptr
    else:
        x_ptr, w_ptr, y_ptr = k_ptr, kw_ptr, ko_ptr
    offset = (rows // HEADS).to(tl.int64) * TOKEN_STRIDE + rows % HEADS * 128
    x = tl.load(x_ptr + offset + dims, mask=rows < ROWS, other=0.0).to(tl.float32)
    # Keep the exact arithmetic and row tiling of _rms_norm_tiled_onepass.
    mean_square = tl.sum(x * x, axis=1, keep_dims=True) / 128
    rstd = tl.math.rsqrt(mean_square + EPS)
    w = tl.load(w_ptr + dims)
    normalized = (x * rstd * w).to(y_ptr.dtype.element_ty).to(tl.float32)
    partner = tl.gather(
        normalized, tl.broadcast_to(dims ^ 1, (BLOCK_ROWS, 128)), axis=1
    )
    pos = rows // HEADS % TOKENS
    cos = tl.load(
        cache_ptr + pos * CACHE_STRIDE + dims // 2, mask=rows < ROWS, other=0.0
    )
    sin = tl.load(
        cache_ptr + pos * CACHE_STRIDE + 64 + dims // 2, mask=rows < ROWS, other=0.0
    )
    # FlashInfer rounds the sine product before the cosine multiply-add.
    signed_partner = tl.where(dims % 2 == 0, -partner, partner)
    rotated = tl.fma(normalized, cos, signed_partner * sin)
    tl.store(y_ptr + rows * 128 + dims, rotated, mask=rows < ROWS)


def flux2_strided_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, dim = q.shape
    assert dim == 128 and k.shape == q.shape
    assert q.stride() == k.stride()
    assert q.stride(-1) == 1 and q.stride(-2) == 128
    assert q.stride(0) == tokens * q.stride(1)
    assert q.dtype == k.dtype == q_weight.dtype == k_weight.dtype == torch.bfloat16
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.shape[1] == 128
    output_q = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    output_k = torch.empty_like(output_q)
    rows = batch * tokens * heads
    block_rows = min(16, triton.next_power_of_2(max(1, rows // 512)))
    with torch.cuda.device(q.device):
        _flux2_strided_qknorm_rope_kernel[(triton.cdiv(rows, block_rows), 2)](
            q,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            output_q,
            output_k,
            rows,
            tokens,
            heads,
            q.stride(1),
            cos_sin_cache.stride(0),
            eps,
            BLOCK_ROWS=block_rows,
        )
    return output_q, output_k
