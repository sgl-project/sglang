import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import cuda_rsqrtf, mul_rn_f32


@triton.jit
def _table_qk_norm_rope_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    pos_ptr,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    MATCH_CUDA: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half_ar).to(tl.float32)

    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    w_ptr = tl.where(is_q, q_weight_ptr.to(tl.int64), k_weight_ptr.to(tl.int64)).to(
        tl.pointer_type(tl.bfloat16)
    )

    row = qkv_ptr + t * row_stride + col0
    if MATCH_CUDA:
        lane = tl.arange(0, 32)
        part = tl.full((32,), 0, tl.float32)
        for i in tl.static_range(D // 32):
            x = tl.load(row + lane * (D // 32) + i).to(tl.float32)
            part = tl.fma(x, x, part)
        inv = cuda_rsqrtf(tl.fma(tl.sum(part, 0), 1.0 / D, EPS))
    else:
        x = tl.load(row + d_ar).to(tl.float32)
        ms = tl.sum(x * x, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
    w1 = tl.load(w_ptr + half_ar).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
    x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
    x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
    x1 = x1.to(tl.bfloat16).to(tl.float32)
    x2 = x2.to(tl.bfloat16).to(tl.float32)
    if MATCH_CUDA:
        o1 = tl.fma(x1, cos, -mul_rn_f32(x2, sin))
        o2 = tl.fma(x2, cos, mul_rn_f32(x1, sin))
    else:
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
    tl.store(row + half_ar, o1.to(tl.bfloat16))
    tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))


def table_qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    eps: float,
    *,
    match_cuda: bool = False,
) -> None:
    """In-place QK RMSNorm + table-lookup neox RoPE on the fused QKV tensor.

    Reads cos/sin from the SAME rotary table as the unfused path, so there is
    no large-position angle drift (unlike theta-recompute kernels). V columns
    are untouched.

    ``match_cuda`` preserves the reduction and rounding of the separate CUDA
    QK-norm and RoPE kernels for head dimensions 64/128/256. It requires BF16
    QKV with contiguous columns, contiguous BF16 weights, a contiguous FP32
    rotary table, and contiguous int32/int64 positions on the same CUDA device.
    Row-strided QKV is supported.
    """
    if match_cuda:
        assert head_dim in (64, 128, 256)
        assert qkv.ndim == 2 and qkv.stride(1) == 1
        assert qkv.shape[1] == (num_q_heads + 2 * num_k_heads) * head_dim
        assert qkv.dtype == q_weight.dtype == k_weight.dtype == torch.bfloat16
        assert q_weight.shape == k_weight.shape == (head_dim,)
        assert q_weight.is_contiguous() and k_weight.is_contiguous()
        assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == head_dim
        assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()
        assert positions.shape == (qkv.shape[0],) and positions.is_contiguous()
        assert positions.dtype in (torch.int32, torch.int64)
        assert qkv.is_cuda and all(
            t.device == qkv.device
            for t in (positions, q_weight, k_weight, cos_sin_cache)
        )
    T = qkv.shape[0]
    if T == 0:
        return
    grid = (T, num_q_heads + num_k_heads)
    _table_qk_norm_rope_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        qkv.stride(0),
        num_q_heads * head_dim,
        NHQ=num_q_heads,
        D=head_dim,
        EPS=eps,
        MATCH_CUDA=match_cuda,
        num_warps=1 if match_cuda else 4,
    )
