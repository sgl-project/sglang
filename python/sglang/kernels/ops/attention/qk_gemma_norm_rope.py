"""Split q/gate/k from a gated-attention qkv projection, Gemma-RMSNorm q and k
per head and apply neox partial RoPE, in one launch.

Keeps the unfused path's bf16 rounding points (chunk/reshape copies, GemmaRMSNorm
on q and k, then rotary_embedding with an fp32 cos/sin cache): v and gate are
identical, and q/k differ only where the RMSNorm reduction order moves a value to
the neighboring bf16 (a few in 1e5 elements, at most a couple of ulps).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_gemma_norm_rope_kernel(
    qkv_ptr,
    pos_ptr,
    cos_sin_ptr,
    qw_ptr,
    kw_ptr,
    q_out_ptr,
    gate_out_ptr,
    k_out_ptr,
    stride_qkv,
    stride_cs,
    eps,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT_DIM: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = ROT_DIM // 2
    offs_r = tl.arange(0, HALF)
    offs_d = tl.arange(0, HEAD_DIM)
    if h < NUM_HEADS:
        # q_gate is laid out per head as [q (HEAD_DIM), gate (HEAD_DIM)]
        src = qkv_ptr + tok * stride_qkv + h * 2 * HEAD_DIM
        g = tl.load(src + HEAD_DIM + offs_d)
        tl.store(gate_out_ptr + (tok * NUM_HEADS + h) * HEAD_DIM + offs_d, g)
        w_ptr = qw_ptr
        dst = q_out_ptr + (tok * NUM_HEADS + h) * HEAD_DIM
    else:
        kh = h - NUM_HEADS
        src = qkv_ptr + tok * stride_qkv + NUM_HEADS * 2 * HEAD_DIM + kh * HEAD_DIM
        w_ptr = kw_ptr
        dst = k_out_ptr + (tok * NUM_KV_HEADS + kh) * HEAD_DIM

    x = tl.load(src + offs_d).to(tl.float32)
    x1 = tl.load(src + offs_r).to(tl.float32)
    x2 = tl.load(src + HALF + offs_r).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    rstd = 1.0 / tl.sqrt(var + eps)
    out_ty = q_out_ptr.dtype.element_ty
    y1 = (
        (x1 * rstd * (1.0 + tl.load(w_ptr + offs_r).to(tl.float32)))
        .to(out_ty)
        .to(tl.float32)
    )
    y2 = (
        (x2 * rstd * (1.0 + tl.load(w_ptr + HALF + offs_r).to(tl.float32)))
        .to(out_ty)
        .to(tl.float32)
    )
    y = x * rstd * (1.0 + tl.load(w_ptr + offs_d).to(tl.float32))

    pos = tl.load(pos_ptr + tok).to(tl.int64)
    # rotary_embedding does this math in the activation dtype: cos/sin and
    # each product are rounded before the add
    cos = tl.load(cos_sin_ptr + pos * stride_cs + offs_r).to(out_ty).to(tl.float32)
    sin = (
        tl.load(cos_sin_ptr + pos * stride_cs + HALF + offs_r).to(out_ty).to(tl.float32)
    )
    o1 = (y1 * cos).to(out_ty).to(tl.float32) - (y2 * sin).to(out_ty).to(tl.float32)
    o2 = (y2 * cos).to(out_ty).to(tl.float32) + (y1 * sin).to(out_ty).to(tl.float32)
    tl.store(dst + offs_r, o1.to(out_ty))
    tl.store(dst + HALF + offs_r, o2.to(out_ty))
    tl.store(dst + offs_d, y.to(out_ty), mask=offs_d >= ROT_DIM)


def qk_gemma_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
):
    """qkv: [T, num_heads * 2 * head_dim + 2 * num_kv_heads * head_dim] with q/gate
    interleaved per head. Returns q [T, H*D], k [T, Hkv*D], v (a view into qkv)
    and gate [T, H*D]."""
    T = qkv.shape[0]
    q = torch.empty(T, num_heads * head_dim, dtype=qkv.dtype, device=qkv.device)
    gate = torch.empty_like(q)
    k = torch.empty(T, num_kv_heads * head_dim, dtype=qkv.dtype, device=qkv.device)
    kv_off = num_heads * 2 * head_dim
    v = qkv[:, kv_off + num_kv_heads * head_dim : kv_off + 2 * num_kv_heads * head_dim]
    if T > 0:
        _qk_gemma_norm_rope_kernel[(T, num_heads + num_kv_heads)](
            qkv,
            positions,
            cos_sin_cache,
            q_weight,
            k_weight,
            q,
            gate,
            k,
            qkv.stride(0),
            cos_sin_cache.stride(0),
            eps,
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            ROT_DIM=rotary_dim,
            num_warps=1,
        )
    return q, k, v, gate
