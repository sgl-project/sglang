"""Fused across-heads q/k RMSNorm + interleaved (GPT-J) RoPE for ROCm.

Reads q and k straight out of the packed QKV GEMM output (row stride 3 * D),
so neither needs a contiguous copy first, and writes both contiguous.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    in_stride,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    cs_stride,
    qo_ptr,
    ko_ptr,
    out_stride,
    eps,
    D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)
    if which == 0:
        src = q_ptr
        w_ptr = wq_ptr
        dst = qo_ptr
    else:
        src = k_ptr
        w_ptr = wk_ptr
        dst = ko_ptr
    base = src + row.to(tl.int64) * in_stride
    obase = dst + row.to(tl.int64) * out_stride
    offs = tl.arange(0, CHUNK)
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for c in tl.static_range(D // CHUNK):
        x = tl.load(base + c * CHUNK + offs).to(tl.float32)
        acc += x * x
    rstd = tl.rsqrt(tl.sum(acc, axis=0) / D + eps)
    half: tl.constexpr = HEAD_DIM // 2
    pidx = tl.arange(0, CHUNK // 2)
    cs_base = row.to(tl.int64) * cs_stride
    for c in tl.static_range(D // CHUNK):
        x = tl.load(base + c * CHUNK + offs).to(tl.float32)
        w = tl.load(w_ptr + c * CHUNK + offs).to(tl.float32)
        n = (x * rstd * w).to(dst.dtype.element_ty).to(tl.float32)
        n0, n1 = tl.split(tl.reshape(n, (CHUNK // 2, 2)))
        f = (c * (CHUNK // 2) + pidx) % half
        cs = tl.load(cos_ptr + cs_base + f).to(tl.float32)
        sn = tl.load(sin_ptr + cs_base + f).to(tl.float32)
        o0 = n0 * cs - n1 * sn
        o1 = n1 * cs + n0 * sn
        o = tl.interleave(o0, o1)
        tl.store(obase + c * CHUNK + offs, o.to(dst.dtype.element_ty))


def rocm_qk_norm_rope(
    q2d: torch.Tensor,
    k2d: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
    inplace: bool = False,
):
    """q2d/k2d: [M, D] views with unit column stride. cos/sin: [M, head_dim // 2].
    Returns (q, k) of shape [M, D], or None when unsupported. ``inplace`` writes
    the result back into q2d/k2d (each row is fully read before it is written):
    MHA v4 runs faster on q/k/v left as views of one packed QKV buffer."""
    M, D = q2d.shape
    chunk = 1024
    if (
        q2d.dtype not in (torch.bfloat16, torch.float16)
        or k2d.shape != q2d.shape
        or k2d.dtype != q2d.dtype
        or q2d.stride(1) != 1
        or k2d.stride(1) != 1
        or q2d.stride(0) != k2d.stride(0)
        or D % chunk
        or chunk % head_dim
        or head_dim % 2
        or cos.shape != (M, head_dim // 2)
        or sin.shape != cos.shape
        or cos.stride(1) != 1
        or sin.stride() != cos.stride()
        or wq.numel() != D
        or wk.numel() != D
    ):
        return None
    if inplace:
        qo, ko = q2d, k2d
    else:
        qo = torch.empty((M, D), dtype=q2d.dtype, device=q2d.device)
        ko = torch.empty_like(qo)
    _qk_norm_rope_kernel[(M, 2)](
        q2d,
        k2d,
        q2d.stride(0),
        wq,
        wk,
        cos,
        sin,
        cos.stride(0),
        qo,
        ko,
        qo.stride(0),
        eps,
        D=D,
        HEAD_DIM=head_dim,
        CHUNK=chunk,
        num_warps=4,
    )
    return qo, ko
