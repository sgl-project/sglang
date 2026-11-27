"""MLA's q_a and kv_a RMSNorms in one launch.

The fused q/kv-a projection gives each token one row ``[q | k_nope | k_pe]``.
``mla_qkv_a_rmsnorm`` normalizes ``q`` and ``k_nope``, each with its own weight
and epsilon, in one launch: one program per row and part. Given the
projection's FP32 split-K partials instead of its output, it first sums them in
index order and rounds once, as the split-K GEMM's own sum kernel does, and also
returns that sum (the projection's output).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _rmsnorm_store(x, w_ptr, out_ptr, cols, mask, width, eps):
    xf = tl.where(mask, x.to(tl.float32), 0.0)
    rstd = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / width + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # x * w is exact in fp32; scaling it rounds once, as RMSNorm's JIT kernel does.
    tl.store(out_ptr + cols, (xf * w * rstd).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["src_row_stride", "src_split_stride"])
def _mla_qkv_a_rmsnorm_kernel(
    src_ptr,
    sum_ptr,
    q_out_ptr,
    kv_out_ptr,
    q_weight_ptr,
    kv_weight_ptr,
    src_row_stride,
    src_split_stride,
    q_eps,
    kv_eps,
    Q: tl.constexpr,
    KV: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # Program (row, 0) owns columns [0, Q) of the row, (row, 1) owns [Q, WIDTH).
    row = tl.program_id(0)
    is_kv = tl.program_id(1)
    start = is_kv * Q
    cols = tl.arange(0, BLOCK)
    in_part = cols < Q + is_kv * (WIDTH - 2 * Q)
    src = src_ptr + row * src_row_stride + start + cols
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if SPLIT_K == 0:
        x = tl.load(src, mask=in_part, other=0.0)
    else:
        # Partials are added in index order, then rounded once.
        acc = tl.load(src, mask=in_part, other=0.0)
        for s in tl.static_range(1, SPLIT_K):
            acc += tl.load(src + s * src_split_stride, mask=in_part, other=0.0)
        x = acc.to(sum_ptr.dtype.element_ty)
        tl.store(sum_ptr + row * WIDTH + start + cols, x, mask=in_part)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if is_kv == 0:
        _rmsnorm_store(x, q_weight_ptr, q_out_ptr + row * Q, cols, cols < Q, Q, q_eps)
    else:
        _rmsnorm_store(
            x, kv_weight_ptr, kv_out_ptr + row * KV, cols, cols < KV, KV, kv_eps
        )


def mla_qkv_a_rmsnorm(
    src: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    kv_weight: torch.Tensor,
    kv_eps: float,
    *,
    pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(latent, q, k_nope)``: the projection output ``[M, WIDTH]``,
    ``rmsnorm(latent[:, :Q])`` and ``rmsnorm(latent[:, Q:Q + KV])``, both
    contiguous. ``src`` is either the projection output (unit column stride),
    which is returned as ``latent``, or its FP32 split-K partials
    ``[SPLIT_K, M, WIDTH]`` (contiguous), whose sum is returned in the weights'
    dtype. Q and KV are the weights' sizes."""
    q_size, kv_size = q_weight.shape[0], kv_weight.shape[0]
    if src.dim() == 3:
        split_k, num_rows, width = src.shape
        assert src.dtype == torch.float32 and src.is_contiguous()
        latent = src.new_empty((num_rows, width), dtype=q_weight.dtype)
        row_stride, split_stride = width, num_rows * width
    else:
        split_k = 0
        num_rows, width = src.shape
        assert src.stride(1) == 1
        latent = src
        row_stride, split_stride = src.stride(0), 0
    assert q_size + kv_size <= width and kv_weight.dtype == q_weight.dtype
    q = latent.new_empty((num_rows, q_size))
    kv = latent.new_empty((num_rows, kv_size))
    if num_rows == 0:
        return latent, q, kv
    block = triton.next_power_of_2(max(q_size, width - q_size))
    use_pdl = is_arch_support_pdl() if pdl is None else pdl
    _mla_qkv_a_rmsnorm_kernel[(num_rows, 2)](
        src,
        latent,
        q,
        kv,
        q_weight,
        kv_weight,
        row_stride,
        split_stride,
        q_eps,
        kv_eps,
        Q=q_size,
        KV=kv_size,
        WIDTH=width,
        SPLIT_K=split_k,
        BLOCK=block,
        USE_PDL=use_pdl,
        num_warps=max(1, min(16, block // 256)),
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return latent, q, kv
