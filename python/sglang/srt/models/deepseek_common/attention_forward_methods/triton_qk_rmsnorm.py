"""Triton fused q/k RMSNorm for MLA attention.

gfx1250 replacement for aiter's ``fused_qk_rmsnorm`` that does not depend on the
aiter ``module_fused_qk_norm_rope_cache_quant_shuffle`` JIT build. Semantics match
a plain RMSNorm (``sglang.srt.layers.layernorm.RMSNorm.forward_native``): compute
the row variance in fp32, scale by ``rsqrt(var + eps)``, multiply by ``weight``
and cast back to the input dtype. No RoPE, no quantization -- this is only used
on the ``quant_type=No`` path where the fused kernel is a pure RMSNorm.

``q`` and ``k`` are usually column slices of the fused_qkv_a output, so rows are
read through their own stride and both norms run in one launch.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# Not layernorm.rmsnorm_fp32: 1 / sqrt(var + eps) rounds differently from rsqrt.
# Each branch keeps its own names: Triton needs matching shapes for a name
# assigned in both arms of an if.
@triton.jit
def _qk_rmsnorm_kernel(
    q_ptr,
    q_w_ptr,
    q_out_ptr,
    q_row_stride,
    q_out_row_stride,
    NQ,
    q_eps,
    k_ptr,
    k_w_ptr,
    k_out_ptr,
    k_row_stride,
    k_out_row_stride,
    NK,
    k_eps,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.program_id(1) == 0:
        q_cols = tl.arange(0, BLOCK_Q)
        q_mask = q_cols < NQ
        xq = tl.load(q_ptr + row * q_row_stride + q_cols, mask=q_mask, other=0.0).to(
            tl.float32
        )
        rq = 1.0 / tl.sqrt(tl.sum(xq * xq, axis=0) / NQ + q_eps)
        wq = tl.load(q_w_ptr + q_cols, mask=q_mask, other=0.0).to(tl.float32)
        tl.store(
            q_out_ptr + row * q_out_row_stride + q_cols,
            (xq * rq * wq).to(q_out_ptr.dtype.element_ty),
            mask=q_mask,
        )
    else:
        k_cols = tl.arange(0, BLOCK_K)
        k_mask = k_cols < NK
        xk = tl.load(k_ptr + row * k_row_stride + k_cols, mask=k_mask, other=0.0).to(
            tl.float32
        )
        rk = 1.0 / tl.sqrt(tl.sum(xk * xk, axis=0) / NK + k_eps)
        wk = tl.load(k_w_ptr + k_cols, mask=k_mask, other=0.0).to(tl.float32)
        tl.store(
            k_out_ptr + row * k_out_row_stride + k_cols,
            (xk * rk * wk).to(k_out_ptr.dtype.element_ty),
            mask=k_mask,
        )


def _rows(x: torch.Tensor) -> torch.Tensor:
    # The kernel reads each row with unit column stride and any row stride.
    x2d = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    return x2d if x2d.stride(-1) == 1 else x2d.contiguous()


def fused_qk_rmsnorm_triton(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm ``q`` (with ``q_weight``/``q_eps``) and ``k`` (with
    ``k_weight``/``k_eps``) independently. Matches the signature and return
    convention of the aiter ``fused_qk_rmsnorm`` shim used in ``forward_mla``.
    """
    q2d, k2d = _rows(q), _rows(k)
    M, NQ, NK = q2d.shape[0], q2d.shape[1], k2d.shape[1]
    assert k2d.shape[0] == M, f"q has {M} rows, k has {k2d.shape[0]}"
    q_out = torch.empty((M, NQ), dtype=q.dtype, device=q.device)
    k_out = torch.empty((M, NK), dtype=k.dtype, device=k.device)
    if M:
        block_q = triton.next_power_of_2(NQ)
        block_k = triton.next_power_of_2(NK)
        _qk_rmsnorm_kernel[(M, 2)](
            q2d,
            q_weight,
            q_out,
            q2d.stride(0),
            q_out.stride(0),
            NQ,
            float(q_eps),
            k2d,
            k_weight,
            k_out,
            k2d.stride(0),
            k_out.stride(0),
            NK,
            float(k_eps),
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            # Sized by the smaller row: on gfx1250 K2 shapes (1536 / 512) this is
            # the fastest choice for prefill rows and even for decode.
            num_warps=min(max(min(block_q, block_k) // 256, 1), 16),
        )
    return q_out.reshape(q.shape), k_out.reshape(k.shape)


def _rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf * weight).to(orig_dtype)


def fused_qk_rmsnorm_torch(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference equivalent of ``fused_qk_rmsnorm_triton`` (bisect
    aid for the decode degeneration investigation)."""
    return _rmsnorm_torch(q, q_weight, q_eps), _rmsnorm_torch(k, k_weight, k_eps)
