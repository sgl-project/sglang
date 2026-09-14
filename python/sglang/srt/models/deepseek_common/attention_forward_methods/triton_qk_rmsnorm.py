"""Triton fused q/k RMSNorm for MLA attention.

Drop-in replacement for aiter's ``fused_qk_rmsnorm`` on hardware where the aiter
``module_fused_qk_norm_rope_cache_quant_shuffle`` kernel cannot be built (e.g.
gfx1250, whose composable_kernel version is incompatible with the aiter fork's
``rope_common.h`` / ``ck_tile/vec_convert.h``). Semantics match a plain
RMSNorm (``sglang.srt.layers.layernorm.RMSNorm.forward_native``): compute the
row variance in fp32, scale by ``rsqrt(var + eps)``, multiply by ``weight`` and
cast back to the input dtype. No RoPE, no quantization -- this is only used on
the ``quant_type=No`` path where the fused kernel is a pure RMSNorm.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    row_stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * row_stride
    out_row = out_ptr + row * row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(out_row + cols, y.to(out_row.dtype.element_ty), mask=mask)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_shape = x.shape
    N = orig_shape[-1]
    x2d = x.reshape(-1, N).contiguous()
    out = torch.empty_like(x2d)

    M = x2d.shape[0]
    if M == 0:
        return out.reshape(orig_shape)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 16)

    _rmsnorm_kernel[(M,)](
        x2d,
        weight,
        out,
        x2d.stride(0),
        N,
        float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)


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
    q_out = _rmsnorm(q, q_weight, q_eps)
    k_out = _rmsnorm(k, k_weight, k_eps)
    return q_out, k_out


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
