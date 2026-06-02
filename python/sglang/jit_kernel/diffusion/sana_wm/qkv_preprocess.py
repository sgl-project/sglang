# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sana_wm_qkv_gdn_preprocess_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    bn = tl.program_id(0)
    b = bn // N
    n = bn - b * N

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    h = offs_c // D
    d = offs_c - h * D

    qkv_base = bn * 3 * C
    q = tl.load(qkv_ptr + qkv_base + offs_c, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(qkv_ptr + qkv_base + C + offs_c, mask=mask, other=0.0).to(
        tl.float32
    )

    q_rstd = tl.rsqrt(tl.sum(q * q, axis=0) / C + EPS)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=0) / C + EPS)

    q_weight = tl.load(q_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    q = tl.maximum(q * q_rstd * q_weight, 0.0)
    k = tl.maximum(k * k_rstd * k_weight, 0.0) * K_SCALE
    v = tl.load(qkv_ptr + qkv_base + 2 * C + offs_c, mask=mask, other=0.0)

    # Output layout is contiguous (B, H, D, N), the layout consumed by the
    # GDN scan. This removes two PyTorch permutes from the hot path.
    out_base = ((b * H + h) * D + d) * N + n
    tl.store(q_out_ptr + out_base, q, mask=mask)
    tl.store(k_out_ptr + out_base, k, mask=mask)
    tl.store(v_out_ptr + out_base, v, mask=mask)


def sana_wm_qkv_gdn_preprocess(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SANA-WM GDN Q/K RMSNorm, ReLU, K scale, and QKV transpose.

    Args:
        qkv: Contiguous ``(B, N, 3, H, D)`` tensor. The K short-convolution
            must already be reflected in ``qkv``.
        q_weight: RMSNorm weight over the flattened ``H * D`` channel axis.
        k_weight: RMSNorm weight over the flattened ``H * D`` channel axis.
        k_scale: SANA-WM key scale ``D**-0.5 * S**-0.5``.
        eps: RMSNorm epsilon.

    Returns:
        ``(q, k, v)`` tensors with contiguous ``(B, H, D, N)`` layout.
    """
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Expected qkv shape (B, N, 3, H, D), got {qkv.shape}.")
    if not qkv.is_cuda:
        raise ValueError("sana_wm_qkv_gdn_preprocess requires a CUDA tensor.")
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    B, N, _, H, D = qkv.shape
    C = H * D
    if q_weight.numel() != C or k_weight.numel() != C:
        raise ValueError(
            "SANA-WM q/k norm weights must match the flattened local channel "
            f"dimension C={C}, got {q_weight.numel()} and {k_weight.numel()}."
        )

    q_out = torch.empty((B, H, D, N), device=qkv.device, dtype=qkv.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    block_c = triton.next_power_of_2(C)
    num_warps = 8 if block_c >= 2048 else 4

    with torch.get_device_module().device(qkv.device):
        _sana_wm_qkv_gdn_preprocess_kernel[(B * N,)](
            qkv,
            q_out,
            k_out,
            v_out,
            q_weight.contiguous(),
            k_weight.contiguous(),
            N=N,
            H=H,
            D=D,
            C=C,
            EPS=eps,
            K_SCALE=k_scale,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return q_out, k_out, v_out


def can_use_sana_wm_qkv_gdn_preprocess(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> bool:
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        return False
    if not qkv.is_cuda or not qkv.is_contiguous():
        return False
    if qkv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    _, _, _, H, D = qkv.shape
    return q_weight.numel() == H * D and k_weight.numel() == H * D

