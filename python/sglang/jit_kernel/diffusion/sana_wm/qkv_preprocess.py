# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

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


@triton.jit
def _sana_wm_fused_qk_inv_rms_kernel(
    qkv_ptr,
    q_inv_rms_ptr,
    k_inv_rms_ptr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    bn = tl.program_id(0)
    row_base = bn * 3 * C
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    q = tl.load(qkv_ptr + row_base + offs_c, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(qkv_ptr + row_base + C + offs_c, mask=mask, other=0.0).to(
        tl.float32
    )

    q_inv = tl.rsqrt(tl.sum(q * q, axis=0) / C + EPS)
    k_inv = tl.rsqrt(tl.sum(k * k, axis=0) / C + EPS)

    tl.store(q_inv_rms_ptr + bn, q_inv)
    tl.store(k_inv_rms_ptr + bn, k_inv)


def sana_wm_fused_qk_inv_rms(
    qkv: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Q/K inverse RMS tables consumed by Sana's fused GDN phases."""
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Expected qkv shape (B, N, 3, H, D), got {qkv.shape}.")
    if not qkv.is_cuda:
        raise ValueError("sana_wm_fused_qk_inv_rms requires a CUDA tensor.")
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    B, N, _, H, D = qkv.shape
    C = H * D
    q_inv_rms = torch.empty((B, N), device=qkv.device, dtype=torch.float32)
    k_inv_rms = torch.empty_like(q_inv_rms)
    block_c = triton.next_power_of_2(C)
    num_warps = 8 if block_c >= 2048 else 4

    with torch.get_device_module().device(qkv.device):
        _sana_wm_fused_qk_inv_rms_kernel[(B * N,)](
            qkv,
            q_inv_rms,
            k_inv_rms,
            C=C,
            EPS=eps,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return q_inv_rms, k_inv_rms


def prepare_sana_wm_rope_tables(
    rotary_emb: Optional[torch.Tensor],
    N: int,
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare SANA-WM complex RoPE frequencies as interleaved cos/sin tables.

    SANA-WM stores RoPE as complex frequencies with shape ``(1, 1, N, D/2)``.
    The Triton preprocessing kernels use the same layout convention as Sana:
    ``y[d] = x[d] * cos[d] + x[d ^ 1] * sin[d]``, where even lanes carry
    ``-sin`` and odd lanes carry ``+sin``.
    """
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)
    if freqs.shape != (N, D // 2):
        raise ValueError(
            f"Expected rotary_emb shape compatible with (N, D/2)={(N, D // 2)}, "
            f"got {tuple(freqs.shape)}."
        )
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack((-sin_half, sin_half), dim=-1).reshape(N, D)
    return rope_cos.contiguous(), rope_sin.contiguous()


@triton.jit
def _sana_wm_qkv_gdn_preprocess_rope_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_rot_out_ptr,
    k_rot_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
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
    d_pair = d + 1 - 2 * (d % 2)
    offs_c_pair = h * D + d_pair

    qkv_base = bn * 3 * C
    q_raw = tl.load(qkv_ptr + qkv_base + offs_c, mask=mask, other=0.0).to(
        tl.float32
    )
    k_raw = tl.load(qkv_ptr + qkv_base + C + offs_c, mask=mask, other=0.0).to(
        tl.float32
    )
    q_pair_raw = tl.load(
        qkv_ptr + qkv_base + offs_c_pair, mask=mask, other=0.0
    ).to(tl.float32)
    k_pair_raw = tl.load(
        qkv_ptr + qkv_base + C + offs_c_pair, mask=mask, other=0.0
    ).to(tl.float32)

    q_rstd = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) / C + EPS)
    k_rstd = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) / C + EPS)

    q_weight = tl.load(q_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    q_pair_weight = tl.load(q_weight_ptr + offs_c_pair, mask=mask, other=0.0).to(
        tl.float32
    )
    k_pair_weight = tl.load(k_weight_ptr + offs_c_pair, mask=mask, other=0.0).to(
        tl.float32
    )

    q = tl.maximum(q_raw * q_rstd * q_weight, 0.0)
    k = tl.maximum(k_raw * k_rstd * k_weight, 0.0) * K_SCALE
    q_pair = tl.maximum(q_pair_raw * q_rstd * q_pair_weight, 0.0)
    k_pair = tl.maximum(k_pair_raw * k_rstd * k_pair_weight, 0.0) * K_SCALE
    v = tl.load(qkv_ptr + qkv_base + 2 * C + offs_c, mask=mask, other=0.0)

    rope_cos = tl.load(rope_cos_ptr + n * D + d, mask=mask, other=1.0).to(
        tl.float32
    )
    rope_sin = tl.load(rope_sin_ptr + n * D + d, mask=mask, other=0.0).to(
        tl.float32
    )
    q_rot = q * rope_cos + q_pair * rope_sin
    k_rot = k * rope_cos + k_pair * rope_sin

    out_base = ((b * H + h) * D + d) * N + n
    tl.store(q_out_ptr + out_base, q, mask=mask)
    tl.store(k_out_ptr + out_base, k, mask=mask)
    tl.store(v_out_ptr + out_base, v, mask=mask)
    tl.store(q_rot_out_ptr + out_base, q_rot, mask=mask)
    tl.store(k_rot_out_ptr + out_base, k_rot, mask=mask)


def sana_wm_qkv_gdn_preprocess_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SANA-WM GDN preprocessing and RoPE application.

    Returns ``(q, k, v, q_rot, k_rot)`` in contiguous ``(B, H, D, N)`` layout.
    This mirrors Sana's inference path where RoPE tables are expanded once and
    consumed inside the fused GDN kernels.
    """
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Expected qkv shape (B, N, 3, H, D), got {qkv.shape}.")
    if not qkv.is_cuda:
        raise ValueError("sana_wm_qkv_gdn_preprocess_rope requires a CUDA tensor.")
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    B, N, _, H, D = qkv.shape
    C = H * D
    if D % 2 != 0:
        raise ValueError(f"SANA-WM RoPE preprocessing requires even D, got D={D}.")
    if q_weight.numel() != C or k_weight.numel() != C:
        raise ValueError(
            "SANA-WM q/k norm weights must match the flattened local channel "
            f"dimension C={C}, got {q_weight.numel()} and {k_weight.numel()}."
        )

    rope_cos, rope_sin = prepare_sana_wm_rope_tables(rotary_emb, N, D, qkv.device)
    q_out = torch.empty((B, H, D, N), device=qkv.device, dtype=qkv.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    q_rot_out = torch.empty_like(q_out)
    k_rot_out = torch.empty_like(q_out)
    block_c = triton.next_power_of_2(C)
    num_warps = 8 if block_c >= 2048 else 4

    with torch.get_device_module().device(qkv.device):
        _sana_wm_qkv_gdn_preprocess_rope_kernel[(B * N,)](
            qkv,
            q_out,
            k_out,
            v_out,
            q_rot_out,
            k_rot_out,
            q_weight.contiguous(),
            k_weight.contiguous(),
            rope_cos,
            rope_sin,
            N=N,
            H=H,
            D=D,
            C=C,
            EPS=eps,
            K_SCALE=k_scale,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return q_out, k_out, v_out, q_rot_out, k_rot_out


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


def can_use_sana_wm_fused_qk_inv_rms(qkv: torch.Tensor) -> bool:
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        return False
    if not qkv.is_cuda or not qkv.is_contiguous():
        return False
    return qkv.dtype in (torch.float16, torch.bfloat16, torch.float32)


def can_use_sana_wm_qkv_gdn_preprocess_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
) -> bool:
    if not can_use_sana_wm_qkv_gdn_preprocess(qkv, q_weight, k_weight):
        return False
    _, N, _, _, D = qkv.shape
    if D % 2 != 0:
        return False
    if rotary_emb is None:
        return True
    if not rotary_emb.is_cuda:
        return False
    return tuple(rotary_emb.squeeze(0).squeeze(0).shape) == (N, D // 2)
