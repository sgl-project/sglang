# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.diffusion.sana_wm.fused_gdn_chunkwise import (
    fused_bigdn_bidi_chunkwise,
)
from sglang.jit_kernel.diffusion.sana_wm.qkv_preprocess import (
    can_use_sana_wm_fused_qk_inv_rms,
    prepare_sana_wm_rope_tables,
    sana_wm_fused_qk_inv_rms,
)


def sana_wm_fused_bigdn_bidi(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
    k_scale: float,
    eps: float,
    norm_eps: float,
    dot_precision: int = 0,
) -> torch.Tensor:
    """Run the upstream Sana-WM fused bidirectional GDN pipeline.

    This mirrors ``Sana/diffusion/model/ops/fused_gdn.fused_bigdn_func`` while
    keeping SGLang-owned imports and guards. The returned layout is
    ``(B, N, H, D)``.
    """
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    _, N, _, _, D = qkv.shape
    q_inv_rms, k_inv_rms = sana_wm_fused_qk_inv_rms(qkv, eps=norm_eps)
    return sana_wm_fused_bigdn_bidi_with_inv_rms(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
    )


def sana_wm_fused_bigdn_bidi_with_inv_rms(
    qkv: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
    k_scale: float,
    eps: float,
    norm_eps: float,
    dot_precision: int = 0,
) -> torch.Tensor:
    """Run fused bidirectional GDN with caller-provided Q/K inv-RMS.

    SANA-WM TP keeps Q/K heads sharded but normalizes over the full hidden
    dimension. The runtime computes the cross-rank inv-RMS and passes the local
    norm-weight shard through this entry point.
    """
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    _, N, _, _, D = qkv.shape
    rope_cos, rope_sin = prepare_sana_wm_rope_tables(rotary_emb, N, D, qkv.device)
    return fused_bigdn_bidi_chunkwise(
        qkv,
        q_inv_rms.contiguous(),
        k_inv_rms.contiguous(),
        q_weight.float().contiguous(),
        k_weight.float().contiguous(),
        rope_cos,
        rope_sin,
        beta.contiguous(),
        decay.contiguous(),
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
    )


def can_use_sana_wm_fused_bigdn_bidi_with_inv_rms(
    qkv: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
) -> bool:
    if not can_use_sana_wm_fused_qk_inv_rms(qkv):
        return False
    B, N, _, H, D = qkv.shape
    if N != F * S or F <= 0 or S <= 0:
        return False
    if tuple(q_inv_rms.shape) != (B, N) or tuple(k_inv_rms.shape) != (B, N):
        return False
    if not q_inv_rms.is_cuda or not k_inv_rms.is_cuda:
        return False
    if q_weight.numel() != H * D or k_weight.numel() != H * D:
        return False
    if D % 2 != 0:
        return False
    if beta.ndim != 4 or decay.ndim != 3:
        return False
    if tuple(beta.shape) != (B, H, F, S) or tuple(decay.shape) != (B, H, F):
        return False
    if not beta.is_cuda or not decay.is_cuda:
        return False
    if rotary_emb is None:
        return True
    if not rotary_emb.is_cuda:
        return False
    return tuple(rotary_emb.squeeze(0).squeeze(0).shape) == (N, D // 2)


def can_use_sana_wm_fused_bigdn_bidi(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
) -> bool:
    if not can_use_sana_wm_fused_qk_inv_rms(qkv):
        return False
    B, N, _, H, D = qkv.shape
    if N != F * S or F <= 0 or S <= 0:
        return False
    if q_weight.numel() != H * D or k_weight.numel() != H * D:
        return False
    if D % 2 != 0:
        return False
    if beta.ndim != 4 or decay.ndim != 3:
        return False
    if tuple(beta.shape) != (B, H, F, S) or tuple(decay.shape) != (B, H, F):
        return False
    if not beta.is_cuda or not decay.is_cuda:
        return False
    if rotary_emb is None:
        return True
    if not rotary_emb.is_cuda:
        return False
    return tuple(rotary_emb.squeeze(0).squeeze(0).shape) == (N, D // 2)
