# SPDX-License-Identifier: Apache-2.0
"""Phase 2: SageAttention-3 Blackwell FP4 self-attention via sgl-kernel.

Pure-Python port of the old vendored ``sage3_attention.cu``::
``run_sage3_fmha_packed_qkv`` glue — bf16 q/k/v in, bf16 out — calling the
generic sgl-kernel ops ``sgl_kernel.sage3.sage3_mha_fwd`` and
``scaled_fp4_quant``. This avoids the standalone ``sageattn3_blackwell`` pip
package (not on any PyPI mirror) by using the FP4 kernel already shipped in
sgl-kernel (sm_120a build).

Algorithm (mirrors ``run_sage3_fmha_packed_qkv``):
  1. Pad Mq/Mk up to a multiple of 128.
  2. Center K per-head (subtract mean over Mk); center Q per 128-token block
     (``subtract_group_mean``), retaining the per-block mean ``qm``.
  3. ``delta_s = qm @ k_padded.T`` (float32) — the FP4 per-block scale-correction
     matrix SageAttention-3 applies inside its FP4 attention.
  4. FP4-quantize q (plain), k (permute), v (trans) via ``scaled_fp4_quant``.
  5. ``sage3_mha_fwd(q_fp4, k_fp4, v_fp4, sfq, sfk, sfv, delta_s, unpadded_k=Mk,
     softmax_scale=1/sqrt(D), is_causal=False, per_block_mean=True, is_bf16=True)``.

OmniDreams self-attention is bidirectional over the AR window (no mask), so
``is_causal=False`` matches the existing ``F.sdpa`` call. head_dim must be 64 or
128 (OmniDreams uses 128). Falls back to the caller's sdpa path on any error.
"""

from __future__ import annotations

import math

import torch

_SAGE3_BLOCK = 128
_SAGE3_HEAD_DIMS = (64, 128)

try:
    from sgl_kernel.sage3 import sage3_mha_fwd, scaled_fp4_quant

    _SAGE3_AVAILABLE = True
except Exception:
    _SAGE3_AVAILABLE = False


def _pad_to_block(x: torch.Tensor, block: int = _SAGE3_BLOCK) -> torch.Tensor:
    """Pad dim 2 of ``[B, H, M, D]`` up to a multiple of ``block`` (zero pad)."""
    m = x.size(2)
    m_round = ((m + block - 1) // block) * block
    if m == m_round:
        return x.contiguous()
    out = torch.zeros(
        x.size(0), x.size(1), m_round, x.size(3), dtype=x.dtype, device=x.device
    )
    out[:, :, :m].copy_(x)
    return out


def sage3_self_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """SageAttention-3 FP4 self-attention (sgl-kernel, Blackwell sm_120a).

    Args:
        q, k, v: ``[B, H, M, D]`` bf16, post-RoPE / post-cache-assemble. D in
            {64, 128}. Mq and Mk may differ (Q attends over the cached K window).
        scale: softmax scale; defaults to ``1/sqrt(D)`` (matches ``F.sdpa``).
        is_causal: False for OmniDreams bidirectional self-attn.

    Returns: ``[B, H, Mq, D]`` bf16.
    """
    if not _SAGE3_AVAILABLE:
        raise RuntimeError("sgl_kernel.sage3 ops not available")
    B, H, Mq, D = q.shape
    Mk = k.size(2)
    if D not in _SAGE3_HEAD_DIMS:
        raise ValueError(f"sage3 head_dim must be 64 or 128, got {D}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("sage3_self_attn requires bfloat16 q/k/v")
    softmax_scale = scale if scale and scale > 0 else 1.0 / math.sqrt(D)

    q_padded = _pad_to_block(q)
    k_centered = k - k.mean(dim=2, keepdim=True)
    k_padded = _pad_to_block(k_centered)
    v_padded = _pad_to_block(v)

    QL = q_padded.size(2)
    KL = k_padded.size(2)
    groups = QL // _SAGE3_BLOCK  # QL is a multiple of 128
    # subtract_group_mean (per_block_mean=True): one mean per 128-token block.
    q_blocks = q_padded.view(B, H, groups, _SAGE3_BLOCK, D)
    qm = q_blocks.mean(dim=3)  # [B, H, groups, D]
    q_centered = q_padded - qm.unsqueeze(3).expand(
        B, H, groups, _SAGE3_BLOCK, D
    ).reshape(B, H, QL, D)

    # delta_s: per-block FP4 scale-correction matrix, float32 [B, H, groups, KL].
    delta_s = (qm @ k_padded.transpose(-2, -1)).to(torch.float32).contiguous()

    u8, f8 = torch.uint8, torch.float8_e4m3fn
    dev = q.device
    q_fp4 = torch.empty(B, H, QL, D // 2, dtype=u8, device=dev)
    q_sf = torch.empty(B, H, QL, D // 16, dtype=f8, device=dev)
    k_fp4 = torch.empty(B, H, KL, D // 2, dtype=u8, device=dev)
    k_sf = torch.empty(B, H, KL, D // 16, dtype=f8, device=dev)
    v_fp4 = torch.empty(B, H, D, KL // 2, dtype=u8, device=dev)
    v_sf = torch.empty(B, H, D, KL // 16, dtype=f8, device=dev)
    scaled_fp4_quant(q_centered, q_fp4, q_sf, 1, 0)  # plain
    scaled_fp4_quant(k_padded.contiguous(), k_fp4, k_sf, 1, 1)  # permute (K)
    scaled_fp4_quant(v_padded.contiguous(), v_fp4, v_sf, 1, 2)  # trans (V)

    out = sage3_mha_fwd(
        q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf, delta_s, Mk, None,
        softmax_scale, is_causal, True, True,
    )[0]  # [B, H, QL, D]
    return out[:, :, :Mq].contiguous()
