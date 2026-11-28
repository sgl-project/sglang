"""ROCm gfx950 grouped-head split-KV verify attention for GQA DSpark drafts.

Covers fixed-width causal and non-causal GQA blocks (including non-power-of-two
``kv_group_num`` such as 6), optional sliding windows (``window >= max_len_extend``),
and FP8 KV caches with BF16 dots. Rejects custom masks, sinks, logit soft-cap,
XAI temperature scaling, ragged extend batches, and empty KV rails.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.verify_mla import (
    DEFAULT_BLOCK_H,
    DEFAULT_BLOCK_N,
    DEFAULT_NUM_WARPS,
    _unit_stride_rows,
)
from sglang.kernels.ops.attention.verify_mla_draft import DraftVerifyMLA

# Tuned per head_dim on MI355X (gfx950) under CUDA graphs (kv_len=2048, D=64).
_GQA_DRAFT_BLOCK_CONFIG: dict[int, tuple[int, int, int]] = {
    64: (2, 128, 4),
    128: (2, 128, 4),
}


class _DraftVerifyGQAHip(DraftVerifyMLA):
    """``DraftVerifyMLA`` specialized for multi-KV-head GQA drafts on gfx950."""

    @staticmethod
    def block_config(head_dim: int) -> tuple[int, int, int]:
        return _GQA_DRAFT_BLOCK_CONFIG.get(
            head_dim, (DEFAULT_BLOCK_H, DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS)
        )


_CACHE: dict[tuple, _DraftVerifyGQAHip] = {}


def _select_block_h(kv_group_num: int, preferred_block_h: int) -> int:
    """Largest power-of-two <= preferred_block_h dividing kv_group_num."""
    max_pow2_divisor = kv_group_num & -kv_group_num
    return min(preferred_block_h, max_pow2_divisor)


def _can_handle_gqa_draft_hip(
    q_extend: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    custom_mask: torch.Tensor | None,
    max_len_extend: int,
    *,
    sliding_window_size: int = -1,
    sinks: torch.Tensor | None = None,
    logit_cap: float = 0.0,
    xai_temperature_len: int = -1,
) -> bool:
    """Static shape/dtype gate (zero host syncs, safe inside CUDA graph capture)."""
    if custom_mask is not None or sinks is not None:
        return False
    if (logit_cap and logit_cap > 0) or (
        xai_temperature_len is not None and xai_temperature_len > 0
    ):
        return False
    if (
        q_extend.dim() != 3
        or k_extend.dim() != 3
        or v_extend.dim() != 3
        or k_buffer.dim() != 3
        or v_buffer.dim() != 3
    ):
        return False
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    if h_kv <= 0 or h_q % h_kv != 0:
        return False
    if k_buffer.shape[1] != h_kv or v_buffer.shape[1] != h_kv:
        return False
    if (
        q_extend.shape[2] != k_extend.shape[2]
        or q_extend.shape[2] != k_buffer.shape[2]
        or q_extend.shape[2] != v_extend.shape[2]
        or v_extend.shape[2] != v_buffer.shape[2]
    ):
        return False
    if kv_indices.numel() == 0:
        return False
    bs = qo_indptr.shape[0] - 1
    try:
        mle = int(max_len_extend)
    except (TypeError, ValueError):
        return False
    if bs < 1 or mle < 1 or q_extend.shape[0] != bs * mle:
        return False
    windowed = sliding_window_size is not None and sliding_window_size > 0
    return not (windowed and sliding_window_size < mle)


def _get_runner(
    *,
    max_bs: int,
    h_q: int,
    head_dim: int,
    v_head_dim: int,
    l_ext: int,
    device: torch.device,
    kv_group_num: int,
) -> _DraftVerifyGQAHip:
    key = (h_q, head_dim, v_head_dim, l_ext, str(device), kv_group_num)
    vk = _CACHE.get(key)
    if vk is None:
        pref_bh, block_n, num_warps = _DraftVerifyGQAHip.block_config(head_dim)
        block_h = _select_block_h(kv_group_num, pref_bh)
        vk = _DraftVerifyGQAHip(
            max_bs,
            h_q,
            head_dim,
            v_head_dim,
            l_ext,
            device=device,
            block_h=block_h,
            block_n=block_n,
            num_warps=num_warps,
            kv_group_num=kv_group_num,
        )
        _CACHE[key] = vk
    else:
        vk.grow_buffers(max_bs)
    return vk


def verify_gqa_draft_fwd_hip(
    q_extend: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    o_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    custom_mask: torch.Tensor | None,
    is_causal: bool,
    mask_indptr: torch.Tensor | None,
    max_len_extend: int,
    k_scale: float,
    v_scale: float,
    sm_scale: float | None = None,
    logit_cap: float = 0.0,
    skip_prefix_custom_mask: bool = True,
    sliding_window_size: int = -1,
    sinks: torch.Tensor | None = None,
    window_kv_offsets: torch.Tensor | None = None,
    xai_temperature_len: int = -1,
    max_bs: int | None = None,
    bf16_dot: bool = False,
) -> bool:
    """Grouped-head split-KV verify attention for GQA DSpark drafts on gfx950."""
    del mask_indptr, skip_prefix_custom_mask, window_kv_offsets
    if not _can_handle_gqa_draft_hip(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indices,
        custom_mask,
        max_len_extend,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
    ):
        return False

    bs = qo_indptr.shape[0] - 1
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    l_ext = int(max_len_extend)
    kv_group_num = h_q // h_kv
    windowed = sliding_window_size is not None and sliding_window_size > 0

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    if not isinstance(k_scale, (int, float)) or not isinstance(v_scale, (int, float)):
        return False
    k_scale_f = float(k_scale)
    v_scale_f = float(v_scale)

    cap_bs = bs if (max_bs is None or max_bs < bs) else max_bs
    vk = _get_runner(
        max_bs=cap_bs,
        h_q=h_q,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        l_ext=l_ext,
        device=q_extend.device,
        kv_group_num=kv_group_num,
    )
    vk(
        q_extend,
        _unit_stride_rows(k_extend),
        _unit_stride_rows(v_extend),
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=o_extend,
        k_scale=k_scale_f,
        v_scale=v_scale_f,
        is_causal=is_causal,
        bf16_dot=bf16_dot,
        window=int(sliding_window_size) if windowed else 0,
    )
    return True
