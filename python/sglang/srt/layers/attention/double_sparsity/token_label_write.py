"""Token-label write — per-slot channel projection from projected K_nope.

Each KV slot is stored as a ``[num_heads_local, label_dim]`` fp16 label
row by selecting ``label_dim`` channels from the projected no-PE K.

The ``k_nope`` argument (shape ``[num_tokens, num_heads_local, nope_dim]``)
must already be the projected K_nope (``nope_dim == qk_nope_head_dim`` columns
per head) produced by applying the ``kv_b_proj`` K-side projection at the write
hook site in ``dsa_backend.py``.
This module does NOT perform FP8 dequant or ``kv_b_proj`` projection; it
performs only the channel selection step.

Slot-indexed by ``out_cache_loc`` (physical KV slot indices), matching the
KV cache write path exactly.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def invalidate_token_label_slots(
    written: torch.Tensor,    # bool [L, T]  (table.written)
    layer_id: int,
    cache_loc: torch.Tensor,  # int64/int32 [num_tokens]  (out_cache_loc)
) -> None:
    """Mark token-label slots as invalid before a KV-slot is reused.

    Sets ``written[layer_id, cache_loc] = False`` so that newly-allocated
    physical slots cannot be selected by ``retrieve_topk`` based on stale
    labels left by a previously-freed request.  Must be called from
    ``_select_topk_indices`` *before* ``selector.retrieve_topk``; the
    companion ``token_label_write`` call later in the same forward step
    restores ``written = True`` once fresh labels are available.
    """
    if cache_loc.numel() == 0:
        return
    written[layer_id].index_fill_(0, cache_loc.long(), False)


def token_label_write(
    signatures: torch.Tensor,   # [L, T, H_local, label_dim]  (table.signatures)
    written: torch.Tensor,       # bool [L, T]                 (table.written)
    layer_id: int,
    cache_loc: torch.Tensor,     # int64/int32 [num_tokens]    (out_cache_loc)
    k_nope: torch.Tensor,        # [num_tokens, H_local, nope_dim]  projected K_nope
    channel_selection_layer: torch.Tensor,  # [H_local, label_dim] int32
    scales: Optional[torch.Tensor] = None,  # fp16 [L, T, H_local] (table.scales) for the int8 path
) -> None:
    """Write projected K_nope labels for ``cache_loc`` slots.

    For each token ``t`` writes:
        signatures[layer_id, cache_loc[t], :, :] = k_nope[t, :, channel_selection_layer]

    When ``scales`` is provided (``signatures`` is int8), the gathered fp32
    labels are symmetric-quantized per ``(token, head)`` vector: the scale is
    ``max(|label|) / 127`` and the stored int8 is ``round(label / scale)``.
    Both ``signatures`` and ``scales`` are written in-place. When ``scales`` is
    ``None``, the labels are stored at the signature dtype (fp16) unchanged.

    Mutates ``signatures``/``scales``/``written`` in-place.  No host syncs.
    Thread-safe under the assumption that each slot is owned by exactly one
    request at a time.
    """

    num_tokens = k_nope.shape[0]
    if num_tokens == 0:
        return

    # channel_selection_layer: [H_local, label_dim] int32
    # k_nope: [T, H_local, nope_dim]
    # Gather selected channels → [T, H_local, label_dim]
    H, label_dim = channel_selection_layer.shape
    sel_idx = channel_selection_layer.long().unsqueeze(0).expand(num_tokens, -1, -1)
    labels = torch.gather(k_nope.to(torch.float32), dim=-1, index=sel_idx)  # [T, H, label_dim]

    slot = cache_loc.long()
    if scales is None:
        signatures[layer_id].index_copy_(0, slot, labels.to(signatures.dtype))
    else:
        # Symmetric int8 per (token, head) vector. scale = max(|label|) / 127.
        amax = labels.abs().amax(dim=-1, keepdim=True)  # [T, H, 1] fp32
        scale = amax / 127.0
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quant = (
            torch.round(labels / safe_scale)
            .clamp_(-127, 127)
            .to(torch.int8)
        )  # [T, H, label_dim] int8
        signatures[layer_id].index_copy_(0, slot, quant)
        scales[layer_id].index_copy_(0, slot, scale.squeeze(-1).to(scales.dtype))
    written[layer_id].index_fill_(0, slot, True)
