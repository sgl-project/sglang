"""Token-label write — per-slot channel projection from projected K_nope.

Each KV slot is stored as a ``[num_heads_local, label_dim]`` fp16 label
row by selecting ``label_dim`` channels from the 128-d projected nope K.

The ``k_nope`` argument (shape ``[num_tokens, num_heads_local, nope_dim]``)
must already be the projected 128-d K_nope produced by applying the
``kv_b_proj`` K-side projection at the write hook site in ``dsa_backend.py``.
This module does NOT perform FP8 dequant or ``kv_b_proj`` projection; it
performs only the channel selection step.

Slot-indexed by ``out_cache_loc`` (physical KV slot indices), matching the
KV cache write path exactly.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def token_label_write(
    signatures: torch.Tensor,   # [L, T, H_local, label_dim]  (table.signatures)
    written: torch.Tensor,       # bool [L, T]                 (table.written)
    layer_id: int,
    cache_loc: torch.Tensor,     # int64/int32 [num_tokens]    (out_cache_loc)
    k_nope: torch.Tensor,        # [num_tokens, H_local, nope_dim]  projected K_nope
    channel_selection_layer: torch.Tensor,  # [H_local, label_dim] int32
) -> None:
    """Write projected K_nope labels for ``cache_loc`` slots.

    For each token ``t`` writes:
        signatures[layer_id, cache_loc[t], :, :] = k_nope[t, :, channel_selection_layer]

    Mutates ``signatures`` and ``written`` in-place.  Thread-safe under the
    assumption that each slot is owned by exactly one request at a time.
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
    signatures[layer_id].index_copy_(0, slot, labels.to(signatures.dtype))
    written[layer_id].index_fill_(0, slot, True)
