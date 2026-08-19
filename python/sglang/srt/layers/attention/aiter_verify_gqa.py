"""AITER target-verify KV layout for GQA (one TP-local KV head).

Triton ``verify_shared_kv_fwd`` (PR #34517) is the grouped-head kernel on the
Triton attention backend. AITER serving does not call that kernel: EAGLE
topk=1 verify goes through ``unified_attention``. The equivalent win is to
pass the true KV-head count so the wrapper sets ``num_queries_per_kv =
tp_q_head_num`` and packs Q heads against one KV load.

The historical stride-0 ``.expand()`` to ``tp_q_head_num`` made the wrapper
see ``num_queries_per_kv = 1`` (fake MHA). On gfx950 that also inflates the
2D program count until ``unified_attention`` drops off the 3D split-KV
kernel onto the 2D kernel at high batch.
"""

from __future__ import annotations

from typing import Tuple

import torch


def pack_unified_verify_kv(
    k_unified: torch.Tensor,
    v_unified: torch.Tensor,
    tp_k_head_num: int,
    tp_q_head_num: int,
    gqa_pack: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return K/V views for AITER ``unified_attention`` target_verify.

    Layout is ``(num_blocks, page_size, num_kv_heads, head_dim)``.

    When ``gqa_pack`` is true (production default), keep ``num_kv_heads`` as
    the real TP-local count. When false, restore the old expand-to-Q-heads
    path for A/B.
    """
    if not gqa_pack and tp_k_head_num == 1 and tp_q_head_num > 1:
        k_unified = k_unified.expand(-1, -1, tp_q_head_num, -1)
        v_unified = v_unified.expand(-1, -1, tp_q_head_num, -1)
    return k_unified, v_unified
