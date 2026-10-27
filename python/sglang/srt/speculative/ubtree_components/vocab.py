# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Vocabulary-parallel helpers for UBTree candidate selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_ubtree_lm_head(lm_head) -> torch.Tensor:
    from sglang.srt.speculative.dflash_utils import is_dense_head_weight

    weight = getattr(lm_head, "weight", None)
    if weight is None:
        raise RuntimeError("UBTREE requires a target lm_head with a weight tensor")
    if not is_dense_head_weight(weight):
        raise ValueError(
            "UBTREE requires a dense FP16/BF16/FP32 target lm_head; "
            f"got {weight.dtype}. Quantized target lm_head is unsupported."
        )
    return weight


def vocab_parallel_topk(
    hidden_states: torch.Tensor,
    lm_head,
    *,
    k: int,
    vocab_size: int,
    tp_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact global Top-K over a replicated or TP-sharded LM head."""

    if hidden_states.ndim != 2:
        raise ValueError("UBTree LM-head input must be rank 2")
    if k <= 0 or k > int(vocab_size):
        raise ValueError(f"invalid UBTree candidate_k={k} for vocab={vocab_size}")
    weight = validate_ubtree_lm_head(lm_head)
    hidden = (
        hidden_states
        if hidden_states.dtype == weight.dtype
        else hidden_states.to(weight.dtype)
    )

    shard = getattr(lm_head, "shard_indices", None)
    if shard is None:
        if int(weight.shape[0]) < int(vocab_size):
            raise ValueError(
                "UBTree target lm_head has fewer rows than the configured vocabulary"
            )
        return F.linear(hidden, weight[:vocab_size]).float().topk(k, dim=-1)

    local_count = int(shard.num_org_elements)
    local_start = int(shard.org_vocab_start_index)
    if local_count > 0:
        local_logits = F.linear(hidden, weight[:local_count]).float()
        local_k = min(k, local_count)
        local_values, local_ids = local_logits.topk(local_k, dim=-1)
        local_ids = local_ids.to(torch.int64) + local_start
    else:
        local_k = 0
        shape = (int(hidden.shape[0]), 0)
        local_values = torch.empty(shape, dtype=torch.float32, device=hidden.device)
        local_ids = torch.empty(shape, dtype=torch.int64, device=hidden.device)

    # Keep an identical collective shape even if the final vocabulary shard is
    # shorter than K because of padding.
    if local_k < k:
        pad = k - local_k
        local_values = torch.cat(
            [
                local_values,
                torch.full(
                    (int(hidden.shape[0]), pad),
                    torch.finfo(torch.float32).min,
                    dtype=torch.float32,
                    device=hidden.device,
                ),
            ],
            dim=-1,
        )
        local_ids = torch.cat(
            [
                local_ids,
                torch.zeros(
                    (int(hidden.shape[0]), pad),
                    dtype=torch.int64,
                    device=hidden.device,
                ),
            ],
            dim=-1,
        )

    if tp_group is None:
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
    tp_size = int(tp_group.world_size)
    if tp_size == 1:
        return local_values, local_ids

    num_tokens = int(hidden.shape[0])
    gathered_values = torch.empty(
        (tp_size * num_tokens * k), dtype=torch.float32, device=hidden.device
    )
    gathered_ids = torch.empty(
        (tp_size * num_tokens * k), dtype=torch.int64, device=hidden.device
    )
    tp_group.all_gather_into_tensor(gathered_values, local_values.contiguous().view(-1))
    tp_group.all_gather_into_tensor(gathered_ids, local_ids.contiguous().view(-1))
    gathered_values = (
        gathered_values.view(tp_size, num_tokens, k)
        .permute(1, 0, 2)
        .reshape(num_tokens, tp_size * k)
    )
    gathered_ids = (
        gathered_ids.view(tp_size, num_tokens, k)
        .permute(1, 0, 2)
        .reshape(num_tokens, tp_size * k)
    )
    values, indices = gathered_values.topk(k, dim=-1)
    return values, gathered_ids.gather(-1, indices)


__all__ = ["vocab_parallel_topk"]
