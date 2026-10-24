# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""UBTree proposals, executed eagerly or inside the full draft CUDA Graph."""

from __future__ import annotations

import torch

from sglang.srt.speculative.ubtree_components.config import UBTreeConfig
from sglang.srt.speculative.ubtree_components.tree import (
    UBTreeBatch,
    build_ubtree_batch,
)
from sglang.srt.speculative.ubtree_components.vocab import vocab_parallel_topk


def build_ubtree_proposal(
    *,
    hidden: torch.Tensor,
    bonus_tokens: torch.Tensor,
    config: UBTreeConfig,
    num_nodes: int,
    lm_head,
    selector,
    tp_group=None,
) -> UBTreeBatch:
    batch_size = hidden.shape[0]
    values, candidates = vocab_parallel_topk(
        hidden.reshape(-1, hidden.shape[-1]),
        lm_head,
        k=config.candidate_k,
        vocab_size=selector.vocab_size,
        tp_group=tp_group,
    )
    shape = (batch_size, config.depth_count, config.candidate_k)
    candidates = candidates.view(shape)
    edges = selector.build_edge_log_probs(
        hidden,
        bonus_tokens,
        candidates,
        unary_candidate_logits=values.view(shape),
    )
    return build_ubtree_batch(
        anchor_tokens=bonus_tokens,
        candidate_ids=candidates,
        edge_log_probs=edges,
        budget=num_nodes - 1,
        supertree_width=config.supertree_width,
        depth_bonus=config.depth_bonus,
    )
