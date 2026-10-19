"""Greedy (topk=1) MTP drafting over a fixed subset of the vocabulary.

The draft scores only the listed rows of the shared bf16 lm_head, split evenly
across TP ranks; logits stay in subset order and every proposal is mapped back to
its vocabulary id where it is selected, so verify, the next draft step and a peer
drafting over the full vocabulary all see vocabulary ids."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.runtime_context import get_parallel


def build_hot_head_shard(
    *,
    weight: torch.Tensor,
    org_vocab_start: int,
    org_vocab_end: int,
    hot_ids: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """This rank's contiguous slice of the hot rows of a vocab-parallel weight."""
    rows = hot_ids.numel() // tp_size
    own = None
    for dst in range(tp_size):
        ids = hot_ids[dst * rows : (dst + 1) * rows].to(weight.device)
        buf = torch.zeros(
            (rows, weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        pos = ((ids >= org_vocab_start) & (ids < org_vocab_end)).nonzero().squeeze(1)
        buf[pos] = weight[ids[pos] - org_vocab_start]
        if tp_size > 1:
            # One owner per row and integer zeros elsewhere: the sum keeps the bits.
            dist.all_reduce(buf.view(torch.int32), group=group)
        if dst == tp_rank:
            own = buf
        else:
            del buf
    return own


class HotVocabHead(nn.Module):
    """Plain bf16 head over this rank's hot rows (no quant method, no padding)."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


class DraftHotVocab:
    """Owns the hot ids (device int64, sorted) and the draft's hot head."""

    def __init__(self, model, hot_ids: torch.Tensor) -> None:
        parallel = get_parallel()
        head = model.lm_head
        shard = head.shard_indices
        weight = build_hot_head_shard(
            weight=head.weight.data,
            org_vocab_start=shard.org_vocab_start_index,
            org_vocab_end=shard.org_vocab_end_index,
            hot_ids=hot_ids,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
            group=parallel.tp_group.device_group if parallel.tp_size > 1 else None,
        )
        self.vocab_size = model.config.vocab_size
        self.token_ids = hot_ids.to(weight.device)
        self.head = HotVocabHead(weight)
        # The target keeps its own module; only the draft's attribute is rebound.
        model.lm_head = self.head
