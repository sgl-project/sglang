"""The eager LiLiCorr draft seam.

The published recipe folds the head into the draft CUDA graph (see
``lilicorr_draft_sampler``). This path serves the steps that graph cannot:
prefill and extend, batches past the captured buckets, and tp>1, which the folded
sampler declines.

It runs the same head method as the folded path, so the two cannot drift in what
they score. They differ in three ways, all forced by capture: this path chunks the
candidate GEMM instead of writing into a static buffer, it gathers raw embeddings
instead of a precomputed projected table, and it reads the anchor from the worker
attribute instead of a fixed-address buffer.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.speculative.lilicorr_components.lilicorr_candidates import (
    lilicorr_candidates,
    resolve_vocab_shard,
)


def propose_lilicorr_block(
    *,
    head,
    draft_hidden: torch.Tensor,
    lm_head,
    embed_tokens,
    anchor: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reranked draft block in place of the per-slot greedy argmax.

    ``draft_hidden`` is ``[bs, block_size, hidden]``, where slot 0 is the anchor
    position and slots 1.. are the candidate positions. Returns the selected token
    ids ``[bs, block_size - 1]``.

    ``anchor`` is the projected per-request last committed target row, or None. A
    row count that disagrees with the batch means the batch changed size between
    the context append that wrote it and this read; the whole batch then scores
    against an invalid anchor rather than against another request's row.
    """
    bs, block_size, hidden_size = draft_hidden.shape
    slots = block_size - 1
    pass_hidden = draft_hidden[:, 1:, :]
    tp_group = get_tp_group()
    num_org, org_vocab_start = resolve_vocab_shard(lm_head)

    log_probs, ids = lilicorr_candidates(
        hidden_states=pass_hidden.reshape(bs * slots, hidden_size),
        weight=lm_head.weight,
        num_org=num_org,
        org_vocab_start=org_vocab_start,
        topk=int(head.candidate_topk),
        tp_group=tp_group if int(tp_group.world_size) > 1 else None,
    )
    ids = ids.view(bs, slots, int(head.candidate_topk))

    feat = int(head.context_proj.in_features)
    if anchor is None or int(anchor.shape[0]) != bs:
        anchor_hidden = torch.zeros(
            (bs, feat), device=draft_hidden.device, dtype=draft_hidden.dtype
        )
        anchor_valid = torch.zeros((bs,), dtype=torch.bool, device=draft_hidden.device)
    else:
        anchor_hidden = anchor.view(bs, feat)
        anchor_valid = torch.ones((bs,), dtype=torch.bool, device=draft_hidden.device)

    # The embedding lookup happens here, outside the head, because on the target
    # model it may be a TP-sharded collective.
    selected = head.select(
        token_embeddings=embed_tokens(ids).detach(),
        candidate_token_ids=ids,
        candidate_log_probs=log_probs.view(bs, slots, int(head.candidate_topk)),
        pass_hidden=pass_hidden,
        anchor_hidden=anchor_hidden,
        anchor_valid=anchor_valid,
    )
    return selected.to(torch.long)
