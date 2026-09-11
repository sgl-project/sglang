"""The eager LiLiCorr draft seam.

The published recipe folds the head into the draft CUDA graph (see
``lilicorr_draft_sampler``). This path serves the steps that graph cannot:
prefill and extend, batches past the captured buckets, and tp>1, which the
folded sampler declines.

It runs the same head method as the folded path, so the two cannot drift in what
they score. They differ only in what capture forces: this path chunks the
candidate GEMM instead of writing into a static buffer, gathers raw embeddings
instead of a precomputed projected table, and takes the anchor as an argument
instead of from a fixed-address buffer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask
from sglang.srt.speculative.lilicorr_components.lilicorr_candidates import (
    lilicorr_candidates,
    resolve_vocab_shard,
)
from sglang.srt.speculative.lilicorr_components.lilicorr_config import SAMPLING_ENABLED


def propose_lilicorr_block(
    *,
    head,
    draft_hidden: torch.Tensor,
    lm_head,
    embed_tokens,
    anchor: Optional[torch.Tensor],
    sampling_info=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Reranked draft block in place of the per-slot greedy argmax.

    ``draft_hidden`` is ``[bs, block_size, hidden]``, where slot 0 is the anchor
    position and slots 1.. are the candidate positions. Returns ``(tokens [bs,
    block_size - 1], candidate_tokens, q_rows)``; the latter two are ``None`` unless
    ``LILICORR_SAMPLING`` is on, and the worker publishes them for the verify.

    Unlike the folded path there are no static buffers to stage into, so the sampling
    tensors are built per call -- which is also why this path exists: it serves the
    steps the graph cannot.
    """
    bs, block_size, hidden_size = draft_hidden.shape
    slots = block_size - 1
    pass_hidden = draft_hidden[:, 1:, :]
    tp_group = get_tp_group()
    num_org, org_vocab_start = resolve_vocab_shard(lm_head)

    log_probs, candidate_tokens = lilicorr_candidates(
        hidden_states=pass_hidden.reshape(bs * slots, hidden_size),
        weight=lm_head.weight,
        num_org=num_org,
        org_vocab_start=org_vocab_start,
        topk=int(head.candidate_topk),
        tp_group=tp_group if int(tp_group.world_size) > 1 else None,
    )
    candidate_tokens = candidate_tokens.view(bs, slots, int(head.candidate_topk))

    feat = int(head.context_proj.in_features)
    if anchor is None or int(anchor.shape[0]) != bs:
        # A row count that disagrees with the batch means the batch changed size
        # between the context append that wrote the anchor and this read, so the
        # whole batch scores as invalid rather than against another request's row.
        anchor_hidden = torch.zeros(
            (bs, feat), device=draft_hidden.device, dtype=draft_hidden.dtype
        )
        anchor_valid = torch.zeros((bs,), dtype=torch.bool, device=draft_hidden.device)
    else:
        anchor_hidden = anchor.view(bs, feat)
        anchor_valid = torch.ones((bs,), dtype=torch.bool, device=draft_hidden.device)

    # The embedding lookup happens here, outside the head, because on the target
    # model it may be a TP-sharded collective.
    common = dict(
        token_embeddings=embed_tokens(candidate_tokens).detach(),
        candidate_tokens=candidate_tokens,
        candidate_log_probs=log_probs.view(bs, slots, int(head.candidate_topk)),
        pass_hidden=pass_hidden,
        anchor_hidden=anchor_hidden,
        anchor_valid=anchor_valid,
    )
    if not SAMPLING_ENABLED:
        return head.select(**common).to(torch.long), None, None

    device = draft_hidden.device
    temperatures = (
        torch.ones(bs, dtype=torch.float32, device=device)
        if sampling_info is None
        # Clamped like DSpark and the DFlash2 selector so a greedy row, whose
        # temperature may be exactly 0, cannot divide by zero before greedy_mask
        # discards its sampled pick anyway.
        else sampling_info.temperatures.view(-1)[:bs].float().clamp_min(1e-5)
    )
    selected, q_rows = head.select_with_proposal(
        uniforms=torch.rand(bs, slots, dtype=torch.float32, device=device),
        temperatures=temperatures,
        greedy_mask=resolve_greedy_mask(
            bs=bs, sampling_info=sampling_info, device=device
        ),
        **common,
    )
    return selected.to(torch.long), candidate_tokens, q_rows
