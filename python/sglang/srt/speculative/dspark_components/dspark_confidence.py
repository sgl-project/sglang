from __future__ import annotations

import torch

from sglang.srt.utils.async_probe import maybe_detect_in_closed_range


def build_markov_embed_stack(
    *,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    markov_head,
    gamma: int,
) -> torch.Tensor:
    prev_seq = torch.cat(
        [anchor_tokens.view(-1, 1), draft_tokens[:, : gamma - 1]], dim=1
    )
    return markov_head.get_prev_embeddings(prev_seq)


def compute_confidence(
    *,
    draft_hidden: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    confidence_head,
    markov_head,
    gamma: int,
) -> torch.Tensor:
    assert confidence_head is not None
    if confidence_head.with_markov:
        markov_embed_stack = build_markov_embed_stack(
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            markov_head=markov_head,
            gamma=gamma,
        )
    else:
        markov_embed_stack = None
    confidence_raw = confidence_head(draft_hidden, markov_embed_stack)
    confidence = confidence_head.apply_sts(confidence_raw)
    maybe_detect_in_closed_range(confidence, 0.0, 1.0, "DSpark confidence")
    return confidence
