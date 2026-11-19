"""PARD-2 row layout for the context pass and the draft block.

Token t stays at position t. That row's feature is P(t - 1), and position 0
reads a zero feature. The draft block is [bonus, pard_token x (W - 1)] at
positions s .. s + W - 1; the K draft tokens are the argmax of rows 0 .. W - 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def gather_shifted_features(
    p_buf: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """P(t - 1) for each row at position t, and a zero feature at position 0.

    ``req_pool_indices`` has one entry per row, or one per row of a 2-D
    ``positions``. Token t stays at position t: EAGLE's left rotation of the ids
    would drop the position-0 row, which PARD-2 pairs with a zero feature.
    """
    positions = positions.to(torch.int64)
    rows = req_pool_indices.to(torch.int64)
    if rows.dim() < positions.dim():
        rows = rows.unsqueeze(1).expand_as(positions)
    # Column -1 would wrap; position 0 is zeroed below.
    slots = req_to_token[rows, (positions - 1).clamp(min=0)].to(torch.int64)
    features = p_buf[slots]
    features.masked_fill_(positions.unsqueeze(-1).eq(0), 0)
    return features


def fill_draft_block(
    bonus_tokens: torch.Tensor, pard_token: int, seq_lens: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.full(
        (bonus_tokens.shape[0], width),
        pard_token,
        dtype=torch.int64,
        device=bonus_tokens.device,
    )
    token_ids[:, 0] = bonus_tokens
    offsets = torch.arange(width, dtype=torch.int64, device=seq_lens.device)
    positions = seq_lens.to(torch.int64).unsqueeze(1) + offsets
    return token_ids, positions


def repeat_last_feature(
    p_buf: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    width: int,
) -> torch.Tensor:
    # Position s reads P(s - 1). No new target feature exists past the context.
    last = gather_shifted_features(p_buf, req_to_token, req_pool_indices, seq_lens)
    return last.unsqueeze(1).expand(-1, width, -1).contiguous()


def validate_pard2_request(req: Req, enable_overlap: bool) -> Optional[str]:
    if enable_overlap and req.return_hidden_states:
        return "PARD2 speculative decoding does not support return_hidden_states yet."
    params = req.sampling_params
    has_grammar = req.grammar is not None or any(
        constraint is not None
        for constraint in (
            params.json_schema,
            params.regex,
            params.ebnf,
            params.structural_tag,
        )
    )
    if has_grammar:
        return "PARD2 speculative decoding does not support constrained decoding yet."
    return None
