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


def _aligned_rows(
    req_pool_indices: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    if req_pool_indices.shape == positions.shape:
        return req_pool_indices.to(dtype=torch.int64)
    if (
        positions.dim() == 2
        and req_pool_indices.dim() == 1
        and req_pool_indices.shape[0] == positions.shape[0]
    ):
        return req_pool_indices.to(dtype=torch.int64).unsqueeze(1).expand_as(positions)
    raise RuntimeError(
        "PARD2 req_pool_indices must have the same shape as positions, or one "
        f"index per batch row; got indices {tuple(req_pool_indices.shape)} "
        f"and positions {tuple(positions.shape)}."
    )


def _shifted_features(
    p_buf: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if p_buf.dim() != 2:
        raise RuntimeError(
            f"PARD2 p_buf must be [num_slots, hidden], got {tuple(p_buf.shape)}."
        )
    if req_to_token.dim() != 2:
        raise RuntimeError(
            "PARD2 req_to_token must be [req_pool, max_len], "
            f"got {tuple(req_to_token.shape)}."
        )
    positions_i = positions.to(dtype=torch.int64)
    rows = _aligned_rows(req_pool_indices, positions_i)
    # Column -1 would wrap. Position 0 is overwritten below.
    prev = (positions_i - 1).clamp(min=0)
    slots = req_to_token[rows, prev].to(dtype=torch.int64)
    features = p_buf[slots]
    # Position 0 pairs token 0 with a zero feature, not P(0) and not a wrap.
    features.masked_fill_(positions_i.unsqueeze(-1).eq(0), 0)
    return features


def gather_shifted_features(
    p_buf: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.shape != positions.shape:
        raise RuntimeError(
            "PARD2 input_ids and positions must share a shape, "
            f"got {tuple(input_ids.shape)} and {tuple(positions.shape)}."
        )
    features = _shifted_features(p_buf, req_to_token, req_pool_indices, positions)
    # Token t stays at position t. EAGLE rotates ids left; that drops this row.
    return input_ids, features


def fill_draft_block(
    bonus_tokens: torch.Tensor,
    pard_token: int,
    seq_lens: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bonus_tokens.dim() != 1 or seq_lens.shape != bonus_tokens.shape:
        raise RuntimeError(
            "PARD2 bonus_tokens and seq_lens must be 1D and the same length, "
            f"got {tuple(bonus_tokens.shape)} and {tuple(seq_lens.shape)}."
        )
    block_width = int(width)
    if block_width < 2:
        raise RuntimeError(
            "PARD2 draft block width must be >= 2 (verify width, K + 1); "
            f"got {block_width}."
        )
    bonus = bonus_tokens.to(dtype=torch.int64)
    seq = seq_lens.to(dtype=torch.int64)
    token_ids = torch.full(
        (bonus.shape[0], block_width),
        int(pard_token),
        dtype=torch.int64,
        device=bonus.device,
    )
    token_ids[:, 0] = bonus
    offsets = torch.arange(block_width, dtype=torch.int64, device=seq.device)
    positions = seq.unsqueeze(1) + offsets
    return token_ids, positions


def repeat_last_feature(
    p_buf: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    width: int,
) -> torch.Tensor:
    if seq_lens.dim() != 1:
        raise RuntimeError(
            f"PARD2 seq_lens must be 1D committed lengths, got {tuple(seq_lens.shape)}."
        )
    block_width = int(width)
    if block_width < 1:
        raise RuntimeError(f"PARD2 repeat width must be >= 1; got {block_width}.")
    # Position s reads P(s - 1). No new target feature exists past the context.
    last = _shifted_features(p_buf, req_to_token, req_pool_indices, seq_lens)
    return last.unsqueeze(1).expand(-1, block_width, -1).contiguous()


def draft_rows_to_tokens(row_logits: torch.Tensor) -> torch.Tensor:
    if row_logits.dim() != 3:
        raise RuntimeError(
            "PARD2 draft logits must have shape [batch, width, vocab], "
            f"got {tuple(row_logits.shape)}."
        )
    if row_logits.shape[1] < 2:
        raise RuntimeError(
            "PARD2 draft block needs at least 2 rows so row W - 1 can be the "
            f"unused tail; got width {int(row_logits.shape[1])}."
        )
    # Row W - 1 is unused. K drafts are the argmax of rows 0 .. W - 2.
    return row_logits[:, :-1, :].argmax(dim=-1)


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
