from __future__ import annotations

from typing import Any

import torch


def apply_draft_token_penalties(
    next_token_logits: torch.Tensor,
    candidates: torch.Tensor,
    sampling_info: Any,
) -> None:
    """Apply penalties from earlier draft tokens to later verify rows in place.

    A verify row ``j`` predicts the target token conditioned on draft tokens
    ``0..j-1``. The precomputed ``acc_*_penalties`` tensors only contain the
    committed context, so they must be extended per row with the earlier draft
    tokens. Presence and repetition penalties are set-based; frequency penalties
    count every draft occurrence.
    """
    presence_penalties = getattr(sampling_info, "draft_presence_penalties", None)
    frequency_penalties = getattr(sampling_info, "draft_frequency_penalties", None)
    repetition_penalties = getattr(sampling_info, "draft_repetition_penalties", None)
    if (
        presence_penalties is None
        and frequency_penalties is None
        and repetition_penalties is None
    ):
        return

    bs, draft_token_num = candidates.shape
    if draft_token_num <= 1:
        return

    logits = next_token_logits.view(bs, draft_token_num, -1)
    device = logits.device
    positions = torch.arange(draft_token_num, device=device)

    # duplicate[b, d] is true when an earlier draft position already holds the
    # same token, so presence/repetition penalties are applied once per token.
    same_token = candidates[:, :, None] == candidates[:, None, :]
    earlier_position = positions[None, :] < positions[:, None]
    duplicate = (same_token & earlier_position[None, :, :]).any(dim=2)

    base_presence = getattr(sampling_info, "draft_cumulated_presence_penalties", None)
    base_repetition = getattr(
        sampling_info, "draft_cumulated_repetition_penalties", None
    )

    for draft_pos in range(draft_token_num - 1):
        tokens = candidates[:, draft_pos]
        tail_len = draft_token_num - draft_pos - 1
        row_indices = torch.arange(bs, device=device).unsqueeze(1).expand(bs, tail_len)
        position_indices = (
            torch.arange(draft_pos + 1, draft_token_num, device=device)
            .unsqueeze(0)
            .expand(bs, tail_len)
        )
        token_indices = tokens.unsqueeze(1).expand(bs, tail_len)
        tail = logits[row_indices, position_indices, token_indices]
        tail_shape = tail.shape

        set_mask = (~duplicate[:, draft_pos]).unsqueeze(1).expand(tail_shape)

        if presence_penalties is not None:
            token_is_present = torch.zeros((bs,), dtype=torch.bool, device=device)
            if base_presence is not None:
                token_is_present = (
                    base_presence.gather(1, tokens.unsqueeze(1)).squeeze(1) != 0
                )
            presence_mask = set_mask & ~token_is_present.unsqueeze(1)
            tail = tail - (
                presence_penalties.to(tail.dtype) * presence_mask.to(tail.dtype)
            )

        if frequency_penalties is not None:
            # Frequency counts each prior occurrence, duplicates included.
            tail = tail - frequency_penalties.to(tail.dtype)

        if repetition_penalties is not None:
            token_already_scaled = torch.zeros((bs,), dtype=torch.bool, device=device)
            if base_repetition is not None:
                token_already_scaled = (
                    base_repetition.gather(1, tokens.unsqueeze(1)).squeeze(1) != 1.0
                )
            repetition_mask = set_mask & ~token_already_scaled.unsqueeze(1)
            scale = repetition_penalties.to(tail.dtype)
            scaled = torch.where(tail < 0, tail * scale, tail / scale)
            tail = torch.where(repetition_mask, scaled, tail)

        logits[row_indices, position_indices, token_indices] = tail
