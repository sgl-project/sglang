# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def effective_token_length(token_masks: torch.Tensor) -> int:
    """Return the last visible token position across the batch."""

    if token_masks.ndim != 2:
        raise ValueError(
            f"Pi0.5 token masks must be [batch, seq], got {token_masks.shape}"
        )
    if token_masks.shape[1] == 0:
        return 0

    positions = torch.arange(
        1,
        token_masks.shape[1] + 1,
        device=token_masks.device,
        dtype=torch.long,
    )
    lengths = torch.where(token_masks.to(torch.bool), positions, 0).amax(dim=1)
    return int(lengths.max().item())


def select_prompt_token_bucket(
    token_length: int,
    buckets: Sequence[int],
) -> int | None:
    """Select the smallest configured bucket containing ``token_length``."""

    if token_length < 0:
        raise ValueError("token_length must be non-negative")
    return next((int(bucket) for bucket in buckets if token_length <= bucket), None)


def bucket_prompt_tokens(
    tokens: torch.Tensor,
    token_masks: torch.Tensor,
    buckets: Sequence[int],
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int, int | None]:
    """Trim or right-pad prompt tensors to a stable CUDA graph bucket."""

    if tokens.ndim != 2:
        raise ValueError(f"Pi0.5 tokens must be [batch, seq], got {tokens.shape}")
    if token_masks.shape != tokens.shape:
        raise ValueError(
            "Pi0.5 tokens and token masks must have identical shapes, got "
            f"{tokens.shape} and {token_masks.shape}"
        )

    logical_length = effective_token_length(token_masks)
    bucket = select_prompt_token_bucket(logical_length, buckets)
    target_length = bucket if bucket is not None else logical_length

    # Preserve the existing empty-prompt fallback when no bucket is selected.
    if target_length == 0 and bucket is None:
        target_length = tokens.shape[1]

    if tokens.shape[1] >= target_length:
        return (
            tokens[:, :target_length],
            token_masks[:, :target_length],
            logical_length,
            bucket,
        )

    padding = target_length - tokens.shape[1]
    return (
        F.pad(tokens, (0, padding), value=pad_token_id),
        F.pad(token_masks, (0, padding), value=False),
        logical_length,
        bucket,
    )
