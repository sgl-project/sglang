"""FlashMLA MegaAttention's chunk-interleaved Q and WO-A layouts."""

from __future__ import annotations

import torch

HEAD_DIM = 512
HEADS_PER_GROUP = 8
Q_CHUNK = 16
O_CHUNK = 32


def permute_mega_wo_a(weight: torch.Tensor) -> torch.Tensor:
    """[G, R, 8 * 512] WO-A -> columns matching the fused kernel's output.

    Each 32-element output chunk interleaves the eight heads of one group.
    The original parameter remains in checkpoint order for reload/fallback.
    """
    groups, rank, hidden = weight.shape
    if hidden != HEADS_PER_GROUP * HEAD_DIM:
        raise ValueError("MegaAttention requires eight 512-dim heads per WO-A group")
    return (
        weight.view(groups, rank, HEADS_PER_GROUP, HEAD_DIM // O_CHUNK, O_CHUNK)
        .transpose(2, 3)
        .contiguous()
        .view(groups, rank, hidden)
    )
