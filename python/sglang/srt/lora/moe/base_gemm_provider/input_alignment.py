"""Select the single-token or upstream general expert-row alignment."""

from __future__ import annotations

import torch

from sglang.kernels.ops.lora.moe.align_rows import (
    moe_align_single_token,
    pair_to_row_map,
)


def align_rows(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    pair_to_row_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align pairs, optionally filling pair_to_row_out [tokens * topk].

    The single-token path fills the row map in its alignment launch.
    """
    if (
        topk_ids.shape[0] == 1
        and topk_ids.shape[1] <= 32
        and topk_ids.dtype == torch.int32
    ):
        return moe_align_single_token(topk_ids, block_size, pair_to_row_out)
    if pair_to_row_out is not None:
        pair_to_row_map(topk_ids, pair_to_row_out)
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(
        topk_ids, block_size, num_experts, ignore_invalid_expert=True
    )
