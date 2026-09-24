"""Block-selection math shared by the backends whose candidates are picked from
dense scores: a candidate source keeps whole blocks of ``block_size`` compressed
positions rather than single positions, and its newest block always."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F


def _candidate_block_topk(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.return_types.topk:
    width = logits.size(-1)
    padding = -width % block_size
    scores = F.pad(logits, (0, padding), value=-torch.inf) if padding else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    return scores.topk(min(topk_blocks, num_blocks), dim=-1)


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    top = _candidate_block_topk(
        logits=logits,
        compress_lens=compress_lens,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)


def candidate_block_mask(
    blocks: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*blocks.shape[:-1], num_blocks + 1), dtype=torch.bool, device=blocks.device
    )
    keep.scatter_(-1, blocks.to(torch.int64).masked_fill(blocks < 0, num_blocks), True)
    return keep[..., :num_blocks].repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    top = _candidate_block_topk(
        logits=logits,
        compress_lens=compress_lens,
        topk_blocks=topk_blocks,
        block_size=block_size,
    )
    width = logits.shape[-1]
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*logits.shape[:-1], num_blocks), dtype=torch.bool, device=logits.device
    ).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
