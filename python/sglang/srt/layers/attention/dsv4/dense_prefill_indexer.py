from __future__ import annotations

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillCandidateBlocks,
    candidate_block_mask,
    mask_topk_scores,
    select_candidate_block_ids,
)

_SCORE_BUDGET_BYTES = 2 << 30


def dense_prefill_topk(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    request_lengths: list[tuple[int, int]],
    topk: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    publish_candidates: bool,
    candidates: PrefillCandidateBlocks | None,
) -> tuple[torch.Tensor, PrefillCandidateBlocks | None]:
    selected = torch.full(
        (q[0].shape[0], topk), -1, dtype=torch.int32, device=weights.device
    )
    published = (
        PrefillCandidateBlocks(request_blocks=[]) if publish_candidates else None
    )
    request_ranges = []
    row = 0
    for query_length, context_length in request_lengths:
        request_ranges.append((row, row + query_length, context_length))
        if published is not None:
            num_blocks = (
                context_length + candidate_block_size - 1
            ) // candidate_block_size
            published.request_blocks.append(
                torch.empty(
                    (query_length, min(candidate_topk_blocks, num_blocks)),
                    dtype=torch.int32,
                    device=weights.device,
                )
            )
        row += query_length
    width = (max((n for _, n in request_lengths), default=0) + 3) // 4 * 4
    step = max(1, _SCORE_BUDGET_BYTES // max(4 * width, 1))
    for offset in range(0, row if width else 0, step):
        rows = slice(offset, min(offset + step, row))
        _select_tile(
            q=(q[0][rows], q[1][rows]),
            kv=kv,
            weights=weights[rows],
            starts=starts[rows],
            lengths=lengths[rows],
            width=width,
            selected=selected[rows],
            block_size=candidate_block_size,
            row_offset=offset,
            request_ranges=request_ranges,
            publish=published,
            consume=candidates,
        )
    return selected, published


def _select_tile(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
    selected: torch.Tensor,
    block_size: int,
    row_offset: int,
    request_ranges: list[tuple[int, int, int]],
    publish: PrefillCandidateBlocks | None,
    consume: PrefillCandidateBlocks | None,
) -> None:
    from deep_gemm import fp8_fp4_mqa_logits

    from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2

    logits = fp8_fp4_mqa_logits(q, kv, weights, starts, starts + lengths, False, width)
    if publish is not None or consume is not None:
        for request, (start, end, context_length) in enumerate(request_ranges):
            begin, stop = max(start, row_offset), min(end, row_offset + logits.shape[0])
            if begin >= stop or context_length == 0:
                continue
            rows = slice(begin - row_offset, stop - row_offset)
            request_rows = slice(begin - start, stop - start)
            scores = logits[rows, :context_length]
            if publish is not None:
                lens = lengths[rows, None]
                scores.masked_fill_(
                    torch.arange(context_length, device=logits.device)[None, :] >= lens,
                    -torch.inf,
                )
                blocks = publish.request_blocks[request][request_rows]
                blocks.copy_(
                    select_candidate_block_ids(
                        logits=scores,
                        compress_lens=lens,
                        topk_blocks=blocks.shape[1],
                        block_size=block_size,
                    )
                )
            else:
                scores.masked_fill_(
                    ~candidate_block_mask(
                        blocks=consume.request_blocks[request][request_rows],
                        width=context_length,
                        block_size=block_size,
                    ),
                    -torch.inf,
                )
    topk_transform_ragged_v2(logits, lengths, out_offsets=starts, out_indices=selected)
    if consume is not None:
        selected.copy_(
            mask_topk_scores(scores=logits, indices=selected, offsets=starts)
        )
