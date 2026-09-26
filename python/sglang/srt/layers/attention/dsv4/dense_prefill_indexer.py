from __future__ import annotations

from typing import List, Optional

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    candidate_block_mask,
    mask_topk_scores,
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.index_logits import (
    flat_index_logits_rows_per_tile,
)
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateIndexer,
    CandidateMetadata,
    PrefillCandidateBlocks,
    PrefillIndexerInputs,
)
from sglang.srt.utils.common import ceil_align

# TODO: use a per-forward mqa_logits_budget_bytes() budget that also
# leaves room for candidate masks and block-selection scratch.
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
    width = ceil_align(max((n for _, n in request_lengths), default=0), 4)
    if row == 0 or width == 0:
        return selected, published
    rows_per_chunk = flat_index_logits_rows_per_tile(
        row, width, heads=q[0].shape[1], budget_bytes=_SCORE_BUDGET_BYTES
    )
    for offset in range(0, row, rows_per_chunk):
        rows = slice(offset, min(offset + rows_per_chunk, row))
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


def _dense_topk(
    inputs: PrefillIndexerInputs,
    out_positions: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    publish: bool,
    candidates: Optional[PrefillCandidateBlocks],
) -> Optional[PrefillCandidateBlocks]:
    selected, published = dense_prefill_topk(
        q=(inputs.q_fp4, inputs.q_sf),
        kv=inputs.kv,
        weights=inputs.weights,
        starts=inputs.request_starts,
        lengths=inputs.compress_lens,
        request_lengths=list(zip(inputs.rows_per_request, inputs.lens_per_request)),
        topk=out_positions.shape[1],
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=block_size,
        publish_candidates=publish,
        candidates=candidates,
    )
    out_positions.copy_(selected)
    return published


def plain_prefill_topk(
    inputs: PrefillIndexerInputs, out_positions: torch.Tensor
) -> None:
    """The top-k of an index layer outside the candidate scheme, over its dense
    scores tile by tile."""
    _dense_topk(
        inputs,
        out_positions,
        topk_blocks=0,
        block_size=1,
        publish=False,
        candidates=None,
    )


class DenseCandidateIndexer(CandidateIndexer):
    """Candidates as block ids per request (``PrefillCandidateBlocks``): the
    source keeps its best blocks from its dense scores, a consumer masks its own
    dense scores to -inf outside them and runs the plain top-k, tile by tile.
    The prefill implementation for the CP layout; Hopper still runs the same
    selection inline in the backend."""

    def __init__(self, topk_blocks: int, block_size: int):
        self.topk_blocks = topk_blocks
        self.block_size = block_size

    def publish_prefill(
        self, inputs: PrefillIndexerInputs, out_positions: torch.Tensor
    ) -> PrefillCandidateBlocks:
        published = _dense_topk(
            inputs,
            out_positions,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            publish=True,
            candidates=None,
        )
        assert published is not None
        return published

    def select_prefill(
        self,
        published: CandidateMetadata,
        inputs: PrefillIndexerInputs,
        out_positions: torch.Tensor,
    ) -> None:
        assert isinstance(published, PrefillCandidateBlocks), "candidate blocks missing"
        _dense_topk(
            inputs,
            out_positions,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            publish=False,
            candidates=published,
        )

    def prefill_tail(
        self, published: CandidateMetadata, tail_lens: List[int]
    ) -> PrefillCandidateBlocks:
        assert isinstance(published, PrefillCandidateBlocks), "candidate blocks missing"
        return published.tail(tail_lens)


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

    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

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
