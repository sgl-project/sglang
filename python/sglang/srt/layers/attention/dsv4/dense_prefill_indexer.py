from __future__ import annotations

from typing import List, Optional

import torch

from sglang.kernels.ops.attention.dsv4.dense_prefill import dense_prefill_topk
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateIndexer,
    CandidateMetadata,
    PrefillCandidateBlocks,
    PrefillIndexerInputs,
)

# TODO: use a per-forward mqa_logits_budget_bytes() budget that also
# leaves room for candidate masks and block-selection scratch.
_SCORE_BUDGET_BYTES = 2 << 30


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
        candidates=candidates.request_blocks if candidates is not None else None,
        budget_bytes=_SCORE_BUDGET_BYTES,
    )
    out_positions.copy_(selected)
    if published is None:
        return None
    return PrefillCandidateBlocks(request_blocks=published)


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
