"""Dense fallback for DeepSeek V4.1 candidate-block selection.

The sparse DeepGEMM candidate kernel is currently limited to SM100.  This
implementation preserves the candidate semantics on other Blackwell devices by
computing dense FP4 paged logits and publishing a boolean block mask.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMasks,
    IndexerInputs,
    mask_topk_scores,
    select_candidate_blocks,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    deep_gemm_fp4_paged_mqa_logits,
    topk_transform_paged_from_metadata,
)


def two_level_decode_logits(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    is_candidate_source: bool,
    uses_candidates: bool,
    topk_blocks: int,
    block_size: int,
    published: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply the published candidate mask or create one at the source layer."""
    if not (is_candidate_source or uses_candidates):
        return logits, None

    lens_col = seq_lens if seq_lens.dim() > 1 else seq_lens.unsqueeze(-1)
    reachable = torch.arange(logits.shape[-1], device=logits.device) < lens_col
    logits = logits.float().masked_fill(~reachable, -torch.inf)

    if is_candidate_source:
        return logits, select_candidate_blocks(
            logits,
            lens_col,
            topk_blocks=topk_blocks,
            block_size=block_size,
        )

    assert torch.is_tensor(published) and published.shape[0] == logits.shape[0], (
        "candidate mask missing for decode"
    )
    return logits.masked_fill(~published[:, : logits.shape[-1]], -torch.inf), None


class TorchCandidateIndexer:
    def __init__(self, topk_blocks: int, block_size: int):
        self.topk_blocks = topk_blocks
        self.block_size = block_size

    @staticmethod
    def _scores(inputs: IndexerInputs) -> torch.Tensor:
        metadata = inputs.metadata
        return deep_gemm_fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            metadata.compressed_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
        )

    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> CandidateMasks:
        metadata = inputs.metadata
        logits, mask = two_level_decode_logits(
            self._scores(inputs),
            metadata.compressed_seq_lens,
            is_candidate_source=True,
            uses_candidates=False,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            published=None,
        )
        topk_transform_paged_from_metadata(logits, metadata, page_indices, raw_indices)
        return CandidateMasks(mask=mask)

    def select_decode(
        self,
        candidate_metadata: CandidateMasks,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        metadata = inputs.metadata
        logits, _ = two_level_decode_logits(
            self._scores(inputs),
            metadata.compressed_seq_lens,
            is_candidate_source=False,
            uses_candidates=True,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            published=candidate_metadata.mask,
        )

        selected = torch.empty_like(page_indices)
        topk_transform_paged_from_metadata(logits, metadata, page_indices, selected)
        selected = mask_topk_scores(logits, selected)
        columns = selected.clamp_min(0).to(torch.int64)
        page_size = metadata.compressed_page_size
        slots = metadata.page_table.gather(1, columns // page_size) * page_size
        slots = slots + columns % page_size
        page_indices.copy_(torch.where(selected >= 0, slots, -1).to(page_indices.dtype))
        if raw_indices is not None:
            raw_indices.copy_(selected)
