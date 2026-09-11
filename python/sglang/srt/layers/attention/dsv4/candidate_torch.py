# TODO(candidate): retire with the last path that still selects through masks
# (Hopper decode, prefill); the paged fp4 decode path already has the DeepGEMM one.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMetadata,
    IndexerInputs,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    fp4_paged_mqa_logits,
    fp32_jit_paged_topk,
    select_candidate_blocks,
)


@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = None  # prefill: [rows_b, lc_b] each


def published_masks(candidate) -> CandidateMasks:
    """The forward's ``candidate_metadata`` as the masks the source published."""
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


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
    """Apply candidate-block filtering and return logits plus an optional published mask.

    Mask columns past each sequence length to -inf before selection: the paged
    logits kernel leaves that tail uninitialized, and an all -inf block means
    unreachable. This path is graph-captured and must not synchronize with the host.
    Work scales with allocated page-table capacity, not live sequence length.
    """
    if not (is_candidate_source or uses_candidates):
        return logits, None

    if (
        logits.is_cuda
        and torch.version.cuda is not None
        and logits.ndim == 2
        and logits.stride(1) == 1
        and seq_lens.device == logits.device
        and seq_lens.dtype in (torch.int32, torch.int64)
        and seq_lens.is_contiguous()
        and seq_lens.shape in ((logits.shape[0],), (logits.shape[0], 1))
        and logits.numel() > 0
        and 0 < block_size <= 1024
    ):
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_logits,
        )

        if not is_candidate_source:
            assert (
                torch.is_tensor(published)
                and published.shape[0] == logits.shape[0]
                and published.shape[1] >= logits.shape[1]
            ), "candidate mask missing for decode"
        return candidate_block_logits(
            logits,
            seq_lens,
            topk_blocks=topk_blocks,
            block_size=block_size,
            published=None if is_candidate_source else published,
        )

    lens_col = seq_lens if seq_lens.dim() > 1 else seq_lens.unsqueeze(-1)
    reachable = torch.arange(logits.shape[-1], device=logits.device) < lens_col
    logits = logits.float().masked_fill(~reachable, -torch.inf)

    if is_candidate_source:
        # The source scores over every reachable position itself and only publishes,
        # which is what the reference does.
        return logits, select_candidate_blocks(
            logits, lens_col, topk_blocks=topk_blocks, block_size=block_size
        )

    assert torch.is_tensor(published) and published.shape[0] == logits.shape[0], (
        "candidate mask missing for decode"
    )
    return logits.masked_fill(~published[:, : logits.shape[-1]], -torch.inf), None


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


class TorchCandidateIndexer:
    def __init__(self, topk_blocks: int, block_size: int):
        self.topk_blocks = topk_blocks
        self.block_size = block_size

    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> CandidateMasks:
        """Layer 20 end to end: dense logits, its own plain top-k into
        ``page_indices`` (and ``raw_indices`` when given), and the mask for the
        layers after it; the backend stores the mask on the forward metadata."""
        metadata = inputs.metadata
        logits = fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            metadata.c4_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_c4_seq_len,
        )
        logits, mask = two_level_decode_logits(
            logits,
            metadata.c4_seq_lens,
            is_candidate_source=True,
            uses_candidates=False,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            published=None,
        )
        fp32_jit_paged_topk(logits, metadata, page_indices, raw_indices)
        return CandidateMasks(mask=mask)

    def select_decode(
        self,
        candidate_metadata: CandidateMasks,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """A consumer: dense logits, masked, top-k, written as slots through the
        page table with ``-1`` past the valid count (and as positions into
        ``raw_indices`` when given)."""
        metadata = inputs.metadata
        page_size = metadata.c4_page_size
        assert isinstance(candidate_metadata, CandidateMasks)
        logits = fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            metadata.c4_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_c4_seq_len,
        )
        logits, _ = two_level_decode_logits(
            logits,
            metadata.c4_seq_lens,
            is_candidate_source=False,
            uses_candidates=True,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
            published=candidate_metadata.mask,
        )
        # raw positions into `selected`; `page_indices` gets the unfiltered slots
        # here and is rewritten with the masked selection just below
        selected = torch.empty_like(page_indices)
        fp32_jit_paged_topk(logits, metadata, page_indices, raw_indices=selected)
        if logits.is_cuda and torch.version.cuda is not None:
            # fused: drop the selections the mask zeroed, page-transform the rest
            from sglang.kernels.ops.attention.dsv4.indexer_postprocess import (
                filter_topk_pages,
            )

            filter_topk_pages(
                logits,
                selected,
                metadata.page_table,
                page_indices,
                page_size,
                raw_indices,
            )
        else:
            selected = mask_topk_scores(logits, selected)
            columns = selected.clamp_min(0).to(torch.int64)
            slots = metadata.page_table.gather(1, columns // page_size) * page_size
            slots = slots + columns % page_size
            page_indices.copy_(torch.where(selected >= 0, slots, -1))
            if raw_indices is not None:
                raw_indices.copy_(selected)
