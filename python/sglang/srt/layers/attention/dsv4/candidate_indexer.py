from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )


class CandidateMetadata:
    """Base of an implementation's published state on
    ``DSV4Metadata.candidate_metadata``."""


@dataclass(frozen=True)
class IndexerInputs:
    """One index-source layer's operands on the paged fp4 decode path (one query
    row per request, or per draft token under verify)."""

    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    metadata: PagedIndexerMetadata  # this ratio's lengths, page table and plans
    # [rows] int, one request id per query row, the rows of one request
    # consecutive (verify: its draft tokens); None = every row its own request
    request_ids: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]


def make_candidate_indexer(
    topk_blocks: int, block_size: int
) -> Optional[DeepGemmCandidateIndexer]:
    """The paged fp4 decode path's two-level indexer; None on Hopper, whose decode
    indexer selects through masks inline."""
    if topk_blocks <= 0 or get_platform().device_sm < 100:
        return None
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        raise RuntimeError(
            "the candidate indexer needs DeepGEMM's paged sparse MQA logits "
            "(sgl-deep-gemm >= 0.2.0 with SGLANG_ENABLE_JIT_DEEPGEMM on)"
        )
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )

    return DeepGemmCandidateIndexer(topk_blocks, block_size)


# TODO(candidate): Hopper decode and prefill still select through these masks
# inline in the backend; move them behind the protocol as publish/select_prefill.
@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = None  # prefill: [rows_b, lc_b] each
    # dense fp4 prefill: per request [rows_b, blocks] uint8 keep flags over blocks
    # of `block_size` compressed positions, all the same width and consecutive
    # row slices of one buffer, so a consumer hands them to the fused masked
    # top-k as one [rows, blocks] tensor (`select_candidate_block_masks`)
    block_masks: Optional[List[torch.Tensor]] = None
    block_size: int = 0


def published_masks(candidate) -> CandidateMasks:
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


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


def select_candidate_block_masks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    *,
    out: torch.Tensor,
) -> None:
    """The selection of :func:`select_candidate_blocks` (unreachable positions
    at -inf) as ``uint8`` keep flags ``out[i, block]``, without the score copy,
    the compare or the position mask: block keys in one pass, then the fused
    ragged top-k over them. ``out`` is zeroed beforehand and one column wider
    than the block count: rows with fewer than ``topk_blocks`` blocks park their
    ``-1`` padding there. ``logits`` rows must be 32-byte aligned."""
    from sglang.kernels.ops.attention.dsv4.candidate_table import amax8_varlen
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    assert block_size == 8, "amax8_varlen keys blocks of 8 positions"
    rows, width = logits.shape
    device = logits.device
    spare = out.shape[1] - 1
    nblocks = (compress_lens + (block_size - 1)) // block_size
    # keys past a row's block count stay uninitialised: the top-k reads a row
    # up to its block count only (v2 wants the stride a multiple of 4)
    nblocks_max = (width + block_size - 1) // block_size
    keys = logits.new_empty(rows, (nblocks_max + 3) // 4 * 4)
    amax8_varlen(logits, compress_lens, out=keys)
    blocks = torch.empty(rows, topk_blocks, dtype=torch.int32, device=device)
    topk_transform_ragged_v2(
        keys,
        nblocks,
        out_offsets=torch.zeros(rows, dtype=torch.int32, device=device),
        out_indices=blocks,
    )
    out.scatter_(1, blocks.masked_fill(blocks < 0, spare).to(torch.int64), 1)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: a bool mask over positions keeping the
    topk_blocks best-scoring blocks per query. Unreachable positions are already -inf
    in logits, so an all -inf block means not reachable yet; the block holding the
    query's newest position is always kept."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
