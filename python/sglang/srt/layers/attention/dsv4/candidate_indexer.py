from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import msgspec
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata


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


@dataclass(frozen=True)
class TritonDecodeInputs:
    """BF16 queries and logical-to-physical slots for the FP4 cache scorer."""

    q: torch.Tensor
    weights: torch.Tensor
    slots: torch.Tensor  # [rows, context], physical index-K slots
    lens: torch.Tensor  # [rows], visible logical K count
    table: torch.Tensor  # paged FP4 payload and scales
    page_size: int
    topk: int


class PrefillIndexerInputs(msgspec.Struct, frozen=True):
    """One index layer's operands on the dense fp4 prefill path: the chunk's query
    rows, one request's rows consecutive."""

    q_fp4: torch.Tensor  # [rows, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, heads] int32, packed ue8m0
    weights: torch.Tensor  # [rows, heads] fp32 head weights
    compress_lens: torch.Tensor  # [rows] int32, compressed positions the row sees
    # [rows] int32: first flattened-K column of the row's request; selections
    # are flattened-K columns, request start + compressed position
    request_starts: torch.Tensor
    lens_per_request: List[int]  # compressed positions at each request's newest token
    rows_per_request: List[int]  # query rows of each request
    # the flattened index K of the batch (int8 [n, 64], int32 [n]), column j of
    # a row's dense scores against kv[request_starts[t] + j]
    kv: Tuple[torch.Tensor, torch.Tensor]
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool
    page_size: int  # index-K pool page size, in slots
    # [rows, kv pages] int32: the row's request in the KV pool, whose pages hold
    # kv_page_size tokens; expand_index_page_table gives the index-K pool's
    kv_page_table: torch.Tensor
    kv_page_size: int
    compress_ratio: int

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]


@dataclass(frozen=True)
class TritonPrefillInputs:
    """Unpacked BF16 operands; requests occupy consecutive query rows.

    ``get_keys(b)`` lazily returns request b's dequantized [length, dim] K,
    so only one request's K needs to be live. Q already has the model's RoPE
    and FP4 rounding applied; scoring preserves the Torch BF16 arithmetic.
    Selections use the same flattened-K coordinates as PrefillIndexerInputs.
    """

    q: torch.Tensor  # [rows, heads, dim], unpacked BF16
    weights: torch.Tensor  # [rows, heads]
    compress_lens: torch.Tensor  # [rows], causal lengths in compressed positions
    request_starts: torch.Tensor  # [rows], flattened-K offsets
    lens_per_request: List[int]
    rows_per_request: List[int]
    get_keys: Callable[[int], torch.Tensor]

    @property
    def num_rows(self) -> int:
        return self.q.shape[0]


PrefillInputs = Union[PrefillIndexerInputs, TritonPrefillInputs]


def expand_index_page_table(
    page_table: torch.Tensor,
    *,
    full_page_size: int,
    compress_ratio: int,
    index_page_size: int,
) -> torch.Tensor:
    """Block table of a low-ratio indexer-K pool, which pages at `index_page_size`
    slots: [bs, n] -> [bs, n * blocks_per_page] int32. The kernel reads compressed
    slot j at page_table[b, j // index_page_size] * index_page_size + j %
    index_page_size, which after this expansion is the c1/c2 pool slot of the same
    position."""
    slots_per_page = full_page_size // compress_ratio
    assert slots_per_page % index_page_size == 0, (
        f"{full_page_size = } / {compress_ratio = } must be a multiple of "
        f"{index_page_size = }"
    )
    blocks_per_page = slots_per_page // index_page_size
    if blocks_per_page == 1:
        return page_table
    bs, n = page_table.shape
    base = page_table.to(torch.int64) * blocks_per_page
    offsets = torch.arange(blocks_per_page, device=page_table.device, dtype=torch.int64)
    expanded = base.unsqueeze(-1) + offsets  # [bs, n, blocks_per_page]
    return expanded.reshape(bs, n * blocks_per_page).to(torch.int32)


class CandidateIndexer(ABC):
    """The two-level low-ratio indexer. The candidate source publishes what its
    consumers need to select among its top candidate blocks, on the forward
    metadata; a consumer selects its top-k from that.

    TritonCandidateIndexer uses Triton for sparse candidate scoring."""

    @abstractmethod
    def publish_decode(
        self, inputs, page_indices, raw_indices=None
    ) -> CandidateMetadata:
        """Write the source top-k and publish candidates for this decode/verify step."""

    @abstractmethod
    def select_decode(self, published, inputs, page_indices, raw_indices=None) -> None:
        """Select this row's top-k from its source's candidates."""

    @abstractmethod
    def publish_prefill(
        self, inputs: PrefillInputs, out_positions: torch.Tensor
    ) -> CandidateMetadata:
        """The source layer's own top-``k`` (``k = out_positions.shape[1]``) as
        flattened-K columns, ``-1`` padded, unordered, plus its candidates for
        the chunk: both come from one pass over its dense scores."""

    @abstractmethod
    def select_prefill(
        self,
        published: CandidateMetadata,
        inputs: PrefillInputs,
        out_positions: torch.Tensor,
    ) -> None:
        """A consumer's top-``k`` over its published candidates, in the same
        layout as ``publish_prefill`` writes."""

    @abstractmethod
    def prefill_tail(
        self, published: CandidateMetadata, tail_lens: List[int]
    ) -> CandidateMetadata:
        """The candidates of the last ``tail_lens[b]`` rows of each request, for
        the late layers that run on the tail only."""


def make_candidate_indexer(
    topk_blocks: int, block_size: int
) -> Optional[CandidateIndexer]:
    """Select the candidate implementation for prefill and decode/verify.

    SM100's explicit prefill override retains its normal decode/verify indexer.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.triton_candidate_indexer import (
        TritonCandidateIndexer,
    )
    from sglang.srt.runtime_context import get_parallel, get_platform

    if topk_blocks <= 0:
        return None
    if get_platform().device_sm < 100:
        return TritonCandidateIndexer(topk_blocks, block_size)
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
    from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import (
        DenseCandidateIndexer,
    )

    prefill_dense = None
    if get_parallel().attn_cp_size > 1:
        prefill_dense = DenseCandidateIndexer(topk_blocks, block_size)
    indexer = DeepGemmCandidateIndexer(
        topk_blocks, block_size, prefill_dense=prefill_dense
    )
    if envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get():
        return TritonCandidateIndexer(topk_blocks, block_size, decode_indexer=indexer)
    return indexer


class PrefillCandidateBlocks(CandidateMetadata, msgspec.Struct):
    request_blocks: List[torch.Tensor]

    def tail(self, lengths: List[int]) -> PrefillCandidateBlocks:
        if len(lengths) != len(self.request_blocks) or any(
            length < 0 or length > blocks.shape[0]
            for blocks, length in zip(self.request_blocks, lengths)
        ):
            raise ValueError("tail lengths must match and fit the published requests")
        return PrefillCandidateBlocks(
            request_blocks=[
                blocks[blocks.shape[0] - length :]
                for blocks, length in zip(self.request_blocks, lengths)
            ]
        )


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
