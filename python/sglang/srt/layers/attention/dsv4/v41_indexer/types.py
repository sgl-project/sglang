"""The inputs of the V4.1 indexer backends, including the selection buffers they fill."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, TypeVar

import msgspec
import torch

from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata


def get_tail_row_indices(
    full_rows_per_request: List[int],
    tail_rows_per_request: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Row indices of each request's tail, copied without a host sync."""
    rows, start = [], 0
    for n, t in zip(full_rows_per_request, tail_rows_per_request):
        rows.extend(range(start + n - t, start + n))
        start += n
    return async_h2d(rows, dtype=torch.int64, device=device)


class CandidateMetadata:
    def tail(self, rows_per_request: List[int]) -> CandidateMetadata:
        raise NotImplementedError(f"{type(self).__name__} is not a prefill publish")


class PrefillInputs(msgspec.Struct, frozen=True, kw_only=True):
    """An eager extend: the backend projects the queries and gathers the index K."""

    indexer: DeepseekV41Indexer
    layer_id: int
    compress_ratio: int
    freqs_cis: torch.Tensor
    x: torch.Tensor  # [rows, hidden] bf16
    q_lora: torch.Tensor  # [rows, q_lora_rank]
    positions: torch.Tensor  # [rows] absolute position of each row

    # [rows] req_to_token row of each query row; None under prefill CP, where a
    # rank's rows are an interleaved subset of the batch's.
    req_rows: Optional[torch.Tensor]
    req_pool_indices: torch.Tensor  # [requests] req_to_token row of each request
    # [rows, pages] int32 at the KV page size; the index-K pool pages it with
    # expand_index_page_table.
    kv_page_table: torch.Tensor
    # Visible full-length context of each request at its newest row, and the
    # query rows of each request in row order.
    seq_lens_cpu: Optional[List[int]]
    rows_per_request: Optional[List[int]]
    rows_per_request_device: Optional[torch.Tensor]

    # The selection, unordered, -1 padded: [rows, topk] int32 request-local
    # compressed positions, and the same picks as index-K pool slots, which is
    # None when this forward's attention reads only the positions (sparse prefill).
    out_raw_indices: torch.Tensor
    out_page_indices: Optional[torch.Tensor]

    # None uses the fixed cap.
    score_budget_bytes: Optional[int] = None

    def reset_outputs(self) -> None:
        self.out_raw_indices.fill_(-1)
        if self.out_page_indices is not None:
            self.out_page_indices.fill_(-1)


class DecodeInputs(msgspec.Struct, frozen=True, kw_only=True):
    """Decode (one row per request) or verify (one row per draft token)."""

    indexer: DeepseekV41Indexer
    layer_id: int
    compress_ratio: int
    freqs_cis: torch.Tensor
    x: torch.Tensor  # [rows, hidden] bf16
    q_lora: torch.Tensor  # [rows, q_lora_rank]
    positions: torch.Tensor  # [rows] absolute position of each row
    req_rows: torch.Tensor  # [rows] req_to_token row of each query row
    paged_metadata: PagedIndexerMetadata
    is_verify: bool

    # [rows, topk] int32 index-K pool slots of the selection, unordered, -1 padded.
    out_page_indices: torch.Tensor

    def reset_outputs(self) -> None:
        self.out_page_indices.fill_(-1)


class CapturedPrefillInputs(msgspec.Struct, frozen=True, kw_only=True):
    """An extend under the prefill graph: projected queries, paged index K."""

    indexer: DeepseekV41Indexer
    layer_id: int
    q: torch.Tensor  # [rows, heads, 128] bf16, roped
    weights: torch.Tensor  # [rows, heads] bf16 head weights
    paged_metadata: PagedIndexerMetadata
    # The selection as pool slots and as request-local compressed positions.
    out_page_indices: torch.Tensor
    out_raw_indices: torch.Tensor


Metadata = TypeVar("Metadata", bound=CandidateMetadata)


class PrefillCandidates(Protocol[Metadata]):
    """The attention backend keeps a publish alive and hands it back unread."""

    def publish_prefill(self, inputs: PrefillInputs) -> Optional[Metadata]: ...

    def consume_prefill(
        self, inputs: PrefillInputs, published: Optional[Metadata]
    ) -> None: ...


class DecodeCandidates(Protocol[Metadata]):
    def publish_decode(self, inputs: DecodeInputs) -> Optional[Metadata]: ...

    def consume_decode(
        self, inputs: DecodeInputs, published: Optional[Metadata]
    ) -> None: ...
