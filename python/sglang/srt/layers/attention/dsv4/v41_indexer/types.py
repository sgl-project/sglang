"""What the attention backend hands the V4.1 indexer backends, and what it gets
back."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, TypeVar

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata


class CandidateMetadata:
    def tail(self, rows_per_request: List[int]) -> CandidateMetadata:
        raise NotImplementedError(f"{type(self).__name__} is not a prefill publish")


class Selection(msgspec.Struct, frozen=True):
    page_indices: torch.Tensor  # index-K pool slots, ascending
    raw_indices: Optional[torch.Tensor]  # the same picks as compressed positions

    def reset(self) -> None:
        self.page_indices.fill_(-1)
        if self.raw_indices is not None:
            self.raw_indices.fill_(-1)


class PrefillInputs(msgspec.Struct, frozen=True, kw_only=True):
    """An eager extend: the backend projects the queries itself and gathers the
    index K it needs."""

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


class CapturedPrefillInputs(msgspec.Struct, frozen=True, kw_only=True):
    """An extend under the prefill graph: the queries and head weights come
    projected, and the index K is read paged, as on decode."""

    indexer: DeepseekV41Indexer
    layer_id: int
    q: torch.Tensor  # [rows, heads, 128] bf16, roped
    weights: torch.Tensor  # [rows, heads] bf16 head weights
    paged_metadata: PagedIndexerMetadata


Metadata = TypeVar("Metadata", bound=CandidateMetadata)


class PrefillCandidates(Protocol[Metadata]):
    """The candidate scheme on prefill: what a publish carries is the backend's
    own; the attention backend keeps it alive and hands it back unread."""

    def publish_prefill(
        self,
        inputs: PrefillInputs,
        out: Selection,
    ) -> Optional[Metadata]: ...

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[Metadata],
        out: Selection,
    ) -> None: ...


class DecodeCandidates(Protocol[Metadata]):
    def publish_decode(
        self,
        inputs: DecodeInputs,
        out: Selection,
    ) -> Optional[Metadata]: ...

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[Metadata],
        out: Selection,
    ) -> None: ...
