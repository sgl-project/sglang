"""Shared forward metadata for MLU attention backends."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, overload

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class ForwardMetadata:
    """Per-forward attention state shared across decoder layers.

    Keep one metadata class per backend as it grows, with optional or defaulted
    fields for new execution modes and kernel scheduling needs. Specialized
    backends can extend this class with cache-layout-specific fields.
    Prepare batch-dependent state once and reuse it across layers.
    """

    # Physical KV page indices, shape [batch_size, max_num_blocks_per_seq].
    block_tables: torch.Tensor

    # Total KV lengths including cached prefixes, shape [batch_size].
    seq_lens: torch.Tensor

    # Cumulative prefill query lengths, shape [prefill_count + 1].
    # Used for extend/mixed batches; None for decode.
    cu_seqlens_q: torch.Tensor | None = None

    # Cumulative prefill KV lengths, shape [prefill_count + 1].
    # Includes cached prefixes; used for extend/mixed batches, None for decode.
    cu_seqlens_kv: torch.Tensor | None = None

    # Maximum prefill query length; only consumed by extend/mixed attention.
    max_seq_len_q: int = 1

    # Maximum prefill KV length including cached prefixes.
    # Set to the context limit for decode, where this field is not consumed.
    max_seq_len_kv: int = 0

    # Whether the prefill requests can read the current K/V directly.
    is_uncached_prefill: bool = False

    mixed_prefill_reqs: int = -1
    mixed_prefill_tokens: int = 0


_MetadataT = TypeVar("_MetadataT", bound=ForwardMetadata)


@overload
def build_forward_metadata(
    forward_batch: "ForwardBatch",
    req_to_token: torch.Tensor,
    page_size: int,
    max_context_len: int,
) -> ForwardMetadata: ...


@overload
def build_forward_metadata(
    forward_batch: "ForwardBatch",
    req_to_token: torch.Tensor,
    page_size: int,
    max_context_len: int,
    *,
    metadata_cls: type[_MetadataT],
) -> _MetadataT: ...


def build_forward_metadata(
    forward_batch: "ForwardBatch",
    req_to_token: torch.Tensor,
    page_size: int,
    max_context_len: int,
    *,
    metadata_cls: type[ForwardMetadata] = ForwardMetadata,
) -> ForwardMetadata:
    """Build shared metadata for a non-idle MLU attention batch."""
    fb = forward_batch
    tables = req_to_token[fb.req_pool_indices, :max_context_len:page_size] // page_size
    seq_lens = fb.seq_lens.to(torch.int32)
    if fb.forward_mode.is_decode():
        forward_metadata = metadata_cls(
            tables, seq_lens, max_seq_len_kv=max_context_len
        )
    elif fb.forward_mode == ForwardMode.EXTEND or fb.forward_mode.is_mixed():
        prefill_count = fb.batch_size
        if fb.forward_mode.is_mixed():
            prefill_count = fb.mixed_prefill_reqs
            if prefill_count is None or not 0 <= prefill_count <= fb.batch_size:
                raise ValueError(
                    "MLU mixed attention requires a valid mixed_prefill_reqs."
                )
            if any(n != 1 for n in fb.extend_seq_lens_cpu[prefill_count:]):
                raise NotImplementedError(
                    "MLU mixed attention currently requires one query token per decode request."
                )
        query_lens = fb.extend_seq_lens_cpu[:prefill_count]
        forward_metadata = metadata_cls(
            block_tables=tables,
            seq_lens=seq_lens,
            cu_seqlens_q=torch.nn.functional.pad(
                torch.cumsum(
                    fb.extend_seq_lens[:prefill_count], dim=0, dtype=torch.int32
                ),
                (1, 0),
            ),
            cu_seqlens_kv=torch.nn.functional.pad(
                torch.cumsum(seq_lens[:prefill_count], dim=0, dtype=torch.int32),
                (1, 0),
            ),
            max_seq_len_q=max(query_lens, default=0),
            max_seq_len_kv=max(fb.seq_lens_cpu[:prefill_count].tolist(), default=0),
            is_uncached_prefill=not any(fb.extend_prefix_lens_cpu[:prefill_count]),
        )
        if fb.forward_mode.is_mixed():
            forward_metadata.mixed_prefill_reqs = prefill_count
            forward_metadata.mixed_prefill_tokens = sum(query_lens)
    else:
        raise NotImplementedError(
            f"Unsupported MLU attention forward mode: {fb.forward_mode}"
        )
    return forward_metadata
