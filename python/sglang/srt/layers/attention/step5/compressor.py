"""Step5 non-online compression plans and ring execution."""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.step5.indexer import append_ring
from sglang.srt.layers.attention.step5.metadata import Step5IndexerMetadata
from sglang.srt.utils.device import async_h2d


def build_prefill_compression_inputs(
    *,
    request_slots: torch.Tensor,
    prefix_lengths: list[int],
    append_lengths: list[int],
    req_to_token: torch.Tensor,
    compressed_index_key_pages: int,
) -> dict:
    """Build the prefill compression plan shared by every DSA layer.

    A request with any invalid completed block is disabled in full; padding rows
    consume input rows but never own ring cells or destinations.
    """
    rows, source = [], 0
    for base, length in zip(prefix_lengths, append_lengths):
        rows.append((base, source, length))
        source += length
    device = req_to_token.device
    raw_slots = torch.as_tensor(request_slots, dtype=torch.int64, device=device)
    slots = raw_slots.clamp(0, req_to_token.shape[0] - 1)
    packed = async_h2d(rows, dtype=torch.int64, device=device).reshape(-1, 3)
    bases, sources, lengths = packed.T.contiguous().unbind(0)
    active = (raw_slots != 0) & (lengths > 0)
    uses = torch.zeros(req_to_token.shape[0], dtype=torch.int32, device=device)
    uses.scatter_add_(0, slots, active.to(torch.int32))
    admitted = (raw_slots == slots) & (uses[slots] == 1)
    max_blocks = max(((row[0] % 8 + row[2]) // 8 for row in rows), default=0)
    block_idx = torch.arange(max_blocks, device=device)[:, None]
    offsets = torch.arange(8, device=device)
    logical = (bases // 8 + block_idx) * 8
    complete = logical + 8 <= bases + lengths
    mapped = req_to_token[
        slots[None, :, None],
        (logical[:, :, None] + offsets).clamp_max(req_to_token.shape[1] - 1),
    ].long()
    first = mapped[:, :, 0]
    valid = (
        (first >= 64)
        & (first < compressed_index_key_pages * 64)
        & (first.remainder(8) == 0)
        & (mapped == first[:, :, None] + offsets).all(2)
    )
    return dict(
        request_slots=slots,
        input_start_positions=bases,
        input_start_rows=sources,
        input_lengths=lengths,
        request_active=active,
        request_mapping_valid=admitted & (~complete | valid).all(0),
        request_count_valid=torch.ones(1, dtype=torch.bool, device=device),
        compressed_key_write_indices=(first // 8).contiguous(),
        num_input_tokens=source,
    )


def compress_index_keys(
    *,
    metadata: Step5IndexerMetadata,
    compressed_index_keys: torch.Tensor,
    index_key_ring: torch.Tensor,
    compression_logits_ring: torch.Tensor,
    ring_positions: torch.Tensor,
    index_key: torch.Tensor,
    compression_logits: torch.Tensor,
) -> None:
    append_ring(
        slots=metadata.request_slots,
        bases=metadata.input_start_positions,
        sources=metadata.input_start_rows,
        lengths=metadata.input_lengths,
        active=metadata.request_active,
        admitted=metadata.request_mapping_valid,
        count_valid=metadata.request_count_valid,
        physical_blocks=metadata.compressed_key_write_indices,
        enabled=metadata.compression_enabled,
        success=metadata.compression_success,
        index_key=index_key,
        compression_logits=compression_logits,
        index_key_ring=index_key_ring,
        compression_logits_ring=compression_logits_ring,
        ring_positions=ring_positions,
        compressed_index_keys=compressed_index_keys,
    )
