"""Step5 non-online compression plans and ring execution."""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.step5.csa_ring import AppendRingPlan, append_ring
from sglang.srt.layers.attention.step5.metadata import Step5IndexerMetadata
from sglang.srt.utils.common import async_h2d


def build_prefill_compression_inputs(
    *,
    request_slots,
    prefix_lengths,
    append_lengths,
    req_to_token,
    compressed_index_key_pages,
) -> dict:
    """Build a batched prefill plan once, shared by every DSA layer.

    CPU spans are native scheduler metadata. All FULL mapping checks are batched
    on device. A request with any invalid completed region is disabled in full.
    Padding consumes source rows but never owns ring cells or destinations.
    """
    if (
        req_to_token.device.type != "cuda"
        or req_to_token.ndim != 2
        or req_to_token.dtype not in (torch.int32, torch.int64)
        or compressed_index_key_pages < 2
    ):
        raise ValueError("CSA append requires native CUDA request/FULL mapping")
    if any(len(v) != len(request_slots) for v in (prefix_lengths, append_lengths)):
        raise ValueError("CSA append batch lengths differ")
    rows, source = [], 0
    for base, length in zip(prefix_lengths, append_lengths):
        if (
            length < 0
            or base < 0
            or base + length > min(1048576, req_to_token.shape[1])
        ):
            raise ValueError("CSA append invalid logical span")
        rows.append((base, source, length))
        source += length
    if source > 2**31 - 1:
        raise ValueError("CSA append packed input exceeds int32")
    device = req_to_token.device
    raw_slots = torch.as_tensor(request_slots, dtype=torch.int64, device=device)
    slots = raw_slots.clamp(0, req_to_token.shape[0] - 1)
    packed = async_h2d(rows, dtype=torch.int64, device=device).reshape(-1, 3)
    bases, sources, lengths = packed.T.contiguous().unbind(0)
    active = (raw_slots != 0) & (lengths > 0)
    uses = torch.zeros(req_to_token.shape[0], dtype=torch.int32, device=device)
    uses.scatter_add_(0, slots, active.to(torch.int32))
    admitted = (raw_slots == slots) & (uses[slots] == 1)
    max_regions = max(((row[0] % 8 + row[2]) // 8 for row in rows), default=0)
    region = torch.arange(max_regions, device=device)[:, None]
    offsets = torch.arange(8, device=device)
    logical = (bases // 8 + region) * 8
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
    batch: Step5IndexerMetadata,
    compressed_index_keys,
    index_key_ring,
    compression_logits_ring,
    positions,
    index_key,
    compression_logits,
    enabled,
    success,
    assert_success=False,
    use_pdl=False,
):
    """Compress history/current inputs, then retain inputs for the next forward.

    One batched launch sequence serves prefill, decode and verify. Acceptance
    changes native sequence lengths only; rejected ring values are overwritten
    when their positions are reused. Region scoring enforces query causality.
    """
    # K is packed; Z may be a row-padded view of the combined projection.
    # No forward-mode flag reaches the kernels.
    for value in (index_key, compression_logits):
        if (
            value.shape != (batch.num_input_tokens, 1, 256)
            or value.dtype != torch.bfloat16
            or value.device != batch.request_slots.device
            or value.stride(2) != 1
            or value.stride(0) < 256
        ):
            raise ValueError(
                "Step5 compression requires BF16 index K/Z rows: "
                f"expected shape={(batch.num_input_tokens, 1, 256)}, "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device}, strides={value.stride()}"
            )
    if not index_key.is_contiguous():
        raise ValueError("Step5 compression requires packed BF16 index K")
    plan = AppendRingPlan(
        slots=batch.request_slots,
        bases=batch.input_start_positions,
        sources=batch.input_start_rows,
        lengths=batch.input_lengths,
        active=batch.request_active,
        admitted=batch.request_mapping_valid,
        count_valid=batch.request_count_valid,
        physical_regions=batch.compressed_key_write_indices,
        max_regions=batch.compressed_key_write_indices.shape[0],
        enabled=enabled,
        success=success,
    )
    return append_ring(
        plan=plan,
        index_key=index_key,
        compression_logits=compression_logits,
        index_key_ring=index_key_ring,
        compression_logits_ring=compression_logits_ring,
        positions=positions,
        compressed_index_keys=compressed_index_keys,
        assert_success=assert_success,
        use_pdl=use_pdl,
    )
