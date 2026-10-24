"""Packed Mamba2 prefill adapter for FlashInfer's checkpoint-capable SSD.

Metadata is built once per forward, not once per layer. FlashInfer requires a
128-token physical extent; a tail is padded as a separate, discarded sequence
so it cannot decay a real request's final state. Checkpoint positions are
exclusive sequence-relative endpoints supplied by the caller's tracking policy.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SSDPrefillMetadata:
    num_tokens: int
    num_sequences: int
    padding: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    seq_chunk_cumsum: torch.Tensor
    checkpoint_token_indices: torch.Tensor | None
    checkpoint_state_slots: torch.Tensor | None
    initial_checkpoint_rows: torch.Tensor | None
    initial_checkpoint_slots: torch.Tensor | None


def prepare_ssd_prefill_metadata(
    lengths: list[int],
    device: torch.device,
    checkpoint_lengths: list[int] | None = None,
    checkpoint_slots: list[int] | torch.Tensor | None = None,
) -> SSDPrefillMetadata:
    """Build packed logical chunks from CPU lengths, without a GPU readback."""
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("SSD prefill requires nonempty positive sequence lengths")
    if (checkpoint_lengths is None) != (checkpoint_slots is None):
        raise ValueError("Checkpoint lengths and slots must be supplied together")
    if checkpoint_lengths is not None:
        if len(checkpoint_lengths) != len(lengths) or len(checkpoint_slots) != len(
            lengths
        ):
            raise ValueError("Expected one checkpoint length and slot per sequence")
        if any(end > length for end, length in zip(checkpoint_lengths, lengths)):
            raise ValueError("Checkpoint extends past its sequence")

    num_tokens = sum(lengths)
    padding = -num_tokens % 128
    packed_lengths = lengths + ([padding] if padding else [])
    seq_ids, chunks, offsets, cumulative = [], [], [], [0]
    endpoints = []
    initial_rows = []
    start = 0
    for row, length in enumerate(packed_lengths):
        seq_ids.extend([row] * length)
        end = (
            checkpoint_lengths[row]
            if checkpoint_lengths is not None and row < len(lengths)
            else -1
        )
        checkpoint = start + end if end >= 0 else -1
        # Native capture happens only at logical-segment ends. Split an
        # interior checkpoint into two segments of the SAME sequence, so
        # state carries through without resetting or launching a second scan.
        # An inactive row must have endpoint -1. Physical slot IDs may remain
        # on device; never synchronize just to inspect a destination.
        if end == 0:
            initial_rows.append(row)
        cursor = start
        while cursor < start + length:
            chunks.append(cursor // 128)
            offsets.append(cursor % 128)
            next_cursor = min((cursor // 128 + 1) * 128, start + length)
            if cursor < checkpoint < next_cursor:
                next_cursor = checkpoint
            cursor = next_cursor
        cumulative.append(len(chunks))
        if checkpoint_lengths is not None:
            endpoints.append(checkpoint if end > 0 else -1)
        start += length

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    slots = None
    if checkpoint_slots is not None:
        slots = (
            checkpoint_slots.to(device=device, dtype=torch.int32)
            if isinstance(checkpoint_slots, torch.Tensor)
            else tensor(checkpoint_slots)
        )
        if padding:
            slots = torch.cat((slots, slots.new_full((1,), -1)))
    initial_rows_tensor = torch.tensor(initial_rows, dtype=torch.long, device=device)
    return SSDPrefillMetadata(
        num_tokens=num_tokens,
        num_sequences=len(lengths),
        padding=padding,
        seq_idx=tensor(seq_ids).unsqueeze(0),
        chunk_indices=tensor(chunks),
        chunk_offsets=tensor(offsets),
        seq_chunk_cumsum=tensor(cumulative),
        checkpoint_token_indices=tensor(endpoints)
        if checkpoint_lengths is not None
        else None,
        checkpoint_state_slots=slots,
        initial_checkpoint_rows=initial_rows_tensor if initial_rows else None,
        initial_checkpoint_slots=slots.index_select(0, initial_rows_tensor).long()
        if initial_rows
        else None,
    )


def flashinfer_ssd_prefill(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    initial_states: torch.Tensor,
    metadata: SSDPrefillMetadata,
    out: torch.Tensor,
    checkpoint_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write token-major output/checkpoints and return real-sequence final states.

    Inputs follow ``ssd_combined_fwd``: x [1,T,H,64], B/C [1,T,G,128],
    initial states [N,H,64,128]. Caller masks cold initial states to zero. The
    checkpoint pool must be contiguous, with distinct destination slots. This
    adapter intentionally includes padding/output-copy costs in its benchmarks."""
    from flashinfer.mamba import ssd_combined_fwd

    if x.shape[0] != 1 or x.shape[1] != metadata.num_tokens:
        raise ValueError("Input extent does not match packed SSD metadata")
    if initial_states.shape[0] != metadata.num_sequences:
        raise ValueError("Expected one initial state per real sequence")
    if (checkpoint_states is None) != (metadata.checkpoint_token_indices is None):
        raise ValueError("Checkpoint metadata and storage must be supplied together")
    if metadata.initial_checkpoint_rows is not None:
        checkpoint_states.index_copy_(
            0,
            metadata.initial_checkpoint_slots,
            initial_states.index_select(0, metadata.initial_checkpoint_rows),
        )

    def pad_tokens(value):
        if not metadata.padding:
            return value
        zeros = value.new_zeros((1, metadata.padding, *value.shape[2:]))
        return torch.cat((value, zeros), dim=1)

    if metadata.padding:
        initial_states = torch.cat(
            (initial_states, initial_states.new_zeros((1, *initial_states.shape[1:]))),
            dim=0,
        )
    result, final_states = ssd_combined_fwd(
        pad_tokens(x),
        pad_tokens(dt),
        A,
        pad_tokens(B),
        pad_tokens(C),
        D=D.to(torch.bfloat16) if D is not None else None,
        dt_bias=dt_bias,
        dt_softplus=True,
        # Upstream Cake SSD uses exact scan with FP16 processed delta. Keep
        # normal nonnegative step sizes; no clamp-based dispatch workaround.
        dt_limit=(0.0, float("inf")),
        initial_states=initial_states,
        seq_idx=metadata.seq_idx,
        chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
        seq_chunk_cumsum=metadata.seq_chunk_cumsum,
        checkpoint_token_indices=metadata.checkpoint_token_indices,
        checkpoint_state_slots=metadata.checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
        return_final_states=True,
    )
    out.copy_(result[:, : metadata.num_tokens])
    return final_states[: metadata.num_sequences]
