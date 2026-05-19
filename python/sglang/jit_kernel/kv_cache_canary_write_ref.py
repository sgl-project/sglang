"""Torch reference implementation of canary_write_step.

Vectorised across write reqs; the per-req chain loop stays serial (a chain is intrinsically sequential — each
step's hash depends on the previous step's full slot state). SWA translation is inline (no pre-pass). When
pseudo_mode == ON, mismatches between actual fb_input_ids[i] / fb_positions[i] and the caller-supplied expected
tensors record a violation but the chain still advances on the actual values.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_cache_canary_verify import (
    _VIOLATION_FIELD_EXPECTED_AUX,
    _VIOLATION_FIELD_EXPECTED_TOKEN,
    _VIOLATION_FIELD_FAIL_REASON_BITS,
    _VIOLATION_FIELD_KERNEL_KIND,
    _VIOLATION_FIELD_POSITION,
    _VIOLATION_FIELD_SLOT_IDX,
    _VIOLATION_FIELD_STORED_CHAIN_HASH,
    _VIOLATION_FIELD_STORED_TOKEN,
    CANARY_CHAIN_ANCHOR,
    VIOLATION_FIELDS,
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
)
from sglang.jit_kernel.kv_cache_canary_verify_ref import (
    _compute_real_kv_hash_vec,
    _splitmix64_finalize_vec,
    _splitmix64_python,
    _to_signed_int64,
)
from sglang.jit_kernel.kv_cache_canary_write import (
    _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH,
    _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH,
    CanaryPseudoMode,
    WritePlan,
)

# Canary slot field offsets within the 4-int64 layout. Kept in sync with the verify ref.
_FIELD_TOKEN: int = 0
_FIELD_POSITION: int = 1
_FIELD_PREV_HASH: int = 2
_FIELD_REAL_KV_HASH: int = 3


def canary_write_step_torch_reference(
    *,
    canary_buf: torch.Tensor,
    plan: WritePlan,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    kernel_kind: CanaryLaunchTag,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    violation_ring: torch.Tensor,
    violation_write_index: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
) -> None:
    """Torch reference for :func:`canary_write_step`. Same signature & byte-equal semantics.

    Vectorised across reqs (precomputes per-chain-step real_kv_hash + pseudo-mode mismatches up front); the
    chain advance itself stays serial because each step's prev_hash depends on the previous slot's stored
    fields. Violations are emitted in (req, j) order so the CUDA kernel can match by sorting its atomic-order
    outputs.
    """
    num_valid_reqs = int(plan.write_num_valid_reqs.detach().to("cpu").item())
    req_capacity = int(plan.write_seed_slot_indices.shape[0])
    active_reqs = max(0, min(num_valid_reqs, req_capacity))

    kernel_run_counter.add_(1)
    if active_reqs <= 0:
        return

    work_device = torch.device("cpu")
    write_offsets = plan.write_offsets.detach().to(
        device=work_device, dtype=torch.int64
    )
    seed_slot_indices = plan.write_seed_slot_indices[:active_reqs].to(
        device=work_device, dtype=torch.int64
    )
    fb_input_ids_host = fb_input_ids.detach().to(device=work_device, dtype=torch.int64)
    fb_positions_host = fb_positions.detach().to(device=work_device, dtype=torch.int64)
    fb_out_cache_loc_host = fb_out_cache_loc.detach().to(
        device=work_device, dtype=torch.int64
    )

    total_entries = int(write_offsets[active_reqs].item())
    if total_entries <= 0:
        return

    slot_indices_all = fb_out_cache_loc_host[:total_entries].clone()
    if full_to_swa_index_mapping is not None:
        lut = full_to_swa_index_mapping.detach().to(
            device=work_device, dtype=torch.int64
        )
        slot_indices_all = _swa_translate(slot_indices=slot_indices_all, lut=lut)

    tokens_all = fb_input_ids_host[:total_entries]
    positions_all = fb_positions_host[:total_entries]

    # Pre-compute real_kv_hash per write entry vectorised. The chain loop will pick the value up by index.
    real_kv_hashes_all = _compute_real_kv_hash_vec(
        slot_indices=slot_indices_all,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=real_kv_hash_mode,
        work_device=work_device,
    )

    # Pseudo-mode mismatches per entry (only used when pseudo_mode == ON).
    pseudo_mismatch_bits = _compute_pseudo_mismatch_bits(
        pseudo_mode=pseudo_mode,
        tokens_all=tokens_all,
        positions_all=positions_all,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        total_entries=total_entries,
        work_device=work_device,
    )

    buf_i64 = (
        canary_buf.detach()
        .to(device=work_device)
        .contiguous()
        .view(torch.int64)
        .clone()
    )
    slot_stride_i64 = int(buf_i64.shape[1])
    if slot_stride_i64 < 4:
        raise ValueError(
            f"kv-canary: canary_buf slot stride must hold at least 4 int64 fields, got {slot_stride_i64}"
        )

    chain_anchor_signed = _to_signed_int64(_splitmix64_python(CANARY_CHAIN_ANCHOR))

    # Vectorised seed-slot prev_hash gather. For seed < 0 use the chain anchor; else load the seed slot's 4
    # fields and splitmix64-mix4 them.
    seed_is_anchor = seed_slot_indices < 0
    safe_seed = torch.where(
        seed_is_anchor, torch.zeros_like(seed_slot_indices), seed_slot_indices
    )
    seed_prev_hashes = buf_i64[safe_seed, _FIELD_PREV_HASH]
    seed_tokens = buf_i64[safe_seed, _FIELD_TOKEN]
    seed_positions = buf_i64[safe_seed, _FIELD_POSITION]
    seed_real_kv_hashes = buf_i64[safe_seed, _FIELD_REAL_KV_HASH]
    initial_chain_from_seed = _splitmix64_finalize_vec(
        seed_prev_hashes ^ seed_tokens ^ seed_positions ^ seed_real_kv_hashes
    )
    initial_chain_hashes = torch.where(
        seed_is_anchor,
        torch.full_like(seed_prev_hashes, chain_anchor_signed),
        initial_chain_from_seed,
    )

    violation_rows: list[list[int]] = []
    total_slots_written = 0

    for r in range(active_reqs):
        entry_start = int(write_offsets[r].item())
        entry_end = int(write_offsets[r + 1].item())
        entry_count = entry_end - entry_start
        if entry_count <= 0:
            continue

        running_prev_hash = int(initial_chain_hashes[r].item()) & ((1 << 64) - 1)

        for j in range(entry_count):
            i = entry_start + j
            slot = int(slot_indices_all[i].item())
            token = int(tokens_all[i].item())
            position = int(positions_all[i].item())
            real_kv_hash_signed = int(real_kv_hashes_all[i].item())
            real_kv_hash_u64 = real_kv_hash_signed & ((1 << 64) - 1)

            mismatch_bits = int(pseudo_mismatch_bits[i].item())
            if mismatch_bits != 0:
                violation_rows.append(
                    _build_pseudo_violation_row(
                        kernel_kind=kernel_kind,
                        slot=slot,
                        token=token,
                        position=position,
                        pseudo_expected_tokens=pseudo_expected_tokens,
                        pseudo_expected_positions=pseudo_expected_positions,
                        entry_index=i,
                        mismatch_bits=mismatch_bits,
                        running_prev_hash=running_prev_hash,
                    )
                )

            buf_i64[slot, _FIELD_TOKEN] = token
            buf_i64[slot, _FIELD_POSITION] = position
            buf_i64[slot, _FIELD_PREV_HASH] = _to_signed_int64(running_prev_hash)
            buf_i64[slot, _FIELD_REAL_KV_HASH] = _to_signed_int64(real_kv_hash_u64)

            combined = (
                running_prev_hash
                ^ (token & ((1 << 64) - 1))
                ^ (position & ((1 << 64) - 1))
                ^ real_kv_hash_u64
            )
            running_prev_hash = _splitmix64_python(combined)

            total_slots_written += 1

    canary_buf.view(torch.int64).copy_(
        buf_i64.to(canary_buf.device).view(canary_buf.shape[0], slot_stride_i64)
    )

    slot_run_counter.add_(total_slots_written)

    if len(violation_rows) == 0:
        return

    base_idx = int(violation_write_index.detach().to("cpu").item())
    ring_capacity = int(violation_ring.shape[0])
    new_rows = torch.tensor(violation_rows, dtype=torch.int64, device=work_device)
    write_count_in_ring = max(0, min(len(violation_rows), ring_capacity - base_idx))
    if write_count_in_ring > 0:
        ring_host = violation_ring.detach().to(device=work_device)
        ring_host[base_idx : base_idx + write_count_in_ring, :] = new_rows[
            :write_count_in_ring, :
        ]
        violation_ring.copy_(ring_host.to(violation_ring.device))

    violation_write_index[0] = violation_write_index[0] + len(violation_rows)


def _swa_translate(*, slot_indices: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """SWA full-pool→swa-pool translation. -1 sentinel rows pass through unchanged."""
    lut_len = int(lut.shape[0])
    sentinel_mask = slot_indices < 0
    safe_idx = torch.where(sentinel_mask, torch.zeros_like(slot_indices), slot_indices)
    if lut_len > 0:
        oob_mask = safe_idx >= lut_len
        safe_idx = torch.where(
            oob_mask, torch.full_like(safe_idx, lut_len - 1), safe_idx
        )
    translated = lut[safe_idx]
    return torch.where(sentinel_mask, slot_indices, translated)


def _compute_pseudo_mismatch_bits(
    *,
    pseudo_mode: CanaryPseudoMode,
    tokens_all: torch.Tensor,
    positions_all: torch.Tensor,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    total_entries: int,
    work_device: torch.device,
) -> torch.Tensor:
    """Vectorised per-entry pseudo-mode mismatch bitfield. Returns int64 [total_entries].

    OFF mode → all zeros (kernel never reads the pseudo_* tensors). ON mode → fb_input_ids vs
    pseudo_expected_tokens and fb_positions vs pseudo_expected_positions, OR'd into a bitfield per entry.
    """
    if int(pseudo_mode) == int(CanaryPseudoMode.OFF):
        return torch.zeros(total_entries, dtype=torch.int64, device=work_device)

    expected_tokens = pseudo_expected_tokens.detach().to(
        device=work_device, dtype=torch.int64
    )[:total_entries]
    expected_positions = pseudo_expected_positions.detach().to(
        device=work_device, dtype=torch.int64
    )[:total_entries]
    bits = torch.zeros(total_entries, dtype=torch.int64, device=work_device)
    bits |= torch.where(
        tokens_all != expected_tokens,
        torch.full_like(bits, _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH),
        torch.zeros_like(bits),
    )
    bits |= torch.where(
        positions_all != expected_positions,
        torch.full_like(bits, _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH),
        torch.zeros_like(bits),
    )
    return bits


def _build_pseudo_violation_row(
    *,
    kernel_kind: CanaryLaunchTag,
    slot: int,
    token: int,
    position: int,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    entry_index: int,
    mismatch_bits: int,
    running_prev_hash: int,
) -> list[int]:
    expected_token = int(pseudo_expected_tokens.detach().to("cpu")[entry_index].item())
    expected_position = int(
        pseudo_expected_positions.detach().to("cpu")[entry_index].item()
    )
    row = [0] * VIOLATION_FIELDS
    row[_VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
    row[_VIOLATION_FIELD_SLOT_IDX] = slot
    row[_VIOLATION_FIELD_POSITION] = position
    row[_VIOLATION_FIELD_STORED_TOKEN] = token
    row[_VIOLATION_FIELD_EXPECTED_TOKEN] = expected_token
    # Write path: stored_chain_hash carries the running prev_hash about to be written; expected stores the
    # caller-supplied expected position (no chain-hash oracle on the write side because the chain is being
    # produced, not verified).
    row[_VIOLATION_FIELD_STORED_CHAIN_HASH] = _to_signed_int64(running_prev_hash)
    row[_VIOLATION_FIELD_EXPECTED_AUX] = expected_position
    row[_VIOLATION_FIELD_FAIL_REASON_BITS] = mismatch_bits
    return row
