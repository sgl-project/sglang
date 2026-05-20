"""Torch reference implementation of canary_write_step.

Per-entry Python for-loops + scalar ops. fb_out_cache_loc is consumed opaquely (SWA translation is the
caller's responsibility); entries with slot < 0 are skipped. When pseudo_mode == ON, mismatches between
actual fb_input_ids[i] / fb_positions[i] and the caller-supplied expected tensors record a violation but the
chain still advances on the actual values.
"""

from __future__ import annotations

import torch

from sglang.jit_kernel.kv_canary.verify import (
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
from sglang.jit_kernel.kv_canary.verify_ref import (
    _compute_real_kv_hash_scalar,
    _to_signed_int64,
    splitmix64,
)
from sglang.jit_kernel.kv_canary.write import (
    _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH,
    _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH,
    CanaryPseudoMode,
    WritePlan,
)

_U64_MASK: int = (1 << 64) - 1

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
    """Torch reference for :func:`canary_write_step`. Same signature & byte-equal semantics."""
    work_device = torch.device("cpu")

    kernel_run_counter.add_(1)

    num_valid_reqs = int(plan.write_num_valid_reqs.detach().to("cpu").item())
    req_capacity = int(plan.write_seed_slot_indices.shape[0])
    active_reqs = max(0, min(num_valid_reqs, req_capacity))
    if active_reqs <= 0:
        return

    write_offsets_host = plan.write_offsets.detach().to(
        device=work_device, dtype=torch.int64
    )
    seed_slot_indices_host = plan.write_seed_slot_indices[:active_reqs].to(
        device=work_device, dtype=torch.int64
    )
    fb_input_ids_host = fb_input_ids.detach().to(device=work_device, dtype=torch.int64)
    fb_positions_host = fb_positions.detach().to(device=work_device, dtype=torch.int64)
    fb_out_cache_loc_host = fb_out_cache_loc.detach().to(
        device=work_device, dtype=torch.int64
    )

    total_entries = int(write_offsets_host[active_reqs].item())
    if total_entries <= 0:
        return

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

    chain_anchor_u64 = splitmix64(CANARY_CHAIN_ANCHOR)

    pseudo_mode_on = int(pseudo_mode) != int(CanaryPseudoMode.OFF)
    pseudo_expected_tokens_host = pseudo_expected_tokens.detach().to(
        device=work_device, dtype=torch.int64
    )
    pseudo_expected_positions_host = pseudo_expected_positions.detach().to(
        device=work_device, dtype=torch.int64
    )

    violation_rows: list[list[int]] = []
    total_slots_written = 0

    for r in range(active_reqs):
        entry_start = int(write_offsets_host[r].item())
        entry_end = int(write_offsets_host[r + 1].item())
        entry_count = entry_end - entry_start
        if entry_count <= 0:
            continue

        seed_slot = int(seed_slot_indices_host[r].item())
        if seed_slot < 0:
            running_prev_hash = chain_anchor_u64
        else:
            seed_prev_hash_signed = int(buf_i64[seed_slot, _FIELD_PREV_HASH].item())
            seed_token_signed = int(buf_i64[seed_slot, _FIELD_TOKEN].item())
            seed_position_signed = int(buf_i64[seed_slot, _FIELD_POSITION].item())
            seed_real_kv_signed = int(buf_i64[seed_slot, _FIELD_REAL_KV_HASH].item())
            seed_combined = (
                (seed_prev_hash_signed & _U64_MASK)
                ^ (seed_token_signed & _U64_MASK)
                ^ (seed_position_signed & _U64_MASK)
                ^ (seed_real_kv_signed & _U64_MASK)
            )
            running_prev_hash = splitmix64(seed_combined)

        for j in range(entry_count):
            i = entry_start + j
            slot = int(fb_out_cache_loc_host[i].item())
            if slot < 0:
                continue
            token = int(fb_input_ids_host[i].item())
            position = int(fb_positions_host[i].item())

            real_kv_hash_u64 = _compute_real_kv_hash_scalar(
                slot_idx=slot,
                real_kv_sources=real_kv_sources,
                real_kv_hash_mode=real_kv_hash_mode,
                work_device=work_device,
            )

            if pseudo_mode_on:
                mismatch_bits = 0
                expected_token = int(pseudo_expected_tokens_host[i].item())
                expected_position = int(pseudo_expected_positions_host[i].item())
                if token != expected_token:
                    mismatch_bits |= _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH
                if position != expected_position:
                    mismatch_bits |= _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH
                if mismatch_bits != 0:
                    row = [0] * VIOLATION_FIELDS
                    row[_VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
                    row[_VIOLATION_FIELD_SLOT_IDX] = slot
                    row[_VIOLATION_FIELD_POSITION] = position
                    row[_VIOLATION_FIELD_STORED_TOKEN] = token
                    row[_VIOLATION_FIELD_EXPECTED_TOKEN] = expected_token
                    row[_VIOLATION_FIELD_STORED_CHAIN_HASH] = _to_signed_int64(
                        running_prev_hash
                    )
                    row[_VIOLATION_FIELD_EXPECTED_AUX] = expected_position
                    row[_VIOLATION_FIELD_FAIL_REASON_BITS] = mismatch_bits
                    violation_rows.append(row)

            buf_i64[slot, _FIELD_TOKEN] = token
            buf_i64[slot, _FIELD_POSITION] = position
            buf_i64[slot, _FIELD_PREV_HASH] = _to_signed_int64(running_prev_hash)
            buf_i64[slot, _FIELD_REAL_KV_HASH] = _to_signed_int64(real_kv_hash_u64)

            combined = (
                running_prev_hash
                ^ (token & _U64_MASK)
                ^ (position & _U64_MASK)
                ^ real_kv_hash_u64
            )
            running_prev_hash = splitmix64(combined)

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
