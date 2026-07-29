from __future__ import annotations

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.verify import (
    VerifyOrWriteContext,
)
from sglang.jit_kernel.kv_canary.verify_ref import (
    _compute_real_kv_hash_scalar,
    _to_signed_int64,
    compute_slot_hash,
    splitmix64_mix3,
)
from sglang.jit_kernel.kv_canary.write import WritePlan


def launch_canary_write_kernel_torch_reference(
    *,
    context: VerifyOrWriteContext,
    plan: WritePlan,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    enable_write_input_assert: bool,
    expected_input_tokens: torch.Tensor | None,
    expected_input_positions: torch.Tensor | None,
) -> None:
    canary_buf = context.canary_buf
    kernel_kind = context.kernel_kind
    violation_ring = context.violation_ring
    violation_write_index = context.violation_write_index
    slot_run_counter = context.slot_run_counter
    kernel_run_counter = context.kernel_run_counter
    real_kv_sources = context.real_kv_sources
    real_kv_hash_mode = context.real_kv_hash_mode
    enable_chain_position_assert_value = int(
        context.enable_chain_position_assert.detach().to("cpu").item()
    )

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
    input_ids_host = input_ids.detach().to(device=work_device, dtype=torch.int64)
    positions_host = positions.detach().to(device=work_device, dtype=torch.int64)
    out_cache_loc_host = out_cache_loc.detach().to(
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

    if enable_write_input_assert:
        if expected_input_tokens is None or expected_input_positions is None:
            raise ValueError(
                "kv-canary: expected input tensors are required when enable_write_input_assert=True"
            )
        expected_input_tokens_host = expected_input_tokens.detach().to(
            device=work_device, dtype=torch.int64
        )
        expected_input_positions_host = expected_input_positions.detach().to(
            device=work_device, dtype=torch.int64
        )
    else:
        if expected_input_tokens is not None or expected_input_positions is not None:
            raise ValueError(
                "kv-canary: expected input tensors must be None when enable_write_input_assert=False"
            )
        expected_input_tokens_host = None
        expected_input_positions_host = None

    violation_rows: list[list[int]] = []
    total_slots_written = 0

    for r in range(active_reqs):
        entry_start = int(write_offsets_host[r].item())
        entry_end = int(write_offsets_host[r + 1].item())
        entry_count = entry_end - entry_start
        if entry_count <= 0:
            continue

        seed_slot = int(seed_slot_indices_host[r].item())
        running_prev_hash = compute_slot_hash(buf_i64, seed_slot)

        do_chain_position_assert = (seed_slot >= 0) and (
            enable_chain_position_assert_value != 0
        )
        if do_chain_position_assert:
            running_prev_position = int(
                buf_i64[seed_slot, consts.CANARY_FIELD_POSITION].item()
            )
        else:
            running_prev_position = 0

        for entry_offset in range(entry_count):
            entry_idx = entry_start + entry_offset
            slot = int(out_cache_loc_host[entry_idx].item())
            if slot < 0:
                continue
            token = int(input_ids_host[entry_idx].item())
            position = int(positions_host[entry_idx].item())

            real_kv_hash_u64 = _compute_real_kv_hash_scalar(
                slot_idx=slot,
                real_kv_sources=real_kv_sources,
                real_kv_hash_mode=real_kv_hash_mode,
                work_device=work_device,
            )

            if enable_write_input_assert:
                assert expected_input_tokens_host is not None
                assert expected_input_positions_host is not None
                mismatch_bits = consts.FailReason(0)
                expected_token = int(expected_input_tokens_host[entry_idx].item())
                expected_position = int(expected_input_positions_host[entry_idx].item())
                if token != expected_token:
                    mismatch_bits |= consts.FailReason.WRITE_TOKEN_MISMATCH
                if position != expected_position:
                    mismatch_bits |= consts.FailReason.WRITE_POSITION_MISMATCH
                if mismatch_bits != consts.FailReason(0):
                    row = [0] * consts.VIOLATION_FIELDS
                    row[consts.VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
                    row[consts.VIOLATION_FIELD_SLOT_IDX] = slot
                    row[consts.VIOLATION_FIELD_POSITION] = position
                    row[consts.VIOLATION_FIELD_STORED_TOKEN] = token
                    row[consts.VIOLATION_FIELD_EXPECTED_TOKEN] = expected_token
                    row[consts.VIOLATION_FIELD_STORED_CHAIN_HASH] = _to_signed_int64(
                        running_prev_hash
                    )
                    row[consts.VIOLATION_FIELD_EXPECTED_AUX] = expected_position
                    row[consts.VIOLATION_FIELD_FAIL_REASON_BITS] = int(mismatch_bits)
                    violation_rows.append(row)

            if do_chain_position_assert:
                expected_position_chain = running_prev_position + 1
                if position != expected_position_chain:
                    row = [0] * consts.VIOLATION_FIELDS
                    row[consts.VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
                    row[consts.VIOLATION_FIELD_SLOT_IDX] = slot
                    row[consts.VIOLATION_FIELD_POSITION] = position
                    row[consts.VIOLATION_FIELD_STORED_TOKEN] = token
                    row[consts.VIOLATION_FIELD_EXPECTED_TOKEN] = token
                    row[consts.VIOLATION_FIELD_STORED_CHAIN_HASH] = _to_signed_int64(
                        running_prev_hash
                    )
                    row[consts.VIOLATION_FIELD_EXPECTED_AUX] = expected_position_chain
                    row[consts.VIOLATION_FIELD_FAIL_REASON_BITS] = int(
                        consts.FailReason.WRITE_POSITION_MISMATCH
                    )
                    violation_rows.append(row)
                running_prev_position = position

            buf_i64[slot, consts.CANARY_FIELD_TOKEN] = token
            buf_i64[slot, consts.CANARY_FIELD_POSITION] = position
            buf_i64[slot, consts.CANARY_FIELD_PREV_HASH] = _to_signed_int64(
                running_prev_hash
            )
            buf_i64[slot, consts.CANARY_FIELD_REAL_KV_HASH] = _to_signed_int64(
                real_kv_hash_u64
            )

            running_prev_hash = splitmix64_mix3(running_prev_hash, token, position)

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
