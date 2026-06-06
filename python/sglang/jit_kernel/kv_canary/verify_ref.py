from __future__ import annotations

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.consts import splitmix64, splitmix64_mix3
from sglang.jit_kernel.kv_canary.verify import (
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
)

_U64_MASK: int = (1 << 64) - 1
_I64_SIGN_BIT: int = 1 << 63


def launch_canary_verify_kernel_torch_reference(
    *,
    context: VerifyOrWriteContext,
    plan: VerifyPlan,
    check_verify_expected_token: bool,
) -> None:
    canary_buf = context.canary_buf
    kernel_kind = context.kernel_kind
    violation_ring = context.violation_ring
    violation_write_index = context.violation_write_index
    slot_run_counter = context.slot_run_counter
    kernel_run_counter = context.kernel_run_counter
    real_kv_sources = context.real_kv_sources
    real_kv_hash_mode = context.real_kv_hash_mode

    work_device = torch.device("cpu")

    kernel_run_counter.add_(1)

    enable = int(plan.enable.detach().to("cpu").item())
    if enable == 0:
        return

    num_valid = int(
        plan.verify_slot_indices.new_empty(()).copy_(plan.verify_num_valid[0]).item()
    )
    capacity = int(plan.verify_slot_indices.shape[0])
    active = max(0, min(num_valid, capacity))
    if active <= 0:
        return

    slot_indices_host = plan.verify_slot_indices[:active].to(
        device=work_device, dtype=torch.int64
    )
    if check_verify_expected_token:
        expected_input_ids_host = plan.verify_expected_tokens[:active].to(
            device=work_device, dtype=torch.int64
        )
    else:
        expected_input_ids_host = torch.full(
            (active,), -1, dtype=torch.int64, device=work_device
        )
    expected_positions_host = plan.verify_expected_positions[:active].to(
        device=work_device, dtype=torch.int64
    )
    prev_slot_indices_host = plan.verify_prev_slot_indices[:active].to(
        device=work_device, dtype=torch.int64
    )

    slot_run_counter.add_(active)

    kept_slots: list[int] = []
    kept_expected_positions: list[int] = []
    kept_expected_input_ids: list[int] = []
    kept_prev_slots: list[int] = []
    for k in range(active):
        s = int(slot_indices_host[k].item())
        # Skip SGLang's padded-token dummy KV slot so unfilled req_to_token entries (zero-initialized) do not
        # produce spurious chain_hash / position violations.
        if s != consts.TOKEN_TO_KV_SLOT_PADDING:
            kept_slots.append(s)
            kept_expected_positions.append(int(expected_positions_host[k].item()))
            kept_expected_input_ids.append(int(expected_input_ids_host[k].item()))
            kept_prev_slots.append(int(prev_slot_indices_host[k].item()))
    active = len(kept_slots)
    if active <= 0:
        return
    slot_indices_list: list[int] = kept_slots
    expected_positions_list: list[int] = kept_expected_positions
    expected_input_ids_list: list[int] = kept_expected_input_ids
    prev_slot_indices_list: list[int] = kept_prev_slots

    buf_i64 = canary_buf.detach().to(device=work_device).contiguous().view(torch.int64)
    slot_stride_i64 = int(buf_i64.shape[1])
    if slot_stride_i64 < 4:
        raise ValueError(
            f"kv-canary: canary_buf slot stride must hold at least 4 int64 fields, got {slot_stride_i64}"
        )

    violation_rows: list[list[int]] = []

    for k in range(active):
        slot_idx = slot_indices_list[k]
        expected_position = expected_positions_list[k]
        expected_input_id = expected_input_ids_list[k]
        prev_slot = prev_slot_indices_list[k]

        stored_token = int(buf_i64[slot_idx, consts.CANARY_FIELD_TOKEN].item())
        stored_position = int(buf_i64[slot_idx, consts.CANARY_FIELD_POSITION].item())
        stored_chain_hash = int(buf_i64[slot_idx, consts.CANARY_FIELD_PREV_HASH].item())
        stored_real_kv_hash = int(
            buf_i64[slot_idx, consts.CANARY_FIELD_REAL_KV_HASH].item()
        )

        prev_reachable = prev_slot != consts.TOKEN_TO_KV_SLOT_PADDING
        if prev_reachable:
            expected_chain_hash = _to_signed_int64(
                compute_slot_hash(buf_i64, prev_slot)
            )
        else:
            expected_chain_hash = stored_chain_hash

        expected_real_kv_hash_u64 = _compute_real_kv_hash_scalar(
            slot_idx=slot_idx,
            real_kv_sources=real_kv_sources,
            real_kv_hash_mode=real_kv_hash_mode,
            work_device=work_device,
        )
        expected_real_kv_hash = _to_signed_int64(expected_real_kv_hash_u64)

        fail_reason = consts.FailReason(0)
        if prev_reachable and stored_chain_hash != expected_chain_hash:
            fail_reason |= consts.FailReason.VERIFY_CHAIN_HASH_MISMATCH
        if check_verify_expected_token:
            if expected_input_id != -1 and stored_token != expected_input_id:
                fail_reason |= consts.FailReason.VERIFY_TOKEN_MISMATCH
        if stored_position != expected_position:
            fail_reason |= consts.FailReason.VERIFY_POSITION_MISMATCH
        if stored_real_kv_hash != expected_real_kv_hash:
            fail_reason |= consts.FailReason.VERIFY_REAL_KV_HASH_MISMATCH

        if fail_reason != consts.FailReason(0):
            row = [0] * consts.VIOLATION_FIELDS
            row[consts.VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
            row[consts.VIOLATION_FIELD_SLOT_IDX] = slot_idx
            row[consts.VIOLATION_FIELD_POSITION] = stored_position
            row[consts.VIOLATION_FIELD_STORED_TOKEN] = stored_token
            row[consts.VIOLATION_FIELD_EXPECTED_TOKEN] = expected_input_id
            row[consts.VIOLATION_FIELD_STORED_CHAIN_HASH] = stored_chain_hash
            row[consts.VIOLATION_FIELD_EXPECTED_AUX] = expected_chain_hash
            row[consts.VIOLATION_FIELD_FAIL_REASON_BITS] = int(fail_reason)
            violation_rows.append(row)

    if len(violation_rows) == 0:
        return

    num_new_violations = len(violation_rows)
    base_idx = int(
        violation_write_index.new_empty(()).copy_(violation_write_index[0]).item()
    )
    ring_capacity = int(violation_ring.shape[0])

    new_rows = torch.zeros(
        (num_new_violations, consts.VIOLATION_FIELDS), dtype=torch.int64
    )
    for v, row in enumerate(violation_rows):
        for f in range(consts.VIOLATION_FIELDS):
            new_rows[v, f] = row[f]

    write_count_in_ring = max(0, min(num_new_violations, ring_capacity - base_idx))
    if write_count_in_ring > 0:
        ring_host = violation_ring.detach().to(device=work_device)
        ring_host[base_idx : base_idx + write_count_in_ring, :] = new_rows[
            :write_count_in_ring, :
        ]
        violation_ring.copy_(ring_host.to(violation_ring.device))

    violation_write_index[0] = violation_write_index[0] + num_new_violations


def _to_signed_int64(value: int) -> int:
    value &= _U64_MASK
    if value >= _I64_SIGN_BIT:
        value -= 1 << 64
    return value


def compute_slot_hash(buf_i64: torch.Tensor, source_slot_idx: int) -> int:
    if source_slot_idx < 0:
        return splitmix64(consts.CANARY_CHAIN_ANCHOR)
    token = int(buf_i64[source_slot_idx, consts.CANARY_FIELD_TOKEN].item())
    position = int(buf_i64[source_slot_idx, consts.CANARY_FIELD_POSITION].item())
    prev_hash = int(buf_i64[source_slot_idx, consts.CANARY_FIELD_PREV_HASH].item())
    return splitmix64_mix3(prev_hash, token, position)


def _compute_real_kv_hash_scalar(
    *,
    slot_idx: int,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: consts.RealKvHashMode,
    work_device: torch.device,
) -> int:
    mode = int(real_kv_hash_mode)
    if mode == int(consts.RealKvHashMode.NONE) or len(real_kv_sources) == 0:
        return 0

    acc: int = 0

    for source in real_kv_sources:
        page_size = source.page_size
        num_bytes_per_token = source.num_bytes_per_token
        read_bytes = source.read_bytes
        tensor_u8 = (
            source.tensor.detach().to(device=work_device).contiguous().view(torch.uint8)
        )

        row = slot_idx // page_size
        col_within_page = slot_idx % page_size
        col_start = col_within_page * num_bytes_per_token

        effective_read_bytes = (
            16 if mode == int(consts.RealKvHashMode.PARTIAL) else read_bytes
        )
        raw_bytes: list[int] = []
        for b in range(effective_read_bytes):
            raw_bytes.append(int(tensor_u8[row, col_start + b].item()))

        source_hash = _splitmix64_fold_bytes_scalar(raw_bytes=raw_bytes)

        combined = acc ^ source_hash
        acc = splitmix64(combined)

    return acc


def _splitmix64_fold_bytes_scalar(*, raw_bytes: list[int]) -> int:
    read_bytes = len(raw_bytes)
    pad = (8 - read_bytes % 8) % 8
    padded = raw_bytes + [0] * pad
    num_words = len(padded) // 8

    acc: int = 0
    for w in range(num_words):
        word: int = 0
        for k in range(8):
            word |= padded[w * 8 + k] << (8 * k)
        word &= _U64_MASK
        acc = splitmix64(acc ^ word)

    return acc
