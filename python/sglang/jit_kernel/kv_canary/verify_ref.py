from __future__ import annotations

import torch

from sglang.jit_kernel.kv_canary.verify import (
    _FAIL_REASON_BIT_CHAIN_HASH,
    _FAIL_REASON_BIT_POSITION,
    _FAIL_REASON_BIT_REAL_KV_HASH,
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
    VerifyPlan,
)

_U64_MASK: int = (1 << 64) - 1
_I64_SIGN_BIT: int = 1 << 63

# Canary slot field offsets within the 4-int64 layout.
_FIELD_TOKEN: int = 0
_FIELD_POSITION: int = 1
_FIELD_PREV_HASH: int = 2
_FIELD_REAL_KV_HASH: int = 3


def canary_verify_step_torch_reference(
    *,
    canary_buf: torch.Tensor,
    plan: VerifyPlan,
    kernel_kind: CanaryLaunchTag,
    violation_ring: torch.Tensor,
    violation_write_index: torch.Tensor,
    slot_run_counter: torch.Tensor,
    kernel_run_counter: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
) -> None:
    work_device = torch.device("cpu")

    kernel_run_counter.add_(1)

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
    expected_positions_host = plan.verify_positions[:active].to(
        device=work_device, dtype=torch.int64
    )
    prev_slot_indices_host = plan.verify_prev_slot_indices[:active].to(
        device=work_device, dtype=torch.int64
    )

    slot_run_counter.add_(active)

    kept_slots: list[int] = []
    kept_expected_positions: list[int] = []
    kept_prev_slots: list[int] = []
    for k in range(active):
        s = int(slot_indices_host[k].item())
        if s != 0:
            kept_slots.append(s)
            kept_expected_positions.append(int(expected_positions_host[k].item()))
            kept_prev_slots.append(int(prev_slot_indices_host[k].item()))
    active = len(kept_slots)
    if active <= 0:
        return
    slot_indices_list: list[int] = kept_slots
    expected_positions_list: list[int] = kept_expected_positions
    prev_slot_indices_list: list[int] = kept_prev_slots

    buf_i64 = canary_buf.detach().to(device=work_device).contiguous().view(torch.int64)
    slot_stride_i64 = int(buf_i64.shape[1])
    if slot_stride_i64 < 4:
        raise ValueError(
            f"kv-canary: canary_buf slot stride must hold at least 4 int64 fields, got {slot_stride_i64}"
        )

    chain_anchor_signed = _to_signed_int64(splitmix64(CANARY_CHAIN_ANCHOR))

    violation_rows: list[list[int]] = []

    for k in range(active):
        slot_idx = slot_indices_list[k]
        expected_position = expected_positions_list[k]
        prev_slot = prev_slot_indices_list[k]

        stored_token = int(buf_i64[slot_idx, _FIELD_TOKEN].item())
        stored_position = int(buf_i64[slot_idx, _FIELD_POSITION].item())
        stored_chain_hash = int(buf_i64[slot_idx, _FIELD_PREV_HASH].item())
        stored_real_kv_hash = int(buf_i64[slot_idx, _FIELD_REAL_KV_HASH].item())

        if prev_slot < 0:
            expected_chain_hash = chain_anchor_signed
        else:
            prev_ph = int(buf_i64[prev_slot, _FIELD_PREV_HASH].item())
            prev_tok = int(buf_i64[prev_slot, _FIELD_TOKEN].item())
            prev_pos = int(buf_i64[prev_slot, _FIELD_POSITION].item())
            prev_rkv = int(buf_i64[prev_slot, _FIELD_REAL_KV_HASH].item())
            combined = (
                (prev_ph & _U64_MASK)
                ^ (prev_tok & _U64_MASK)
                ^ (prev_pos & _U64_MASK)
                ^ (prev_rkv & _U64_MASK)
            )
            expected_chain_hash = _to_signed_int64(splitmix64(combined))

        expected_real_kv_hash_u64 = _compute_real_kv_hash_scalar(
            slot_idx=slot_idx,
            real_kv_sources=real_kv_sources,
            real_kv_hash_mode=real_kv_hash_mode,
            work_device=work_device,
        )
        expected_real_kv_hash = _to_signed_int64(expected_real_kv_hash_u64)

        fail_reason = 0
        if stored_chain_hash != expected_chain_hash:
            fail_reason |= _FAIL_REASON_BIT_CHAIN_HASH
        if stored_position != expected_position:
            fail_reason |= _FAIL_REASON_BIT_POSITION
        if stored_real_kv_hash != expected_real_kv_hash:
            fail_reason |= _FAIL_REASON_BIT_REAL_KV_HASH

        if fail_reason != 0:
            row = [0] * VIOLATION_FIELDS
            row[_VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
            row[_VIOLATION_FIELD_SLOT_IDX] = slot_idx
            row[_VIOLATION_FIELD_POSITION] = stored_position
            row[_VIOLATION_FIELD_STORED_TOKEN] = stored_token
            row[_VIOLATION_FIELD_EXPECTED_TOKEN] = 0
            row[_VIOLATION_FIELD_STORED_CHAIN_HASH] = stored_chain_hash
            row[_VIOLATION_FIELD_EXPECTED_AUX] = expected_chain_hash
            row[_VIOLATION_FIELD_FAIL_REASON_BITS] = fail_reason
            violation_rows.append(row)

    if len(violation_rows) == 0:
        return

    num_new_violations = len(violation_rows)
    base_idx = int(
        violation_write_index.new_empty(()).copy_(violation_write_index[0]).item()
    )
    ring_capacity = int(violation_ring.shape[0])

    new_rows = torch.zeros((num_new_violations, VIOLATION_FIELDS), dtype=torch.int64)
    for v, row in enumerate(violation_rows):
        for f in range(VIOLATION_FIELDS):
            new_rows[v, f] = row[f]

    write_count_in_ring = max(0, min(num_new_violations, ring_capacity - base_idx))
    if write_count_in_ring > 0:
        ring_host = violation_ring.detach().to(device=work_device)
        ring_host[base_idx : base_idx + write_count_in_ring, :] = new_rows[
            :write_count_in_ring, :
        ]
        violation_ring.copy_(ring_host.to(violation_ring.device))

    violation_write_index[0] = violation_write_index[0] + num_new_violations


def splitmix64(value: int) -> int:
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def _to_signed_int64(value: int) -> int:
    value &= _U64_MASK
    if value >= _I64_SIGN_BIT:
        value -= 1 << 64
    return value


def _compute_real_kv_hash_scalar(
    *,
    slot_idx: int,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    work_device: torch.device,
) -> int:
    """Compute one uint64 real-KV fingerprint for a single slot index.

    Off mode or no sources -> 0. PARTIAL mode -> splitmix64-fold first min(16, read_bytes) bytes
    (little-endian word-pack, at most 2 words). ALL mode -> splitmix64-fold every 8-byte little-endian
    word (zero-padded). Each source's contribution is mixed into the running accumulator via
    splitmix64(acc ^ source_hash). When read_bytes <= 16, PARTIAL and ALL produce identical hashes.
    """
    mode = int(real_kv_hash_mode)
    if mode == int(RealKvHashMode.OFF) or len(real_kv_sources) == 0:
        return 0

    acc: int = 0

    for source in real_kv_sources:
        if source.read_bytes <= 0:
            continue

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
            min(16, read_bytes) if mode == int(RealKvHashMode.PARTIAL) else read_bytes
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


# The vectorised helpers below are kept because write_ref.py (and the real-KV path in verify) originally
# imported them. They remain correct but are no longer called from the main verify loop above.


def _splitmix64_finalize_vec(words: torch.Tensor) -> torch.Tensor:
    x = words
    x = _xor_shift_mul(x, 30, _to_signed_int64(0xBF58476D1CE4E5B9))
    x = _xor_shift_mul(x, 27, _to_signed_int64(0x94D049BB133111EB))
    x = x ^ _logical_shr(x, 31)
    return x


def _xor_shift_mul(x: torch.Tensor, shift: int, multiplier_signed: int) -> torch.Tensor:
    shifted = _logical_shr(x, shift)
    mixed = x ^ shifted
    return mixed * multiplier_signed


def _logical_shr(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    mask = (1 << (64 - shift)) - 1
    return (x >> shift) & mask


def _splitmix64_mix4_vec(
    prev_hash: torch.Tensor,
    token: torch.Tensor,
    position: torch.Tensor,
    real_kv_hash: torch.Tensor,
) -> torch.Tensor:
    combined = prev_hash ^ token ^ position ^ real_kv_hash
    return _splitmix64_finalize_vec(combined)


def _compute_real_kv_hash_vec(
    *,
    slot_indices: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    work_device: torch.device,
) -> torch.Tensor:
    num_entries = int(slot_indices.shape[0])
    acc = torch.zeros(num_entries, dtype=torch.int64, device=work_device)
    mode = int(real_kv_hash_mode)
    if mode == int(RealKvHashMode.OFF) or len(real_kv_sources) == 0:
        return acc

    for k in range(num_entries):
        slot_idx = int(slot_indices[k].item())
        hash_u64 = _compute_real_kv_hash_scalar(
            slot_idx=slot_idx,
            real_kv_sources=real_kv_sources,
            real_kv_hash_mode=real_kv_hash_mode,
            work_device=work_device,
        )
        acc[k] = _to_signed_int64(hash_u64)

    return acc
