"""Torch reference implementation of canary_verify_step.

Vectorised across active verify entries; no python for-loops over the entry count. The CUDA kernel that
eventually lands must reproduce this output byte-for-byte (violation_ring contents, write-index increment,
counter increments).

The violation row schema (8 int64 fields in fixed order) is documented at module scope in
kv_canary/verify.py via the _VIOLATION_FIELD_* constants; readers should consume those names rather than
indexing positionally.
"""

from __future__ import annotations

from typing import Optional

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
    pad_sentinel_slot: int = -1,
) -> None:
    """Torch reference for :func:`canary_verify_step`. Same signature & byte-equal semantics.

    Vectorised across active verify entries. Violations are emitted in a deterministic order (entry index
    ascending) so the CUDA kernel can match by sorting its atomic-order outputs.
    """
    work_device = torch.device("cpu")

    kernel_run_counter.add_(1)
    result = _load_active_verify_entries(
        plan=plan,
        work_device=work_device,
        pad_sentinel_slot=pad_sentinel_slot,
        slot_run_counter=slot_run_counter,
    )
    if result is None:
        return
    slot_indices, expected_positions, prev_slot_indices, active = result

    (
        stored_tokens,
        stored_positions,
        stored_chain_hashes,
        stored_real_kv_hashes,
        prev_tokens,
        prev_positions,
        prev_chain_hashes,
        prev_real_kv_hashes,
        is_chain_head,
    ) = _load_stored_and_prev_fields(
        canary_buf=canary_buf,
        slot_indices=slot_indices,
        prev_slot_indices=prev_slot_indices,
        work_device=work_device,
    )

    expected_chain_hashes = _compute_expected_chain_hashes(
        prev_chain_hashes=prev_chain_hashes,
        prev_tokens=prev_tokens,
        prev_positions=prev_positions,
        prev_real_kv_hashes=prev_real_kv_hashes,
        is_chain_head=is_chain_head,
    )

    # real-KV hash mixin. Mode dispatch handles OFF (always 0), BIT (one bit per byte XOR-folded), and ALL
    # (splitmix64-fold of every byte). When no sources are supplied the mixin is also disabled.
    expected_real_kv_hashes = _compute_real_kv_hash_vec(
        slot_indices=slot_indices,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=real_kv_hash_mode,
        work_device=work_device,
    )

    fail_bits = _compute_verify_fail_bits(
        stored_chain_hashes=stored_chain_hashes,
        expected_chain_hashes=expected_chain_hashes,
        stored_positions=stored_positions,
        expected_positions=expected_positions,
        stored_real_kv_hashes=stored_real_kv_hashes,
        expected_real_kv_hashes=expected_real_kv_hashes,
    )

    _emit_verify_violations(
        fail_bits=fail_bits,
        slot_indices=slot_indices,
        stored_positions=stored_positions,
        stored_tokens=stored_tokens,
        stored_chain_hashes=stored_chain_hashes,
        expected_chain_hashes=expected_chain_hashes,
        kernel_kind=kernel_kind,
        violation_ring=violation_ring,
        violation_write_index=violation_write_index,
        work_device=work_device,
    )


def _load_active_verify_entries(
    *,
    plan: VerifyPlan,
    work_device: torch.device,
    pad_sentinel_slot: int,
    slot_run_counter: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    num_valid = int(
        plan.verify_slot_indices.new_empty(()).copy_(plan.verify_num_valid[0]).item()
    )
    capacity = int(plan.verify_slot_indices.shape[0])
    active = max(0, min(num_valid, capacity))

    if active <= 0:
        return None

    slot_indices = plan.verify_slot_indices[:active].to(
        device=work_device, dtype=torch.int64
    )
    expected_positions = plan.verify_positions[:active].to(
        device=work_device, dtype=torch.int64
    )
    prev_slot_indices = plan.verify_prev_slot_indices[:active].to(
        device=work_device, dtype=torch.int64
    )

    # slot_run_counter is bumped from the pre-skip active count (kernel computes it from is_active_entry,
    # which is fixed by tid < verify_num_valid before the pad-sentinel branch fires).
    slot_run_counter.add_(active)

    # Mirror the CUDA kernel's pad-sentinel skip: drop entries pointing at the reserved padding slot before
    # any canary_buf indexing so unfilled req_to_token positions (zero-init reads as slot 0 in sglang's pool)
    # do not produce spurious chain_hash/position violations.
    if pad_sentinel_slot >= 0:
        keep_mask = slot_indices != pad_sentinel_slot
        slot_indices = slot_indices[keep_mask]
        expected_positions = expected_positions[keep_mask]
        prev_slot_indices = prev_slot_indices[keep_mask]
        active = int(slot_indices.shape[0])
        if active <= 0:
            return None

    return slot_indices, expected_positions, prev_slot_indices, active


def _load_stored_and_prev_fields(
    *,
    canary_buf: torch.Tensor,
    slot_indices: torch.Tensor,
    prev_slot_indices: torch.Tensor,
    work_device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    buf_i64 = canary_buf.detach().to(device=work_device).contiguous().view(torch.int64)
    slot_stride_i64 = (
        int(buf_i64.shape[1]) // 1
    )  # buf_i64 is [num_slots, slot_stride_i64]
    if slot_stride_i64 < 4:
        raise ValueError(
            f"kv-canary: canary_buf slot stride must hold at least 4 int64 fields, got {slot_stride_i64}"
        )

    stored_tokens = buf_i64[slot_indices, _FIELD_TOKEN]
    stored_positions = buf_i64[slot_indices, _FIELD_POSITION]
    stored_chain_hashes = buf_i64[slot_indices, _FIELD_PREV_HASH]
    stored_real_kv_hashes = buf_i64[slot_indices, _FIELD_REAL_KV_HASH]

    # Compute expected chain hash. For chain-head entries (prev < 0) use splitmix64(CANARY_CHAIN_ANCHOR);
    # otherwise read the predecessor slot's 4 fields and splitmix64-mix4 them.
    is_chain_head = prev_slot_indices < 0
    safe_prev = torch.where(
        is_chain_head, torch.zeros_like(prev_slot_indices), prev_slot_indices
    )
    prev_tokens = buf_i64[safe_prev, _FIELD_TOKEN]
    prev_positions = buf_i64[safe_prev, _FIELD_POSITION]
    prev_chain_hashes = buf_i64[safe_prev, _FIELD_PREV_HASH]
    prev_real_kv_hashes = buf_i64[safe_prev, _FIELD_REAL_KV_HASH]

    return (
        stored_tokens,
        stored_positions,
        stored_chain_hashes,
        stored_real_kv_hashes,
        prev_tokens,
        prev_positions,
        prev_chain_hashes,
        prev_real_kv_hashes,
        is_chain_head,
    )


def _compute_expected_chain_hashes(
    *,
    prev_chain_hashes: torch.Tensor,
    prev_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    prev_real_kv_hashes: torch.Tensor,
    is_chain_head: torch.Tensor,
) -> torch.Tensor:
    chain_anchor_hash = _splitmix64_python(CANARY_CHAIN_ANCHOR)
    chain_anchor_signed = _to_signed_int64(chain_anchor_hash)
    expected_from_prev = _splitmix64_mix4_vec(
        prev_chain_hashes, prev_tokens, prev_positions, prev_real_kv_hashes
    )
    expected_chain_hashes = torch.where(
        is_chain_head,
        torch.full_like(prev_chain_hashes, chain_anchor_signed),
        expected_from_prev,
    )
    return expected_chain_hashes


def _compute_verify_fail_bits(
    *,
    stored_chain_hashes: torch.Tensor,
    expected_chain_hashes: torch.Tensor,
    stored_positions: torch.Tensor,
    expected_positions: torch.Tensor,
    stored_real_kv_hashes: torch.Tensor,
    expected_real_kv_hashes: torch.Tensor,
) -> torch.Tensor:
    fail_bits = torch.zeros_like(stored_chain_hashes, dtype=torch.int64)
    fail_bits |= torch.where(
        stored_chain_hashes != expected_chain_hashes,
        torch.full_like(fail_bits, _FAIL_REASON_BIT_CHAIN_HASH),
        torch.zeros_like(fail_bits),
    )
    fail_bits |= torch.where(
        stored_positions != expected_positions,
        torch.full_like(fail_bits, _FAIL_REASON_BIT_POSITION),
        torch.zeros_like(fail_bits),
    )
    fail_bits |= torch.where(
        stored_real_kv_hashes != expected_real_kv_hashes,
        torch.full_like(fail_bits, _FAIL_REASON_BIT_REAL_KV_HASH),
        torch.zeros_like(fail_bits),
    )
    return fail_bits


def _emit_verify_violations(
    *,
    fail_bits: torch.Tensor,
    slot_indices: torch.Tensor,
    stored_positions: torch.Tensor,
    stored_tokens: torch.Tensor,
    stored_chain_hashes: torch.Tensor,
    expected_chain_hashes: torch.Tensor,
    kernel_kind: CanaryLaunchTag,
    violation_ring: torch.Tensor,
    violation_write_index: torch.Tensor,
    work_device: torch.device,
) -> None:
    violation_mask = fail_bits != 0
    num_new_violations = int(violation_mask.sum().item())
    if num_new_violations == 0:
        return

    # Deterministic order: enumerate active entries left-to-right, emit those that violated, then atomic-write
    # them into the ring starting at the current write index. Beyond capacity, only the write index advances.
    violation_positions = torch.nonzero(violation_mask, as_tuple=False).squeeze(-1)

    sel_slot = slot_indices[violation_positions]
    sel_pos_stored = stored_positions[violation_positions]
    sel_stored_token = stored_tokens[violation_positions]
    sel_stored_chain = stored_chain_hashes[violation_positions]
    sel_expected_chain = expected_chain_hashes[violation_positions]
    sel_fail = fail_bits[violation_positions]

    base_idx = int(
        violation_write_index.new_empty(()).copy_(violation_write_index[0]).item()
    )
    ring_capacity = int(violation_ring.shape[0])

    new_rows = torch.zeros((num_new_violations, VIOLATION_FIELDS), dtype=torch.int64)
    new_rows[:, _VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
    new_rows[:, _VIOLATION_FIELD_SLOT_IDX] = sel_slot
    new_rows[:, _VIOLATION_FIELD_POSITION] = sel_pos_stored
    new_rows[:, _VIOLATION_FIELD_STORED_TOKEN] = sel_stored_token
    new_rows[:, _VIOLATION_FIELD_EXPECTED_TOKEN] = (
        0  # verify path does not have a token oracle
    )
    new_rows[:, _VIOLATION_FIELD_STORED_CHAIN_HASH] = sel_stored_chain
    new_rows[:, _VIOLATION_FIELD_EXPECTED_AUX] = sel_expected_chain
    new_rows[:, _VIOLATION_FIELD_FAIL_REASON_BITS] = sel_fail

    # Write only the rows that land inside the ring; beyond capacity they are dropped while the counter still
    # advances. Mirrors the CUDA atomicAdd contract.
    write_count_in_ring = max(0, min(num_new_violations, ring_capacity - base_idx))
    if write_count_in_ring > 0:
        ring_host = violation_ring.detach().to(device=work_device)
        ring_host[base_idx : base_idx + write_count_in_ring, :] = new_rows[
            :write_count_in_ring, :
        ]
        violation_ring.copy_(ring_host.to(violation_ring.device))

    violation_write_index[0] = violation_write_index[0] + num_new_violations


def _splitmix64_python(value: int) -> int:
    """Standard splitmix64 finalizer over a single uint64. Matches the CUDA splitmix64_finalize."""
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def _to_signed_int64(value: int) -> int:
    """Reinterpret an unsigned uint64 as signed int64 for torch.int64 storage."""
    value &= _U64_MASK
    if value >= _I64_SIGN_BIT:
        value -= 1 << 64
    return value


def _splitmix64_finalize_vec(words: torch.Tensor) -> torch.Tensor:
    """Vectorised splitmix64 finalizer on a flat int64 tensor.

    Performs the standard 3-step xorshift+multiply mix. Uses torch int64 ops; the bit pattern is preserved
    because all operations commute through the signed/unsigned reinterpretation.
    """
    x = words
    x = _xor_shift_mul(x, 30, _to_signed_int64(0xBF58476D1CE4E5B9))
    x = _xor_shift_mul(x, 27, _to_signed_int64(0x94D049BB133111EB))
    # final xor with right-shift-by-31 (logical, on uint64 bits).
    x = x ^ _logical_shr(x, 31)
    return x


def _xor_shift_mul(x: torch.Tensor, shift: int, multiplier_signed: int) -> torch.Tensor:
    """Compute ((x ^ (x >>L shift)) * multiplier) mod 2**64, returning signed int64.

    torch.int64 multiplication wraps mod 2**64 in two's complement, so the bit pattern is preserved.
    """
    shifted = _logical_shr(x, shift)
    mixed = x ^ shifted
    return mixed * multiplier_signed


def _logical_shr(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical (unsigned) right shift on a signed int64 tensor."""
    # Reinterpret-cast through uint64 by masking off the sign bit, then putting it back as a logical shift.
    if shift == 0:
        return x
    # Use bitwise_and with a mask of low bits after the shift; equivalent to (x_u64 >> shift).
    mask = (1 << (64 - shift)) - 1
    return (x >> shift) & mask


def _splitmix64_mix4_vec(
    prev_hash: torch.Tensor,
    token: torch.Tensor,
    position: torch.Tensor,
    real_kv_hash: torch.Tensor,
) -> torch.Tensor:
    """4-arg chain step: XOR all four uint64 inputs, then splitmix64-finalize."""
    combined = prev_hash ^ token ^ position ^ real_kv_hash
    return _splitmix64_finalize_vec(combined)


def _compute_real_kv_hash_vec(
    *,
    slot_indices: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    work_device: torch.device,
) -> torch.Tensor:
    """Compute one int64 real-KV fingerprint per slot, vectorised across entries.

    Off mode → all zeros. BIT mode → XOR-fold a single bit (low bit) of each read byte. ALL mode →
    splitmix64-fold every 8-byte word (zero-padded if read_bytes is not a multiple of 8). Each source's
    contribution is combined via splitmix64-mix into the running per-slot fingerprint.
    """
    num_entries = int(slot_indices.shape[0])
    acc = torch.zeros(num_entries, dtype=torch.int64, device=work_device)
    mode = int(real_kv_hash_mode)
    if mode == int(RealKvHashMode.OFF) or len(real_kv_sources) == 0:
        return acc

    for source in real_kv_sources:
        if source.read_bytes <= 0:
            continue
        bytes_per_slot = _gather_real_kv_bytes(
            slot_indices=slot_indices, source=source, work_device=work_device
        )
        # bytes_per_slot: [num_entries, read_bytes], uint8.
        if mode == int(RealKvHashMode.BIT):
            # XOR-fold low bit per byte into one int64 per entry. Result fits in a small range but stored as
            # int64 to match the canary slot field width.
            low_bits = (bytes_per_slot & 1).to(torch.int64)
            folded = low_bits.sum(dim=1) & 1
            source_hash = folded
        elif mode == int(RealKvHashMode.ALL):
            source_hash = _splitmix64_fold_bytes_vec(bytes_per_slot=bytes_per_slot)
        else:
            raise ValueError(f"kv-canary: unknown RealKvHashMode {mode}")
        # Mix this source's contribution into the running accumulator via splitmix64.
        combined = acc ^ source_hash
        acc = _splitmix64_finalize_vec(combined)

    return acc


def _gather_real_kv_bytes(
    *,
    slot_indices: torch.Tensor,
    source: RealKvSource,
    work_device: torch.device,
) -> torch.Tensor:
    """Pull source.read_bytes leading bytes per slot from source.tensor following the RealKvSource invariant.

    Returns shape [num_entries, read_bytes], dtype uint8.
    """
    page_size = source.page_size
    num_bytes_per_token = source.num_bytes_per_token
    read_bytes = source.read_bytes
    tensor_u8 = (
        source.tensor.detach().to(device=work_device).contiguous().view(torch.uint8)
    )

    rows = slot_indices // page_size
    cols_within_page = slot_indices % page_size
    col_start = cols_within_page * num_bytes_per_token

    # Build [num_entries, read_bytes] index by broadcasting.
    col_offsets = torch.arange(read_bytes, dtype=torch.int64, device=work_device)
    col_idx = col_start.unsqueeze(1) + col_offsets.unsqueeze(0)
    row_idx = rows.unsqueeze(1).expand_as(col_idx)

    return tensor_u8[row_idx, col_idx]


def _splitmix64_fold_bytes_vec(*, bytes_per_slot: torch.Tensor) -> torch.Tensor:
    """ALL-mode fold: pack bytes into 8-byte little-endian words, splitmix64-mix into a running accumulator.

    bytes_per_slot: [num_entries, read_bytes], uint8. Output: [num_entries], int64. read_bytes need not be a
    multiple of 8; trailing bytes are zero-padded.
    """
    num_entries = int(bytes_per_slot.shape[0])
    read_bytes = int(bytes_per_slot.shape[1])
    work_device = bytes_per_slot.device

    # Pad to a multiple of 8.
    pad = (8 - read_bytes % 8) % 8
    if pad > 0:
        bytes_padded = torch.cat(
            [
                bytes_per_slot,
                torch.zeros((num_entries, pad), dtype=torch.uint8, device=work_device),
            ],
            dim=1,
        )
    else:
        bytes_padded = bytes_per_slot

    num_words = bytes_padded.shape[1] // 8
    words_u8 = bytes_padded.view(num_entries, num_words, 8)
    # Little-endian byte->word packing: sum(b[k] << (8*k)) for k in 0..7.
    weights = torch.tensor(
        [1 << (8 * k) for k in range(8)], dtype=torch.int64, device=work_device
    )
    # Promote uint8 to int64 then weighted sum along the inner dim.
    words_i64 = (words_u8.to(torch.int64) * weights).sum(dim=2)

    # Iteratively fold each word into the accumulator: acc = splitmix64(acc XOR word).
    acc = torch.zeros(num_entries, dtype=torch.int64, device=work_device)
    for k in range(num_words):
        acc = _splitmix64_finalize_vec(acc ^ words_i64[:, k])
    return acc
