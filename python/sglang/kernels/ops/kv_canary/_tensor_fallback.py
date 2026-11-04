"""XPU versions of the CUDA-only kv-canary kernels: plan entries, verify and write.

Byte-equal to plan_ref, verify_ref and write_ref, which loop per token on the host and are too slow to serve with.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.consts import to_signed_int64
from sglang.kernels.ops.kv_canary.verify import (
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
)
from sglang.kernels.ops.kv_canary.write import WritePlan

# Leading bytes per slot hashed in RealKvHashMode.PARTIAL; must match verify_ref.
_PARTIAL_REAL_KV_READ_BYTES = 16

_SPLITMIX64_MUL_1 = to_signed_int64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL_2 = to_signed_int64(0x94D049BB133111EB)
_ANCHOR_HASH = to_signed_int64(consts.canary_chain_initial_hash())


def _shift_right_logical(x: torch.Tensor, shift: int) -> torch.Tensor:
    # torch's >> on int64 is arithmetic; mask off the sign-extended bits.
    return (x >> shift) & ((1 << (64 - shift)) - 1)


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """Elementwise consts.splitmix64 on int64 tensors, relying on two's-complement wraparound."""
    x = (x ^ _shift_right_logical(x, 30)) * _SPLITMIX64_MUL_1
    x = (x ^ _shift_right_logical(x, 27)) * _SPLITMIX64_MUL_2
    return x ^ _shift_right_logical(x, 31)


def _canary_fields(canary_buf: torch.Tensor) -> torch.Tensor:
    buf_i64 = canary_buf.detach().view(torch.int64)
    if int(buf_i64.shape[1]) < consts.CANARY_FIELDS_PER_SLOT:
        raise ValueError(
            f"kv-canary: canary_buf slot stride must hold at least {consts.CANARY_FIELDS_PER_SLOT} int64 fields, got {int(buf_i64.shape[1])}"
        )
    return buf_i64


def _swa_translate(
    slots: torch.Tensor, full_to_swa_index_mapping: torch.Tensor | None
) -> torch.Tensor:
    if full_to_swa_index_mapping is None or slots.numel() == 0:
        return slots
    mapping_length = int(full_to_swa_index_mapping.shape[0])
    out_of_range = slots >= mapping_length
    if bool(out_of_range.any()):
        raise ValueError(
            f"kv-canary: SWA slot {int(slots[out_of_range][0])} is outside full_to_swa_index_mapping length {mapping_length}"
        )
    translated = full_to_swa_index_mapping.to(torch.int64)[slots.clamp(min=0)]
    return torch.where(slots < 0, slots, translated)


def launch_plan_entries_tensor_fallback(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor | None,
    verify_offsets_scratch: torch.Tensor,
    verify_enable: torch.Tensor,
    req_to_verify_expected_tokens: torch.Tensor | None,
    req_to_verify_expected_tokens_valid_lens: torch.Tensor | None,
    out_verify_slot_indices: torch.Tensor,
    out_verify_expected_tokens: torch.Tensor,
    out_verify_expected_positions: torch.Tensor,
    out_verify_prev_slot_indices: torch.Tensor,
    kv_token_id_vs_position_offset: int,
    swa_window_size: int,
) -> None:
    """Same contract as plan.entries_kernel.launch_plan_entries_kernel.

    Entry e of request r covers position window_start[r] + (e - offsets[r]); the offsets come from the Triton
    offsets kernel, which also zeroes verify_enable on overflow.
    """
    if (
        req_to_verify_expected_tokens is not None
        and req_to_verify_expected_tokens_valid_lens is None
    ):
        raise ValueError(
            "kv-canary: launch_plan_entries_tensor_fallback requires "
            "req_to_verify_expected_tokens_valid_lens when req_to_verify_expected_tokens is set"
        )
    batch_size = int(req_pool_indices.shape[0])
    if batch_size == 0:
        return
    offsets = verify_offsets_scratch[: batch_size + 1]
    enabled, total_verify = torch.stack(
        [verify_enable[0].to(torch.int64), offsets[batch_size]]
    ).tolist()
    if enabled == 0 or total_verify == 0:
        return

    device = req_to_token.device
    max_seq_len = int(req_to_token.shape[1])
    prefix = prefix_lens.to(device=device, dtype=torch.int64)
    if swa_window_size > 0:
        window_start = (prefix - swa_window_size).clamp(min=0)
    else:
        window_start = torch.zeros_like(prefix)

    entry_indices = torch.arange(total_verify, device=device)
    request_of_entry = torch.searchsorted(offsets[1:], entry_indices, right=True)
    positions = (
        window_start[request_of_entry] + entry_indices - offsets[request_of_entry]
    )
    pool_row = req_pool_indices.to(device=device, dtype=torch.int64)[request_of_entry]
    mapping = full_to_swa_index_mapping

    slots = _swa_translate(req_to_token[pool_row, positions].to(torch.int64), mapping)
    prev_slots_full = req_to_token[
        pool_row, (positions - 1).clamp(0, max_seq_len - 1)
    ].to(torch.int64)
    prev_slots = torch.where(
        positions > 0,
        _swa_translate(torch.where(positions > 0, prev_slots_full, -1), mapping),
        -1,
    )
    expected_tokens = torch.full_like(positions, -1)
    if req_to_verify_expected_tokens is not None:
        valid_len = req_to_verify_expected_tokens_valid_lens.to(
            device=device, dtype=torch.int64
        )[request_of_entry]
        expected_token_pos = positions + int(kv_token_id_vs_position_offset)
        in_range = (expected_token_pos >= 0) & (expected_token_pos < valid_len)
        max_position = max(int(req_to_verify_expected_tokens.shape[1]) - 1, 0)
        gathered = req_to_verify_expected_tokens[
            pool_row, expected_token_pos.clamp(0, max_position)
        ].to(torch.int64)
        expected_tokens = torch.where(in_range, gathered, expected_tokens)

    for output, values in (
        (out_verify_slot_indices, slots),
        (out_verify_expected_tokens, expected_tokens),
        (out_verify_expected_positions, positions),
        (out_verify_prev_slot_indices, prev_slots),
    ):
        output[:total_verify].copy_(values.to(output.dtype))


def _real_kv_hash(
    *,
    slots: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: consts.RealKvHashMode,
) -> torch.Tensor:
    """Per-slot real_kv_hash (signed int64), matching verify_ref._compute_real_kv_hash_scalar."""
    combined_hash = torch.zeros_like(slots)
    mode = int(real_kv_hash_mode)
    if (
        mode == int(consts.RealKvHashMode.NONE)
        or len(real_kv_sources) == 0
        or slots.numel() == 0
    ):
        return combined_hash
    for source in real_kv_sources:
        read_bytes = (
            _PARTIAL_REAL_KV_READ_BYTES
            if mode == int(consts.RealKvHashMode.PARTIAL)
            else source.read_bytes
        )
        source_slots = slots.to(source.tensor.device)
        rows = source_slots // source.page_size
        col_start = (source_slots % source.page_size) * source.num_bytes_per_token
        cols = col_start.unsqueeze(1) + torch.arange(read_bytes, device=rows.device)
        # A per-layer view keeps unit stride in its last dim, so the bytes are gathered in place.
        tensor_u8 = source.tensor.detach().view(torch.uint8)
        # read_bytes is a multiple of 16, so the little-endian words need no padding.
        words = (
            tensor_u8[rows.unsqueeze(1), cols]
            .contiguous()
            .view(torch.int64)
            .to(slots.device)
        )
        source_hash = torch.zeros_like(slots)
        for word_idx in range(words.shape[1]):
            source_hash = _splitmix64(source_hash ^ words[:, word_idx])
        combined_hash = _splitmix64(combined_hash ^ source_hash)
    return combined_hash


def _fail_bit(condition: torch.Tensor, reason: consts.FailReason) -> torch.Tensor:
    return torch.where(condition, int(reason), 0)


def launch_canary_verify_kernel_tensor_fallback(
    *,
    context: VerifyOrWriteContext,
    plan: VerifyPlan,
    check_verify_expected_token: bool,
) -> None:
    """Byte-equal to verify_ref.launch_canary_verify_kernel_torch_reference."""
    buf_i64 = _canary_fields(context.canary_buf)
    context.kernel_run_counter.add_(1)

    capacity = int(plan.verify_slot_indices.shape[0])
    enabled, num_valid = torch.stack(
        [plan.enable[0].to(torch.int64), plan.verify_num_valid[0].to(torch.int64)]
    ).tolist()
    active = max(0, min(num_valid, capacity))
    if enabled == 0 or active == 0:
        return
    context.slot_run_counter.add_(active)

    slots = plan.verify_slot_indices[:active].to(torch.int64)
    # Skip SGLang's padded-token dummy KV slot so zero-initialized req_to_token entries do not produce spurious violations.
    keep = slots != consts.TOKEN_TO_KV_SLOT_PADDING
    slots = slots[keep]
    if slots.numel() == 0:
        return
    positions = plan.verify_expected_positions[:active].to(torch.int64)[keep]
    prev_slots = plan.verify_prev_slot_indices[:active].to(torch.int64)[keep]
    if check_verify_expected_token:
        expected_tokens = plan.verify_expected_tokens[:active].to(torch.int64)[keep]
    else:
        expected_tokens = torch.full_like(slots, -1)

    stored = buf_i64[slots]
    stored_chain_hash = stored[:, consts.CANARY_FIELD_PREV_HASH]
    prev_reachable = prev_slots != consts.TOKEN_TO_KV_SLOT_PADDING
    prev = buf_i64[prev_slots.clamp(min=0)]
    # Tensor form of consts.canary_chain_extend_hash.
    prev_hash = _splitmix64(prev[:, consts.CANARY_FIELD_PREV_HASH])
    prev_hash = _splitmix64(prev_hash ^ prev[:, consts.CANARY_FIELD_TOKEN])
    prev_hash = _splitmix64(prev_hash ^ prev[:, consts.CANARY_FIELD_POSITION])
    prev_hash = torch.where(prev_slots < 0, _ANCHOR_HASH, prev_hash)
    expected_chain_hash = torch.where(prev_reachable, prev_hash, stored_chain_hash)
    expected_real_kv_hash = _real_kv_hash(
        slots=slots,
        real_kv_sources=context.real_kv_sources,
        real_kv_hash_mode=context.real_kv_hash_mode,
    )

    fail_bits = (
        _fail_bit(
            prev_reachable & (stored_chain_hash != expected_chain_hash),
            consts.FailReason.VERIFY_CHAIN_HASH_MISMATCH,
        )
        | _fail_bit(
            (expected_tokens != -1)
            & (stored[:, consts.CANARY_FIELD_TOKEN] != expected_tokens),
            consts.FailReason.VERIFY_TOKEN_MISMATCH,
        )
        | _fail_bit(
            stored[:, consts.CANARY_FIELD_POSITION] != positions,
            consts.FailReason.VERIFY_POSITION_MISMATCH,
        )
        | _fail_bit(
            stored[:, consts.CANARY_FIELD_REAL_KV_HASH] != expected_real_kv_hash,
            consts.FailReason.VERIFY_REAL_KV_HASH_MISMATCH,
        )
    )
    failed = fail_bits != 0
    if not bool(failed.any()):
        return

    rows = torch.zeros(
        (int(failed.sum()), consts.VIOLATION_FIELDS),
        dtype=torch.int64,
        device=slots.device,
    )
    rows[:, consts.VIOLATION_FIELD_KERNEL_KIND] = int(context.kernel_kind)
    rows[:, consts.VIOLATION_FIELD_SLOT_IDX] = slots[failed]
    rows[:, consts.VIOLATION_FIELD_POSITION] = stored[failed][
        :, consts.CANARY_FIELD_POSITION
    ]
    rows[:, consts.VIOLATION_FIELD_STORED_TOKEN] = stored[failed][
        :, consts.CANARY_FIELD_TOKEN
    ]
    rows[:, consts.VIOLATION_FIELD_EXPECTED_TOKEN] = expected_tokens[failed]
    rows[:, consts.VIOLATION_FIELD_STORED_CHAIN_HASH] = stored_chain_hash[failed]
    rows[:, consts.VIOLATION_FIELD_EXPECTED_AUX] = expected_chain_hash[failed]
    rows[:, consts.VIOLATION_FIELD_FAIL_REASON_BITS] = fail_bits[failed]
    _append_violations(context=context, rows=rows)


def _append_violations(*, context: VerifyOrWriteContext, rows: torch.Tensor) -> None:
    """Append rows to the violation ring in order; rows past the ring end are counted but dropped."""
    violation_ring = context.violation_ring
    violation_write_index = context.violation_write_index
    base_idx = int(violation_write_index[0])
    write_count = max(
        0, min(int(rows.shape[0]), int(violation_ring.shape[0]) - base_idx)
    )
    if write_count > 0:
        violation_ring[base_idx : base_idx + write_count].copy_(
            rows[:write_count].to(
                device=violation_ring.device, dtype=violation_ring.dtype
            )
        )
    violation_write_index[0] = violation_write_index[0] + int(rows.shape[0])


class _SlotRecord(NamedTuple):
    token: int
    position: int
    prev_hash: int


class _WriteChain(NamedTuple):
    """Host-side result of folding every request's write chain, in launch order."""

    written_entries: list[int]
    written_records: dict[int, _SlotRecord]
    violation_rows: list[list[int]]


def launch_canary_write_kernel_tensor_fallback(
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
    """Byte-equal to write_ref.launch_canary_write_kernel_torch_reference.

    The per-request chain hash is inherently sequential, so it is folded with host ints over one bulk copy of the launch's inputs; the slot stores and real-KV fingerprints run on device.
    """
    buf_i64 = _canary_fields(context.canary_buf)
    chain_position_assert = int(context.enable_chain_position_assert[0]) != 0
    context.kernel_run_counter.add_(1)

    active_reqs = max(
        0,
        min(
            int(plan.write_num_valid_reqs[0]),
            int(plan.write_seed_slot_indices.shape[0]),
        ),
    )
    if active_reqs <= 0:
        return
    offsets = plan.write_offsets[: active_reqs + 1].tolist()
    total_entries = offsets[active_reqs]
    if total_entries <= 0:
        return
    if enable_write_input_assert:
        if expected_input_tokens is None or expected_input_positions is None:
            raise ValueError(
                "kv-canary: expected input tensors are required when enable_write_input_assert=True"
            )
    elif expected_input_tokens is not None or expected_input_positions is not None:
        raise ValueError(
            "kv-canary: expected input tensors must be None when enable_write_input_assert=False"
        )

    seeds = plan.write_seed_slot_indices[:active_reqs].to(torch.int64)
    chain = _fold_write_chain(
        kernel_kind=int(context.kernel_kind),
        offsets=offsets,
        seeds=seeds.tolist(),
        seed_fields=buf_i64[
            seeds.clamp(min=0), : consts.CANARY_FIELDS_PER_SLOT
        ].tolist(),
        slots=out_cache_loc[:total_entries].tolist(),
        tokens=input_ids[:total_entries].tolist(),
        positions=positions[:total_entries].tolist(),
        expected_tokens=(
            expected_input_tokens[:total_entries].tolist()
            if enable_write_input_assert
            else None
        ),
        expected_positions=(
            expected_input_positions[:total_entries].tolist()
            if enable_write_input_assert
            else None
        ),
        chain_position_assert=chain_position_assert,
    )
    if chain.written_records:
        device = buf_i64.device
        slots = torch.tensor(
            list(chain.written_records), dtype=torch.int64, device=device
        )
        records = torch.tensor(
            list(chain.written_records.values()), dtype=torch.int64, device=device
        )
        buf_i64[slots, consts.CANARY_FIELD_TOKEN] = records[:, 0]
        buf_i64[slots, consts.CANARY_FIELD_POSITION] = records[:, 1]
        buf_i64[slots, consts.CANARY_FIELD_PREV_HASH] = records[:, 2]
        buf_i64[slots, consts.CANARY_FIELD_REAL_KV_HASH] = _real_kv_hash(
            slots=slots,
            real_kv_sources=context.real_kv_sources,
            real_kv_hash_mode=context.real_kv_hash_mode,
        )
    context.slot_run_counter.add_(len(chain.written_entries))
    if chain.violation_rows:
        _append_violations(
            context=context,
            rows=torch.tensor(chain.violation_rows, dtype=torch.int64),
        )


def _violation_row(
    *,
    kernel_kind: int,
    slot: int,
    position: int,
    stored_token: int,
    expected_token: int,
    chain_hash: int,
    expected_aux: int,
    fail_bits: consts.FailReason,
) -> list[int]:
    row = [0] * consts.VIOLATION_FIELDS
    row[consts.VIOLATION_FIELD_KERNEL_KIND] = kernel_kind
    row[consts.VIOLATION_FIELD_SLOT_IDX] = slot
    row[consts.VIOLATION_FIELD_POSITION] = position
    row[consts.VIOLATION_FIELD_STORED_TOKEN] = stored_token
    row[consts.VIOLATION_FIELD_EXPECTED_TOKEN] = expected_token
    row[consts.VIOLATION_FIELD_STORED_CHAIN_HASH] = chain_hash
    row[consts.VIOLATION_FIELD_EXPECTED_AUX] = expected_aux
    row[consts.VIOLATION_FIELD_FAIL_REASON_BITS] = int(fail_bits)
    return row


def _fold_write_chain(
    *,
    kernel_kind: int,
    offsets: list[int],
    seeds: list[int],
    seed_fields: list[list[int]],
    slots: list[int],
    tokens: list[int],
    positions: list[int],
    expected_tokens: list[int] | None,
    expected_positions: list[int] | None,
    chain_position_assert: bool,
) -> _WriteChain:
    """Walks requests in order like the reference; a seed written earlier in this launch reads that new record."""
    chain = _WriteChain(written_entries=[], written_records={}, violation_rows=[])
    for req_idx, seed in enumerate(seeds):
        if seed < 0:
            running_hash = _ANCHOR_HASH
            seed_position = 0
        else:
            if seed in chain.written_records:
                record = chain.written_records[seed]
                seed_token, seed_position, seed_prev_hash = record
            else:
                fields = seed_fields[req_idx]
                seed_token = fields[consts.CANARY_FIELD_TOKEN]
                seed_position = fields[consts.CANARY_FIELD_POSITION]
                seed_prev_hash = fields[consts.CANARY_FIELD_PREV_HASH]
            running_hash = consts.canary_chain_extend_hash(
                seed_prev_hash, seed_token, seed_position
            )
        do_chain_assert = seed >= 0 and chain_position_assert
        prev_position = seed_position if do_chain_assert else 0
        for entry_idx in range(offsets[req_idx], offsets[req_idx + 1]):
            slot = slots[entry_idx]
            if slot < 0:
                continue
            token, position = tokens[entry_idx], positions[entry_idx]
            chain_hash = to_signed_int64(running_hash)
            if expected_tokens is not None and expected_positions is not None:
                fail_bits = consts.FailReason(0)
                if token != expected_tokens[entry_idx]:
                    fail_bits |= consts.FailReason.WRITE_TOKEN_MISMATCH
                if position != expected_positions[entry_idx]:
                    fail_bits |= consts.FailReason.WRITE_POSITION_MISMATCH
                if fail_bits:
                    chain.violation_rows.append(
                        _violation_row(
                            kernel_kind=kernel_kind,
                            slot=slot,
                            position=position,
                            stored_token=token,
                            expected_token=expected_tokens[entry_idx],
                            chain_hash=chain_hash,
                            expected_aux=expected_positions[entry_idx],
                            fail_bits=fail_bits,
                        )
                    )
            if do_chain_assert:
                if position != prev_position + 1:
                    chain.violation_rows.append(
                        _violation_row(
                            kernel_kind=kernel_kind,
                            slot=slot,
                            position=position,
                            stored_token=token,
                            expected_token=token,
                            chain_hash=chain_hash,
                            expected_aux=prev_position + 1,
                            fail_bits=consts.FailReason.WRITE_POSITION_MISMATCH,
                        )
                    )
                prev_position = position
            chain.written_entries.append(entry_idx)
            # A slot written twice keeps its last record, as in the reference.
            chain.written_records[slot] = _SlotRecord(token, position, chain_hash)
            running_hash = consts.canary_chain_extend_hash(
                running_hash, token, position
            )
    return chain
