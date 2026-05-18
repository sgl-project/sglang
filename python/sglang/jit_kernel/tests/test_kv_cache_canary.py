"""GPU kernel tests for KV cache canary.

Cover the stateless-redesign kernel signature: per-verify-entry arrays,
per-write-req chains driven by one thread each, on-device splitmix64 chain
hash recomputation, three independent verify fail_reasons (req_id /
position monotonic / chain hash), and the violation buffer first-violation
latch.
"""

from __future__ import annotations

import torch

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_SLOT_BYTES,
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    VIOLATION_FIELDS,
    FailReason,
    canary_step,
)
from sglang.srt.kv_cache_canary.fingerprint import splitmix64_mix, to_signed_int64
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")

_SEED = 0xC0FFEE1234567890


def _alloc_state(ring_capacity: int = 64) -> dict:
    return dict(
        violation_ring=torch.zeros(
            ring_capacity, VIOLATION_FIELDS, dtype=torch.int64, device="cuda"
        ),
        violation_ring_valid=torch.zeros(
            ring_capacity, dtype=torch.int32, device="cuda"
        ),
        violation_write_index=torch.zeros(1, dtype=torch.int32, device="cuda"),
        first_violation=torch.zeros(VIOLATION_FIELDS, dtype=torch.int64, device="cuda"),
        first_violation_set=torch.zeros(1, dtype=torch.int32, device="cuda"),
        is_errored=torch.zeros(1, dtype=torch.int32, device="cuda"),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device="cuda"),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )


def _alloc_pool(num_slots: int, slot_stride_bytes: int) -> torch.Tensor:
    return torch.zeros(num_slots, slot_stride_bytes, dtype=torch.uint8, device="cuda")


def _i64(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def _i32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _run(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    slot_stride_bytes: int,
    verify_slot_indices: list[int],
    verify_positions: list[int],
    verify_req_ids: list[int],
    verify_prev_slot_indices: list[int],
    verify_active_mask: list[int],
    write_slot_indices: list[int],
    write_token_ids: list[int],
    write_positions: list[int],
    write_req_ids: list[int],
    write_req_seed_slot_indices: list[int],
    write_req_entry_starts: list[int],
    write_req_entry_counts: list[int],
    write_req_active_mask: list[int],
    state: dict,
    kernel_kind: int,
    seed: int = _SEED,
) -> None:
    canary_step(
        src_buf=src.flatten(),
        dst_buf=dst.flatten(),
        slot_stride_bytes=slot_stride_bytes,
        verify_slot_indices=_i64(verify_slot_indices or [0]),
        verify_positions=_i64(verify_positions or [0]),
        verify_req_ids=_i64(verify_req_ids or [0]),
        verify_prev_slot_indices=_i64(verify_prev_slot_indices or [-1]),
        verify_active_mask=_i32(verify_active_mask or [0]),
        write_slot_indices=_i64(write_slot_indices or [0]),
        write_token_ids=_i64(write_token_ids or [0]),
        write_positions=_i64(write_positions or [0]),
        write_req_ids=_i64(write_req_ids or [0]),
        write_req_seed_slot_indices=_i64(write_req_seed_slot_indices or [-1]),
        write_req_entry_starts=_i64(write_req_entry_starts or [0]),
        write_req_entry_counts=_i64(write_req_entry_counts or [0]),
        write_req_active_mask=_i32(write_req_active_mask or [0]),
        seed=seed,
        violation_ring=state["violation_ring"],
        violation_ring_valid=state["violation_ring_valid"],
        violation_write_index=state["violation_write_index"],
        first_violation=state["first_violation"],
        first_violation_set=state["first_violation_set"],
        is_errored=state["is_errored"],
        slot_run_counter=state["slot_run_counter"],
        kernel_run_counter=state["kernel_run_counter"],
        kernel_kind=kernel_kind,
    )
    torch.cuda.synchronize()


def _read_slot(
    buf: torch.Tensor, slot_idx: int, slot_stride_bytes: int
) -> tuple[int, int, int, int]:
    row = buf[slot_idx, :CANARY_SLOT_BYTES].clone().view(torch.int64).cpu().tolist()
    return tuple(int(x) for x in row)


def test_write_chain_seeded_from_kseed_fills_slots_with_splitmix64_chain():
    """First-write path: chain seeds from kSeed and stores (req, token, pos, prev_hash)."""
    slot_stride = 256
    dst = _alloc_pool(8, slot_stride)
    state = _alloc_state()

    tokens = [10, 20, 30]
    positions = [0, 1, 2]
    slot_indices = [4, 5, 6]
    req_id = 7

    _run(
        src=dst,
        dst=dst,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[],
        verify_positions=[],
        verify_req_ids=[],
        verify_prev_slot_indices=[],
        verify_active_mask=[],
        write_slot_indices=slot_indices,
        write_token_ids=tokens,
        write_positions=positions,
        write_req_ids=[req_id] * 3,
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_active_mask=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    assert int(state["is_errored"].item()) == 0
    assert int(state["slot_run_counter"].item()) == 3
    # Recompute the expected chain in Python and bit-wise-match the stored prev_hash.
    expected_prev = _SEED
    for slot_idx, token, position in zip(slot_indices, tokens, positions):
        rid, tid, pos, ph = _read_slot(dst, slot_idx, slot_stride)
        assert rid == req_id
        assert tid == token
        assert pos == position
        assert ph == to_signed_int64(expected_prev)
        expected_prev = splitmix64_mix(expected_prev, token, position)


def test_verify_clean_round_trip_no_violation():
    """Write-then-verify same buffer: clean state -> no violation."""
    slot_stride = 256
    buf = _alloc_pool(8, slot_stride)
    state = _alloc_state()

    tokens = [100, 200, 300]
    positions = [0, 1, 2]
    slot_indices = [0, 1, 2]
    req_id = 11

    # Phase 1: write the chain.
    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[],
        verify_positions=[],
        verify_req_ids=[],
        verify_prev_slot_indices=[],
        verify_active_mask=[],
        write_slot_indices=slot_indices,
        write_token_ids=tokens,
        write_positions=positions,
        write_req_ids=[req_id] * 3,
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_active_mask=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Phase 2: verify every position. prev_slot_indices: -1 for pos 0, the
    # actual previous slot for the rest.
    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=slot_indices,
        verify_positions=positions,
        verify_req_ids=[req_id] * 3,
        verify_prev_slot_indices=[-1, slot_indices[0], slot_indices[1]],
        verify_active_mask=[1, 1, 1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_ids=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_active_mask=[],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )
    assert int(state["is_errored"].item()) == 0
    assert int(state["first_violation_set"].item()) == 0


def test_verify_req_id_mismatch_reports_req_id_fail_reason():
    slot_stride = 128
    buf = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[],
        verify_positions=[],
        verify_req_ids=[],
        verify_prev_slot_indices=[],
        verify_active_mask=[],
        write_slot_indices=[0],
        write_token_ids=[42],
        write_positions=[0],
        write_req_ids=[5],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        write_req_active_mask=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Verify expecting a different req_id — should trip kFailReasonReqId.
    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_req_ids=[99],
        verify_prev_slot_indices=[-1],
        verify_active_mask=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_ids=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_active_mask=[],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FailReason.REQ_ID.value


def test_verify_position_mismatch_reports_position_monotonic_fail_reason():
    slot_stride = 128
    buf = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[],
        verify_positions=[],
        verify_req_ids=[],
        verify_prev_slot_indices=[],
        verify_active_mask=[],
        write_slot_indices=[0],
        write_token_ids=[42],
        write_positions=[0],
        write_req_ids=[1],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        write_req_active_mask=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Verify with an expected position that doesn't match the slot's stored
    # position field -> kFailReasonPositionMonotonic.
    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[0],
        verify_positions=[5],
        verify_req_ids=[1],
        verify_prev_slot_indices=[-1],
        verify_active_mask=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_ids=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_active_mask=[],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FailReason.POSITION_MONOTONIC.value


def test_verify_chain_hash_mismatch_reports_hash_fail_reason():
    """Corrupt slot[0].prev_hash post-write -> verify on slot[0] sees hash mismatch."""
    slot_stride = 128
    buf = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[],
        verify_positions=[],
        verify_req_ids=[],
        verify_prev_slot_indices=[],
        verify_active_mask=[],
        write_slot_indices=[0],
        write_token_ids=[42],
        write_positions=[0],
        write_req_ids=[1],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[1],
        write_req_active_mask=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Corrupt slot[0].prev_hash field (offset 24 bytes = field index 3).
    buf_view = buf.view(torch.int64).view(-1, CANARY_SLOT_BYTES // 8)
    buf_view[0, 3] = torch.tensor(
        to_signed_int64(0xDEADBEEFDEADBEEF), dtype=torch.int64, device="cuda"
    )

    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_req_ids=[1],
        verify_prev_slot_indices=[-1],
        verify_active_mask=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_ids=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_active_mask=[],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FailReason.HASH.value


def test_inactive_mask_rows_are_skipped_no_io_no_counter():
    """``*_active_mask == 0`` rows = skip-sentinel padding for cuda graph fixed buffers."""
    slot_stride = 256
    buf = _alloc_pool(8, slot_stride)
    state = _alloc_state()

    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[0, 1],
        verify_positions=[0, 1],
        verify_req_ids=[1, 1],
        verify_prev_slot_indices=[-1, 0],
        verify_active_mask=[0, 0],
        write_slot_indices=[0, 1, 2],
        write_token_ids=[10, 20, 30],
        write_positions=[0, 1, 2],
        write_req_ids=[1, 1, 1],
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[3],
        write_req_active_mask=[0],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # Every active mask is 0: kernel must be a no-op on slot I/O and counters.
    assert int(state["is_errored"].item()) == 0
    assert int(state["slot_run_counter"].item()) == 0
    # But the kernel_run_counter still increments — that's what the §5
    # health monitor uses to detect "kernel actually launched".
    assert int(state["kernel_run_counter"].item()) >= 1


def test_first_violation_preserved_across_cascading_mismatches():
    slot_stride = 128
    buf = _alloc_pool(8, slot_stride)
    state = _alloc_state(ring_capacity=4)

    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[],
        verify_positions=[],
        verify_req_ids=[],
        verify_prev_slot_indices=[],
        verify_active_mask=[],
        write_slot_indices=[0, 1, 2, 3],
        write_token_ids=[10, 20, 30, 40],
        write_positions=[0, 1, 2, 3],
        write_req_ids=[1] * 4,
        write_req_seed_slot_indices=[-1],
        write_req_entry_starts=[0],
        write_req_entry_counts=[4],
        write_req_active_mask=[1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    # First mismatch: req_id mismatch at slot 0.
    _run(
        src=buf,
        dst=buf,
        slot_stride_bytes=slot_stride,
        verify_slot_indices=[0],
        verify_positions=[0],
        verify_req_ids=[999],
        verify_prev_slot_indices=[-1],
        verify_active_mask=[1],
        write_slot_indices=[],
        write_token_ids=[],
        write_positions=[],
        write_req_ids=[],
        write_req_seed_slot_indices=[],
        write_req_entry_starts=[],
        write_req_entry_counts=[],
        write_req_active_mask=[],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )
    first_after_initial = state["first_violation"].cpu().tolist()

    # Cascade: 20 more verify launches with wrong req_id.
    for _ in range(20):
        _run(
            src=buf,
            dst=buf,
            slot_stride_bytes=slot_stride,
            verify_slot_indices=[1, 2, 3],
            verify_positions=[1, 2, 3],
            verify_req_ids=[888, 888, 888],
            verify_prev_slot_indices=[0, 1, 2],
            verify_active_mask=[1, 1, 1],
            write_slot_indices=[],
            write_token_ids=[],
            write_positions=[],
            write_req_ids=[],
            write_req_seed_slot_indices=[],
            write_req_entry_starts=[],
            write_req_entry_counts=[],
            write_req_active_mask=[],
            state=state,
            kernel_kind=KERNEL_KIND_HEAD,
        )

    # first_violation row must be byte-identical to the first observed one.
    first_after_cascade = state["first_violation"].cpu().tolist()
    assert first_after_initial == first_after_cascade
    # The write_index has advanced past ring capacity.
    assert int(state["violation_write_index"].item()) > 4
