from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_SLOT_BYTES,
    FAIL_REASON_HASH,
    FAIL_REASON_POSITION,
    FAIL_REASON_POSITION_MONOTONIC,
    FAIL_REASON_REQ_ID,
    FAIL_REASON_TOKEN_ID,
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    VIOLATION_FIELDS,
    FailReason,
    canary_step,
)
from sglang.srt.kv_cache_canary.fingerprint import mix_step
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")


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
        ring_capacity=ring_capacity,
    )


def _alloc_pool(num_slots: int, slot_stride_bytes: int) -> torch.Tensor:
    return torch.zeros(num_slots, slot_stride_bytes, dtype=torch.uint8, device="cuda")


def _run(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    slot_stride_bytes: int,
    slot_indices: list[int],
    expected_req_ids: list[int],
    expected_token_ids: list[int],
    expected_positions: list[int],
    expected_prev_hashes: list[int],
    verify_mask: list[int],
    verify_seq_positions: list[int],
    state: dict,
    kernel_kind: int,
) -> None:
    canary_step(
        src_buf=src,
        dst_buf=dst,
        slot_stride_bytes=slot_stride_bytes,
        slot_indices=torch.tensor(slot_indices, dtype=torch.int64, device="cuda"),
        expected_req_ids=torch.tensor(
            expected_req_ids, dtype=torch.int64, device="cuda"
        ),
        expected_token_ids=torch.tensor(
            expected_token_ids, dtype=torch.int64, device="cuda"
        ),
        expected_positions=torch.tensor(
            expected_positions, dtype=torch.int64, device="cuda"
        ),
        expected_prev_hashes=torch.tensor(
            expected_prev_hashes, dtype=torch.int64, device="cuda"
        ),
        verify_mask=torch.tensor(verify_mask, dtype=torch.int32, device="cuda"),
        verify_seq_positions=torch.tensor(
            verify_seq_positions, dtype=torch.int64, device="cuda"
        ),
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


def test_clean_write_only_step_increments_counters_and_writes_slots():
    slot_stride = 256
    num_slots = 4
    src = _alloc_pool(num_slots * 4, slot_stride)
    dst = _alloc_pool(num_slots * 4, slot_stride)
    state = _alloc_state()

    slot_indices = [0, 1, 2, 3]
    token_ids = [10, 20, 30, 40]
    positions = [0, 1, 2, 3]
    req_ids = [7] * num_slots
    prev_hashes = [0] * num_slots

    _run(
        src=src,
        dst=dst,
        slot_stride_bytes=slot_stride,
        slot_indices=slot_indices,
        expected_req_ids=req_ids,
        expected_token_ids=token_ids,
        expected_positions=positions,
        expected_prev_hashes=prev_hashes,
        verify_mask=[0] * num_slots,
        verify_seq_positions=[-1] * num_slots,
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    assert int(state["is_errored"].item()) == 0
    assert int(state["slot_run_counter"].item()) == num_slots
    assert int(state["kernel_run_counter"].item()) >= 1
    for i in range(num_slots):
        rid, tid, pos, _ = _read_slot(dst, slot_indices[i], slot_stride)
        assert rid == req_ids[i]
        assert tid == token_ids[i]
        assert pos == positions[i]


def test_verify_round_trip_no_violation():
    slot_stride = 256
    num_slots = 3
    buf_a = _alloc_pool(num_slots * 2, slot_stride)
    buf_b = _alloc_pool(num_slots * 2, slot_stride)
    state = _alloc_state()

    req_ids = [11, 11, 11]
    token_ids = [100, 200, 300]
    positions = [0, 1, 2]
    prev_hashes: list[int] = [0]
    for i in range(1, num_slots):
        prev_hashes.append(
            mix_step(prev_hashes[-1], token_ids[i - 1], positions[i - 1])
        )
    slot_indices = [0, 1, 2]

    _run(
        src=buf_a,
        dst=buf_a,
        slot_stride_bytes=slot_stride,
        slot_indices=slot_indices,
        expected_req_ids=req_ids,
        expected_token_ids=token_ids,
        expected_positions=positions,
        expected_prev_hashes=prev_hashes,
        verify_mask=[0] * num_slots,
        verify_seq_positions=[-1] * num_slots,
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    _run(
        src=buf_a,
        dst=buf_b,
        slot_stride_bytes=slot_stride,
        slot_indices=slot_indices,
        expected_req_ids=req_ids,
        expected_token_ids=token_ids,
        expected_positions=positions,
        expected_prev_hashes=prev_hashes,
        verify_mask=[1] * num_slots,
        verify_seq_positions=positions,
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )
    assert int(state["is_errored"].item()) == 0
    assert int(state["first_violation_set"].item()) == 0


def test_req_id_mismatch_reports_violation_with_fail_reason_req_id():
    slot_stride = 128
    src = _alloc_pool(4, slot_stride)
    dst = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[0],
        expected_req_ids=[5],
        expected_token_ids=[42],
        expected_positions=[0],
        expected_prev_hashes=[0],
        verify_mask=[0],
        verify_seq_positions=[-1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    _run(
        src=src,
        dst=dst,
        slot_stride_bytes=slot_stride,
        slot_indices=[0],
        expected_req_ids=[99],
        expected_token_ids=[42],
        expected_positions=[0],
        expected_prev_hashes=[0],
        verify_mask=[1],
        verify_seq_positions=[0],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FAIL_REASON_REQ_ID


def test_first_violation_is_never_overwritten_by_cascading_mismatches():
    slot_stride = 128
    src = _alloc_pool(8, slot_stride)
    dst = _alloc_pool(8, slot_stride)
    state = _alloc_state(ring_capacity=4)

    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[0, 1, 2, 3],
        expected_req_ids=[1, 1, 1, 1],
        expected_token_ids=[10, 20, 30, 40],
        expected_positions=[0, 1, 2, 3],
        expected_prev_hashes=[0, 0, 0, 0],
        verify_mask=[0, 0, 0, 0],
        verify_seq_positions=[-1, -1, -1, -1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    _run(
        src=src,
        dst=dst,
        slot_stride_bytes=slot_stride,
        slot_indices=[0],
        expected_req_ids=[1],
        expected_token_ids=[999],
        expected_positions=[0],
        expected_prev_hashes=[0],
        verify_mask=[1],
        verify_seq_positions=[0],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )
    for _ in range(20):
        _run(
            src=src,
            dst=dst,
            slot_stride_bytes=slot_stride,
            slot_indices=[1, 2, 3],
            expected_req_ids=[999, 999, 999],
            expected_token_ids=[20, 30, 40],
            expected_positions=[1, 2, 3],
            expected_prev_hashes=[0, 0, 0],
            verify_mask=[1, 1, 1],
            verify_seq_positions=[1, 2, 3],
            state=state,
            kernel_kind=KERNEL_KIND_HEAD,
        )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FAIL_REASON_TOKEN_ID
    assert int(first[2]) == 0


def test_position_mismatch_reports_fail_reason_position():
    slot_stride = 128
    src = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[2],
        expected_req_ids=[1],
        expected_token_ids=[42],
        expected_positions=[5],
        expected_prev_hashes=[0],
        verify_mask=[0],
        verify_seq_positions=[-1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[2],
        expected_req_ids=[1],
        expected_token_ids=[42],
        expected_positions=[9],
        expected_prev_hashes=[0],
        verify_mask=[1],
        verify_seq_positions=[5],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FAIL_REASON_POSITION


def test_position_monotonic_mismatch_is_caught_independent_of_expected_table():
    """README §3 (b): the kernel must catch a slot whose stored ``input_position``
    disagrees with the verify-entry's sequence-position (``0..K``) even when the
    expected_position field happens to match the stored position. This is the
    indirection-free check that survives a smudged req_to_token_pool."""
    slot_stride = 128
    src = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    # Write the slot with stored input_position = 7.
    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[2],
        expected_req_ids=[1],
        expected_token_ids=[42],
        expected_positions=[7],
        expected_prev_hashes=[0],
        verify_mask=[0],
        verify_seq_positions=[-1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )
    # Verify with expected_position=7 (matches stored) but seq_position=3
    # (the request's 0..K sequence at this verify slot is 3). The slot was
    # written at position 7 → stored position field is 7 → seq_position 3
    # disagrees → fail with POSITION_MONOTONIC.
    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[2],
        expected_req_ids=[1],
        expected_token_ids=[42],
        expected_positions=[7],
        expected_prev_hashes=[0],
        verify_mask=[1],
        verify_seq_positions=[3],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FAIL_REASON_POSITION_MONOTONIC


def test_hash_mismatch_reports_fail_reason_hash_after_external_smudge():
    slot_stride = 128
    src = _alloc_pool(4, slot_stride)
    state = _alloc_state()

    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[1],
        expected_req_ids=[1],
        expected_token_ids=[7],
        expected_positions=[0],
        expected_prev_hashes=[0],
        verify_mask=[0],
        verify_seq_positions=[-1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )

    view = src.view(torch.int64).view(-1, slot_stride // 8)
    view[1, 3] = 0x1234_5678_9ABC_DEF0

    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[1],
        expected_req_ids=[1],
        expected_token_ids=[7],
        expected_positions=[0],
        expected_prev_hashes=[0],
        verify_mask=[1],
        verify_seq_positions=[0],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    assert int(state["is_errored"].item()) == 1
    first = state["first_violation"].cpu().tolist()
    assert int(first[1]) == FAIL_REASON_HASH


def test_violation_ring_valid_latch_marks_complete_rows():
    """C7: ring rows are written under per-row CAS latch; valid bit reaches 2
    when the row's 8 fields are fully stored. A torn row never reaches 2."""
    slot_stride = 128
    src = _alloc_pool(4, slot_stride)
    state = _alloc_state(ring_capacity=4)

    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[0],
        expected_req_ids=[1],
        expected_token_ids=[42],
        expected_positions=[0],
        expected_prev_hashes=[0],
        verify_mask=[0],
        verify_seq_positions=[-1],
        state=state,
        kernel_kind=KERNEL_KIND_HEAD,
    )
    _run(
        src=src,
        dst=src,
        slot_stride_bytes=slot_stride,
        slot_indices=[0],
        expected_req_ids=[999],
        expected_token_ids=[42],
        expected_positions=[0],
        expected_prev_hashes=[0],
        verify_mask=[1],
        verify_seq_positions=[0],
        state=state,
        kernel_kind=KERNEL_KIND_TAIL,
    )

    ring_valid = state["violation_ring_valid"].cpu().tolist()
    # First write touched row 0; valid bit should be 2 (= readable).
    assert int(ring_valid[0]) == 2


def test_fail_reason_enum_values_match_int_constants():
    """U2: FailReason enum exposed from Python must mirror the C++ constants."""
    assert FailReason.REQ_ID.value == FAIL_REASON_REQ_ID
    assert FailReason.TOKEN_ID.value == FAIL_REASON_TOKEN_ID
    assert FailReason.POSITION.value == FAIL_REASON_POSITION
    assert FailReason.HASH.value == FAIL_REASON_HASH
    assert FailReason.POSITION_MONOTONIC.value == FAIL_REASON_POSITION_MONOTONIC


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
