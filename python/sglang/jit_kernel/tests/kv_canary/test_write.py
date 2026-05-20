"""Differential test: CUDA canary_write_step vs the torch reference, byte-equal."""

from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import (
    _VIOLATION_FIELD_FAIL_REASON_BITS,
    CANARY_CHAIN_ANCHOR,
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
)
from sglang.jit_kernel.kv_canary.write import (
    _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH,
    _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH,
    CanaryPseudoMode,
    canary_write_step,
)
from sglang.jit_kernel.kv_canary.write_ref import (
    canary_write_step_torch_reference,
)
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_equal,
    chain_anchor_signed,
    make_canary_buf,
    make_real_kv_source,
    make_real_kv_sources,
    make_write_plan,
    read_slot_fields,
    splitmix64,
    splitmix64_mix4,
    to_signed_int64,
    write_slot_fields,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _run_both(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    canary_write_step(
        canary_buf=cuda_canary_buf,
        plan=plan_cuda,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        violation_ring=cuda_log.ring,
        violation_write_index=cuda_log.write_index,
        slot_run_counter=cuda_log.slot_run_counter,
        kernel_run_counter=cuda_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_cuda,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    canary_write_step_torch_reference(
        canary_buf=ref_canary_buf,
        plan=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()


def _run_both_and_assert_buf_and_state_equal(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    _run_both(
        cuda_canary_buf=cuda_canary_buf,
        ref_canary_buf=ref_canary_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=real_kv_sources_cuda,
        real_kv_sources_ref=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
        kernel_kind=kernel_kind,
    )
    assert_canary_buf_equal(buf_a=cuda_canary_buf, buf_b=ref_canary_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def _setup_pair() -> tuple[torch.Tensor, torch.Tensor]:
    cuda_buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_DEVICE)
    return cuda_buf, cuda_buf.clone()


def _dummy_pseudo_tensors(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(num_tokens, dtype=torch.int32, device=_DEVICE),
        torch.zeros(num_tokens, dtype=torch.int32, device=_DEVICE),
    )


def test_seed_slot_idx_negative_uses_anchor() -> None:
    """``seed_slot_idx == -1`` → initial ``running_prev_hash`` is ``splitmix64(CANARY_CHAIN_ANCHOR)``."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([42], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    stored_token, stored_position, stored_prev_hash, _ = read_slot_fields(
        canary_buf=cuda_buf, slot_idx=3
    )
    assert stored_token == 42
    assert stored_position == 0
    assert stored_prev_hash == chain_anchor_signed()
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_seed_slot_idx_loads_predecessor() -> None:
    """``seed_slot_idx >= 0`` → load 4 fields from ``canary_buf[seed]`` and splitmix64-advance into prev_hash."""
    cuda_buf, ref_buf = _setup_pair()

    # Step: pre-stamp slot 7 with a known chain link.
    seed_token, seed_position = 100, 4
    seed_prev_signed = to_signed_int64(splitmix64(CANARY_CHAIN_ANCHOR))
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=7,
            token=seed_token,
            position=seed_position,
            prev_hash=seed_prev_signed,
            real_kv_hash=0,
        )

    plan_cuda = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[7], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[7], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([999], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([5], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([2], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    expected_prev_hash = splitmix64_mix4(
        splitmix64(CANARY_CHAIN_ANCHOR), seed_token, seed_position, 0
    )
    _, _, stored_prev_hash, _ = read_slot_fields(canary_buf=cuda_buf, slot_idx=2)
    assert stored_prev_hash == to_signed_int64(expected_prev_hash)
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_seed_slot_chain_link_continuous() -> None:
    """After write, ``slot[0].prev_hash`` is consistent with verify's chain reconstruction from seed."""
    # Step 1: write a chain from seed slot=7 → newly written slot=2. Then run verify with prev=7 and
    # assert no violation — i.e., slot[2].prev_hash is the correct splitmix64-mix of seed's 4 fields.
    cuda_buf, ref_buf = _setup_pair()
    seed_token, seed_position = 11, 0
    seed_prev_signed = to_signed_int64(splitmix64(CANARY_CHAIN_ANCHOR))
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=7,
            token=seed_token,
            position=seed_position,
            prev_hash=seed_prev_signed,
            real_kv_hash=0,
        )

    plan_cuda = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[7], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[7], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([222], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([2], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    # Step 2: verify slot[2] with prev=7 — expects no violation.
    from sglang.jit_kernel.kv_canary.verify import canary_verify_step
    from sglang.jit_kernel.tests.kv_canary.canary_helpers import make_verify_plan

    verify_plan = make_verify_plan(
        slot_indices=[2], positions=[1], prev_slot_indices=[7], device=_DEVICE
    )
    verify_log = FakeViolationLog.allocate(device=_DEVICE)
    canary_verify_step(
        canary_buf=cuda_buf,
        plan=verify_plan,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=verify_log.ring,
        violation_write_index=verify_log.write_index,
        slot_run_counter=verify_log.slot_run_counter,
        kernel_run_counter=verify_log.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    torch.cuda.synchronize()
    assert int(verify_log.write_index[0].item()) == 0


def test_chain_link_byte_equal_5_step() -> None:
    """5-step chain, buf / ring / counters byte-equal against ref."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 5], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 5], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(5)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both_and_assert_buf_and_state_equal(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )


def test_mock_mode_off_ignores_expected() -> None:
    """``pseudo_mode = OFF`` → expected tensors are ignored (we pass garbage to prove the kernel skips them)."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([1, 2, 3], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    # Garbage expected tensors that, if the kernel mistakenly reads, would generate mismatches.
    pseudo_tokens = torch.tensor([999, 999, 999], dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.tensor([999, 999, 999], dtype=torch.int32, device=_DEVICE)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_mock_mode_on_match_no_violation() -> None:
    """``pseudo_mode = ON`` and expected matches actual → no violation, chain advances."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([7, 8, 9], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = fb_input_ids.clone()
    pseudo_positions = fb_positions.clone()
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.ON,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_mock_mode_on_token_mismatch_records_violation() -> None:
    """``pseudo_mode = ON`` token mismatch → violation recorded; chain advances on ACTUAL token."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([42], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = torch.tensor([99], dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.ON,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    fail_bits = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH
    # Chain advances on actual (42), not expected (99). Stored token should be 42.
    stored_token, _, _, _ = read_slot_fields(canary_buf=cuda_buf, slot_idx=0)
    assert stored_token == 42
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_mock_mode_on_position_mismatch_records_violation() -> None:
    """``pseudo_mode = ON`` position mismatch → violation recorded; chain advances on ACTUAL position."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([42], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([7], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = torch.tensor([42], dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.ON,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    fail_bits = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH
    _, stored_position, _, _ = read_slot_fields(canary_buf=cuda_buf, slot_idx=0)
    assert stored_position == 7
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_mock_mode_chain_advances_on_actual_not_expected() -> None:
    """Expected differs from actual on every entry → downstream verify must NOT cascade chain errors."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([10, 20, 30], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    # Every actual differs from expected.
    pseudo_tokens = torch.tensor([999, 999, 999], dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.tensor([999, 999, 999], dtype=torch.int32, device=_DEVICE)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.ON,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    # All 3 entries should fire a violation row.
    assert int(cuda_log.write_index[0].item()) == 3
    # Run a downstream verify — it must see no chain mismatch because chain advanced on actuals.
    from sglang.jit_kernel.kv_canary.verify import canary_verify_step
    from sglang.jit_kernel.tests.kv_canary.canary_helpers import make_verify_plan

    verify_plan = make_verify_plan(
        slot_indices=[0, 1, 2],
        positions=[0, 1, 2],
        prev_slot_indices=[-1, 0, 1],
        device=_DEVICE,
    )
    verify_log = FakeViolationLog.allocate(device=_DEVICE)
    canary_verify_step(
        canary_buf=cuda_buf,
        plan=verify_plan,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=verify_log.ring,
        violation_write_index=verify_log.write_index,
        slot_run_counter=verify_log.slot_run_counter,
        kernel_run_counter=verify_log.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    torch.cuda.synchronize()
    assert int(verify_log.write_index[0].item()) == 0
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_negative_slot_skips_entry() -> None:
    """``fb_out_cache_loc[i] < 0`` → that entry is skipped: no buf write, no violation, no
    slot_run_counter bump. Covers both SWA out-of-window (after caller-side LUT gather) and
    explicit padding intents.
    """
    cuda_buf, ref_buf = _setup_pair()
    # Two entries: first writes to slot 4 normally; second has slot=-1 and must be skipped.
    plan_cuda = make_write_plan(
        write_offsets=[0, 2], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 2], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([42, 99], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([4, -1], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(2)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    stored_token, _, _, _ = read_slot_fields(canary_buf=cuda_buf, slot_idx=4)
    assert stored_token == 42
    # slot_run_counter counts only non-skipped entries (1, not 2).
    assert int(cuda_log.slot_run_counter.item()) == 1
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_pre_translated_slot_writes_normally() -> None:
    """``fb_out_cache_loc[i] >= 0`` → the kernel writes to exactly that slot, with no LUT applied. This
    confirms the kernel is SWA-agnostic: SWA endpoints feed the same shape of input here after their
    host-side gather, so the contract is symmetric across FULL / SWA groups.
    """
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([55], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    # Slot 4 here could equally be a FULL-group raw out_cache_loc value, or the result of an SWA
    # endpoint's host gather. The kernel can't tell the difference and that's the point.
    fb_out_cache_loc = torch.tensor([4], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    stored_token, _, _, _ = read_slot_fields(canary_buf=cuda_buf, slot_idx=4)
    assert stored_token == 55
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_real_kv_mode_off_writes_zero() -> None:
    """``RealKvHashMode.OFF`` → ``real_kv_hash`` field is written as 0 regardless of source presence."""
    cuda_buf, ref_buf = _setup_pair()
    sources = make_real_kv_sources(count=2, device=_DEVICE)

    plan_cuda = make_write_plan(
        write_offsets=[0, 2], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 2], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(2)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources,
        real_kv_sources_ref=sources,
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    _, _, _, real_kv_0 = read_slot_fields(canary_buf=cuda_buf, slot_idx=0)
    _, _, _, real_kv_1 = read_slot_fields(canary_buf=cuda_buf, slot_idx=1)
    assert real_kv_0 == 0
    assert real_kv_1 == 0
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def _run_real_kv_mode_byte_equal_case(mode: RealKvHashMode) -> None:
    cuda_buf, ref_buf = _setup_pair()
    sources_cuda = make_real_kv_sources(count=2, device=_DEVICE)
    sources_ref = tuple(
        RealKvSource(
            tensor=s.tensor.clone(),
            page_size=s.page_size,
            num_bytes_per_token=s.num_bytes_per_token,
            read_bytes=s.read_bytes,
        )
        for s in sources_cuda
    )

    plan_cuda = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 3], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([10, 20, 30], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(3)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both_and_assert_buf_and_state_equal(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=mode,
    )


def test_real_kv_mode_bit_byte_equal() -> None:
    _run_real_kv_mode_byte_equal_case(RealKvHashMode.BIT)


def test_real_kv_mode_all_byte_equal() -> None:
    _run_real_kv_mode_byte_equal_case(RealKvHashMode.ALL)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_real_kv_sources_fold_1_to_4(count: int) -> None:
    """Folding ``count`` sources sequentially → CUDA matches ref for every count in {1..4}."""
    cuda_buf, ref_buf = _setup_pair()
    sources_cuda = make_real_kv_sources(count=count, device=_DEVICE)
    sources_ref = tuple(
        RealKvSource(
            tensor=s.tensor.clone(),
            page_size=s.page_size,
            num_bytes_per_token=s.num_bytes_per_token,
            read_bytes=s.read_bytes,
        )
        for s in sources_cuda
    )

    plan_cuda = make_write_plan(
        write_offsets=[0, 2], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 2], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(2)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both_and_assert_buf_and_state_equal(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )


def test_real_kv_source_above_4_raises() -> None:
    """``len(real_kv_sources) > 4`` → host wrapper raises ValueError before launching."""
    cuda_buf = make_canary_buf(device=_DEVICE)
    plan = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    log = FakeViolationLog.allocate(device=_DEVICE)
    sources = make_real_kv_sources(count=4, device=_DEVICE)
    extra = make_real_kv_source(device=_DEVICE)
    too_many = sources + (extra,)

    with pytest.raises(ValueError, match="at most 4 RealKvSource"):
        canary_write_step(
            canary_buf=cuda_buf,
            plan=plan,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            pseudo_mode=CanaryPseudoMode.OFF,
            pseudo_expected_tokens=pseudo_tokens,
            pseudo_expected_positions=pseudo_positions,
            violation_ring=log.ring,
            violation_write_index=log.write_index,
            slot_run_counter=log.slot_run_counter,
            kernel_run_counter=log.kernel_run_counter,
            real_kv_sources=too_many,
            real_kv_hash_mode=RealKvHashMode.OFF,
        )


def test_kernel_run_counter_per_call() -> None:
    """``kernel_run_counter`` increments by 1 per call (even when ``write_num_valid_reqs == 0``)."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 0], seed_slot_indices=[-1], num_valid_reqs=0, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 0], seed_slot_indices=[-1], num_valid_reqs=0, device=_DEVICE
    )
    fb_input_ids = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    for _ in range(3):
        _run_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            pseudo_mode=CanaryPseudoMode.OFF,
            pseudo_expected_tokens=pseudo_tokens,
            pseudo_expected_positions=pseudo_positions,
            cuda_log=cuda_log,
            ref_log=ref_log,
            real_kv_sources_cuda=(),
            real_kv_sources_ref=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )

    assert int(cuda_log.kernel_run_counter[0].item()) == 3
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_slot_run_counter_sums_entries() -> None:
    """``slot_run_counter`` += sum(entry_count) across all active reqs in this call."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 2, 5],
        seed_slot_indices=[-1, -1],
        num_valid_reqs=2,
        device=_DEVICE,
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 2, 5],
        seed_slot_indices=[-1, -1],
        num_valid_reqs=2,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(5)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.slot_run_counter[0].item()) == 5
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_empty_plan_no_op() -> None:
    """``write_num_valid_reqs = 0`` → no buf write, no slot_run_counter bump, only kernel_run_counter += 1."""
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 0],
        seed_slot_indices=[-1],
        num_valid_reqs=0,
        req_capacity=4,
        device=_DEVICE,
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 0],
        seed_slot_indices=[-1],
        num_valid_reqs=0,
        req_capacity=4,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert int(cuda_log.slot_run_counter[0].item()) == 0
    assert int(cuda_log.kernel_run_counter[0].item()) == 1
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_padding_block_skipped() -> None:
    """``blockIdx.x >= write_num_valid_reqs[0]`` → block early-exits, no write to canary_buf."""
    cuda_buf, ref_buf = _setup_pair()
    # Allocate plan with req_capacity=4 but only declare 1 active req.
    plan_cuda = make_write_plan(
        write_offsets=[0, 1],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        req_capacity=4,
        device=_DEVICE,
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        req_capacity=4,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(1)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    # Only slot 0 should have been written; padding blocks 1..3 must not touch the buffer.
    stored_token, _, _, _ = read_slot_fields(canary_buf=cuda_buf, slot_idx=0)
    assert stored_token == 1
    for slot_idx in (1, 2, 3):
        stored_token_other, _, _, _ = read_slot_fields(
            canary_buf=cuda_buf, slot_idx=slot_idx
        )
        assert stored_token_other == 0
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


@pytest.mark.parametrize("hardcoded", [True])
def test_chain_link_byte_equal_5_step_hardcoded(hardcoded: bool) -> None:
    """5-step write chain with hand-computed splitmix64 expected fields per slot."""
    assert hardcoded
    cuda_buf, ref_buf = _setup_pair()
    plan_cuda = make_write_plan(
        write_offsets=[0, 5], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 5], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    tokens = [101, 202, 303, 404, 505]
    positions = [0, 1, 2, 3, 4]
    out_cache_loc = [0, 1, 2, 3, 4]
    real_kv_hashes = [0, 0, 0, 0, 0]

    # Step 1: compute the expected stored prev_hash sequence in pure Python via splitmix64.
    expected_prev_hashes_u64: list[int] = []
    running = splitmix64(CANARY_CHAIN_ANCHOR)
    for token, position, real_kv_hash in zip(tokens, positions, real_kv_hashes):
        expected_prev_hashes_u64.append(running)
        running = splitmix64_mix4(running, token, position, real_kv_hash)
    expected_prev_hashes_signed = [to_signed_int64(h) for h in expected_prev_hashes_u64]

    fb_input_ids = torch.tensor(tokens, dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor(positions, dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int32, device=_DEVICE)
    pseudo_tokens, pseudo_positions = _dummy_pseudo_tensors(5)
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    # Step 2: verify every slot's stored 4 fields match the hardcoded expected sequence.
    for slot_idx, expected_token, expected_position, expected_prev_signed in zip(
        out_cache_loc, tokens, positions, expected_prev_hashes_signed
    ):
        stored_token, stored_position, stored_prev_hash, stored_real_kv_hash = (
            read_slot_fields(canary_buf=cuda_buf, slot_idx=slot_idx)
        )
        assert stored_token == expected_token
        assert stored_position == expected_position
        assert stored_prev_hash == expected_prev_signed
        assert stored_real_kv_hash == 0
    assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)
