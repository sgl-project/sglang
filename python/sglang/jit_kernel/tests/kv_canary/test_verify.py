"""Differential test: CUDA canary_verify_step vs the torch reference, byte-equal."""

from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import (
    _FAIL_REASON_BIT_CHAIN_HASH,
    _FAIL_REASON_BIT_POSITION,
    _FAIL_REASON_BIT_REAL_KV_HASH,
    _VIOLATION_FIELD_FAIL_REASON_BITS,
    _VIOLATION_FIELD_KERNEL_KIND,
    CANARY_CHAIN_ANCHOR,
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
    canary_verify_step,
)
from sglang.jit_kernel.kv_canary.verify_ref import (
    canary_verify_step_torch_reference,
)
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode
from sglang.jit_kernel.kv_canary.write_ref import (
    canary_write_step_torch_reference,
)
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    assert_canary_state_equal,
    chain_anchor_signed,
    make_canary_buf,
    make_real_kv_source,
    make_real_kv_sources,
    make_verify_plan,
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
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    real_kv_sources_cuda: tuple[RealKvSource, ...],
    real_kv_sources_ref: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    canary_verify_step(
        canary_buf=cuda_canary_buf,
        plan=plan_cuda,
        kernel_kind=kernel_kind,
        violation_ring=cuda_log.ring,
        violation_write_index=cuda_log.write_index,
        slot_run_counter=cuda_log.slot_run_counter,
        kernel_run_counter=cuda_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_cuda,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    canary_verify_step_torch_reference(
        canary_buf=ref_canary_buf,
        plan=plan_ref,
        kernel_kind=kernel_kind,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()


def _run_both_and_assert_state_equal(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
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
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=real_kv_sources_cuda,
        real_kv_sources_ref=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
        kernel_kind=kernel_kind,
    )
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def _setup_pair_with_canned_chain(
    *,
    num_slots: int = 16,
    slot_stride_bytes: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate (cuda_buf, ref_buf) of identical contents.

    The buffers start zero-filled; tests stamp chain state via ``write_slot_fields`` on both copies so the
    CUDA and ref paths see byte-equal input.
    """
    cuda_buf = make_canary_buf(
        num_slots=num_slots, slot_stride_bytes=slot_stride_bytes, device=_DEVICE
    )
    ref_buf = cuda_buf.clone()
    return cuda_buf, ref_buf


def _stamp_chain(
    *,
    cuda_buf: torch.Tensor,
    ref_buf: torch.Tensor,
    tokens: list[int],
    positions: list[int],
    slot_indices: list[int],
    real_kv_hashes: list[int] | None = None,
) -> list[int]:
    """Pre-stamp a real chain into the buffers; return the expected stored prev_hash per slot (signed)."""
    n = len(tokens)
    real_kv_hashes = real_kv_hashes or [0] * n
    running_prev_hash = splitmix64(CANARY_CHAIN_ANCHOR)
    stored_prev_hashes: list[int] = []
    for slot_idx, token, position, real_kv_hash in zip(
        slot_indices, tokens, positions, real_kv_hashes
    ):
        signed_prev = to_signed_int64(running_prev_hash)
        write_slot_fields(
            canary_buf=cuda_buf,
            slot_idx=slot_idx,
            token=token,
            position=position,
            prev_hash=signed_prev,
            real_kv_hash=to_signed_int64(real_kv_hash),
        )
        write_slot_fields(
            canary_buf=ref_buf,
            slot_idx=slot_idx,
            token=token,
            position=position,
            prev_hash=signed_prev,
            real_kv_hash=to_signed_int64(real_kv_hash),
        )
        stored_prev_hashes.append(signed_prev)
        running_prev_hash = splitmix64_mix4(
            running_prev_hash, token, position, real_kv_hash
        )
    return stored_prev_hashes


def test_chain_head_anchor() -> None:
    """``prev_slot_idx == -1`` → kernel uses ``splitmix64(CANARY_CHAIN_ANCHOR)`` as the expected prev_hash."""
    # Step 1: stamp slot 5 such that stored.prev_hash already equals splitmix64(CANARY_CHAIN_ANCHOR).
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=5,
            token=42,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    # Step 2: a single-entry plan with prev_slot_idx = -1 should record no violation.
    plan_cuda = make_verify_plan(
        slot_indices=[5], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[5], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_chain_link_byte_equal_5_step() -> None:
    """5-step chain, CUDA vs ref byte-equal across ring / counters / canary_buf (read-only)."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    slot_indices = [0, 1, 2, 3, 4]
    tokens = [11, 22, 33, 44, 55]
    positions = [0, 1, 2, 3, 4]
    _stamp_chain(
        cuda_buf=cuda_buf,
        ref_buf=ref_buf,
        tokens=tokens,
        positions=positions,
        slot_indices=slot_indices,
    )
    prev_slot_indices = [-1, 0, 1, 2, 3]

    plan_cuda = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=prev_slot_indices,
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=prev_slot_indices,
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_violation_token_mismatch() -> None:
    """Stored token differs from a fresh write at the same slot → TOKEN-side accounting via the chain bit."""
    # Verify kernel doesn't have a TOKEN fail bit per se — token mismatch propagates into next-slot
    # CHAIN_HASH mismatch. Inject token corruption at slot 1 and verify slot 2 sees CHAIN_HASH bit set.
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    slot_indices = [0, 1, 2]
    tokens = [100, 200, 300]
    positions = [0, 1, 2]
    _stamp_chain(
        cuda_buf=cuda_buf,
        ref_buf=ref_buf,
        tokens=tokens,
        positions=positions,
        slot_indices=slot_indices,
    )

    # Step: corrupt the stored token at slot 1 in both buffers — chain hash propagates downstream.
    write_slot_fields(
        canary_buf=cuda_buf,
        slot_idx=1,
        token=999,
        position=1,
        prev_hash=0,
        real_kv_hash=0,
    )
    write_slot_fields(
        canary_buf=ref_buf,
        slot_idx=1,
        token=999,
        position=1,
        prev_hash=0,
        real_kv_hash=0,
    )

    plan_cuda = make_verify_plan(
        slot_indices=[2],
        positions=[2],
        prev_slot_indices=[1],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[2],
        positions=[2],
        prev_slot_indices=[1],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 1
    fail_bits = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_CHAIN_HASH
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_violation_position_mismatch() -> None:
    """Stored position differs from what the slot's chain reconstruction would yield → POSITION bit."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    # Stamp slot 7 with a valid head chain but stored position = 0; ask verify to expect position 5.
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=7,
            token=42,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    plan_cuda = make_verify_plan(
        slot_indices=[7], positions=[5], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[7], positions=[5], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 1
    fail_bits = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_POSITION
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_violation_position_diverges_from_plan() -> None:
    """Plan-supplied position contradicts stored position → POSITION bit (verify trusts plan, not +1)."""
    # Step: a clean chain head with stored position 0; plan claims position 99 — kernel must flag POSITION.
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=3,
            token=11,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    plan_cuda = make_verify_plan(
        slot_indices=[3], positions=[99], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[3], positions=[99], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    fail_bits = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_POSITION
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_violation_prev_hash_mismatch() -> None:
    """Stored prev_hash differs from predecessor-derived expectation → CHAIN_HASH bit."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    slot_indices = [0, 1]
    tokens = [10, 20]
    positions = [0, 1]
    _stamp_chain(
        cuda_buf=cuda_buf,
        ref_buf=ref_buf,
        tokens=tokens,
        positions=positions,
        slot_indices=slot_indices,
    )

    # Step: corrupt slot 1's stored prev_hash with a bogus signed int64.
    write_slot_fields(
        canary_buf=cuda_buf,
        slot_idx=1,
        token=20,
        position=1,
        prev_hash=0x1234567812345678,
        real_kv_hash=0,
    )
    write_slot_fields(
        canary_buf=ref_buf,
        slot_idx=1,
        token=20,
        position=1,
        prev_hash=0x1234567812345678,
        real_kv_hash=0,
    )

    plan_cuda = make_verify_plan(
        slot_indices=[1], positions=[1], prev_slot_indices=[0], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[1], positions=[1], prev_slot_indices=[0], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    fail_bits = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_CHAIN_HASH
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_violation_real_kv_hash_mismatch() -> None:
    """Mutate one byte of a RealKvSource tensor after writing the chain → REAL_KV_HASH bit on verify."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    sources_cuda = make_real_kv_sources(count=1, device=_DEVICE)
    sources_ref = tuple(
        RealKvSource(
            tensor=s.tensor.clone(),
            page_size=s.page_size,
            num_bytes_per_token=s.num_bytes_per_token,
            read_bytes=s.read_bytes,
        )
        for s in sources_cuda
    )

    # Step: write a chain with real_kv_hash mixin, then mutate one byte in the source tensors so the next
    # verify reconstructs a hash that differs from the stored one.
    write_plan_cuda = make_write_plan(
        write_offsets=[0, 3],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        device=_DEVICE,
    )
    write_plan_ref = make_write_plan(
        write_offsets=[0, 3],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([7, 8, 9], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = torch.zeros(3, dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.zeros(3, dtype=torch.int32, device=_DEVICE)
    write_log = FakeViolationLog.allocate(device=_DEVICE)

    canary_write_step_torch_reference(
        canary_buf=cuda_buf,
        plan=write_plan_cuda,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tokens,
        pseudo_expected_positions=pseudo_positions,
        violation_ring=write_log.ring,
        violation_write_index=write_log.write_index,
        slot_run_counter=write_log.slot_run_counter,
        kernel_run_counter=write_log.kernel_run_counter,
        real_kv_sources=sources_cuda,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )
    ref_buf.copy_(cuda_buf)

    # Mutate one byte in BOTH copies so the verify recomputed hash diverges from stored.
    sources_cuda[0].tensor[0, 0] ^= 0xFF
    sources_ref[0].tensor.copy_(sources_cuda[0].tensor)

    plan_cuda = make_verify_plan(
        slot_indices=[0],
        positions=[0],
        prev_slot_indices=[-1],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0],
        positions=[0],
        prev_slot_indices=[-1],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )

    fail_bits = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_REAL_KV_HASH
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_real_kv_mode_off_yields_zero() -> None:
    """OFF mode → stored real_kv_hash field stays zero post-write; verify with OFF agrees byte-equal."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    sources = make_real_kv_sources(count=2, device=_DEVICE)

    plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=1,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources,
        real_kv_sources_ref=sources,
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def _run_real_kv_mode_byte_equal_case(mode: RealKvHashMode) -> None:
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
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

    # Write a chain through the ref so both buffers are byte-equal post-write.
    write_plan = make_write_plan(
        write_offsets=[0, 3],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([10, 20, 30], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = torch.zeros(3, dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.zeros(3, dtype=torch.int32, device=_DEVICE)
    log = FakeViolationLog.allocate(device=_DEVICE)
    canary_write_step_torch_reference(
        canary_buf=cuda_buf,
        plan=write_plan,
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
        real_kv_sources=sources_cuda,
        real_kv_hash_mode=mode,
    )
    ref_buf.copy_(cuda_buf)

    plan_cuda = make_verify_plan(
        slot_indices=[0, 1, 2],
        positions=[0, 1, 2],
        prev_slot_indices=[-1, 0, 1],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0, 1, 2],
        positions=[0, 1, 2],
        prev_slot_indices=[-1, 0, 1],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=mode,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_real_kv_mode_bit_byte_equal() -> None:
    _run_real_kv_mode_byte_equal_case(RealKvHashMode.BIT)


def test_real_kv_mode_all_byte_equal() -> None:
    _run_real_kv_mode_byte_equal_case(RealKvHashMode.ALL)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_real_kv_sources_fold_1_to_4(count: int) -> None:
    """Fold ``count`` sources sequentially → CUDA matches ref for every count in {1..4}."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
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

    write_plan = make_write_plan(
        write_offsets=[0, 2],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    log = FakeViolationLog.allocate(device=_DEVICE)
    canary_write_step_torch_reference(
        canary_buf=cuda_buf,
        plan=write_plan,
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
        real_kv_sources=sources_cuda,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )
    ref_buf.copy_(cuda_buf)

    plan_cuda = make_verify_plan(
        slot_indices=[0, 1],
        positions=[0, 1],
        prev_slot_indices=[-1, 0],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0, 1],
        positions=[0, 1],
        prev_slot_indices=[-1, 0],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources_cuda,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_real_kv_source_read_bytes_zero_skipped() -> None:
    """``read_bytes == 0`` → source is skipped; CUDA must not dereference dim-1 beyond the 1-byte dummy."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    # 1-byte dummy tensor with read_bytes = 0; kernel must never touch it.
    dummy = RealKvSource(
        tensor=torch.zeros((1, 1), dtype=torch.uint8, device=_DEVICE),
        page_size=1,
        num_bytes_per_token=1,
        read_bytes=0,
    )

    plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=1,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(dummy,),
        real_kv_sources_ref=(dummy,),
        real_kv_hash_mode=RealKvHashMode.ALL,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_real_kv_source_padding_below_4() -> None:
    """Host wrapper pads to 4 slots when fewer sources are supplied; dummy slots are never dereferenced."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    sources = make_real_kv_sources(count=2, device=_DEVICE)

    plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=1,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both_and_assert_state_equal(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources,
        real_kv_sources_ref=sources,
        real_kv_hash_mode=RealKvHashMode.OFF,
    )


def test_real_kv_source_above_4_raises() -> None:
    """``len(real_kv_sources) > 4`` → host wrapper raises ValueError before launching."""
    canary_buf = make_canary_buf(device=_DEVICE)
    plan = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    log = FakeViolationLog.allocate(device=_DEVICE)
    sources = make_real_kv_sources(count=4, device=_DEVICE)
    extra = make_real_kv_source(device=_DEVICE)
    too_many = sources + (extra,)

    with pytest.raises(ValueError, match="at most 4 RealKvSource"):
        canary_verify_step(
            canary_buf=canary_buf,
            plan=plan,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=log.ring,
            violation_write_index=log.write_index,
            slot_run_counter=log.slot_run_counter,
            kernel_run_counter=log.kernel_run_counter,
            real_kv_sources=too_many,
            real_kv_hash_mode=RealKvHashMode.OFF,
        )


def test_real_kv_source_holey_dim1() -> None:
    """``tensor.shape[1] > page_size * num_bytes_per_token`` → trailing bytes are skipped."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    holey_source = make_real_kv_source(
        num_slots=16,
        num_bytes_per_token=8,
        page_size=1,
        read_bytes=8,
        pad_dim1=16,  # 16 trailing pad bytes per row; must be skipped.
        device=_DEVICE,
    )
    # Fill those skipped trailing bytes with garbage; CUDA must not read them.
    holey_source.tensor[:, 8:].fill_(0xAA)
    sources = (holey_source,)
    sources_ref = (
        RealKvSource(
            tensor=holey_source.tensor.clone(),
            page_size=holey_source.page_size,
            num_bytes_per_token=holey_source.num_bytes_per_token,
            read_bytes=holey_source.read_bytes,
        ),
    )

    write_plan = make_write_plan(
        write_offsets=[0, 2],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    log = FakeViolationLog.allocate(device=_DEVICE)
    canary_write_step_torch_reference(
        canary_buf=cuda_buf,
        plan=write_plan,
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
        real_kv_sources=sources,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )
    ref_buf.copy_(cuda_buf)

    plan_cuda = make_verify_plan(
        slot_indices=[0, 1],
        positions=[0, 1],
        prev_slot_indices=[-1, 0],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0, 1],
        positions=[0, 1],
        prev_slot_indices=[-1, 0],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_page_size_gt_1_access_pattern() -> None:
    """``page_size > 1`` → byte access follows ``(row=slot//page, col=(slot%page)*bpt:)``."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain(num_slots=8)
    src = make_real_kv_source(
        num_slots=8,
        num_bytes_per_token=4,
        page_size=4,  # 2 rows × 4 slots/page × 4 bytes/slot.
        read_bytes=4,
        device=_DEVICE,
    )
    # Each slot's 4 bytes get a slot-specific signature so kernel mis-indexing would shift the hash.
    flat = src.tensor.view(-1)
    for slot_idx in range(8):
        row = slot_idx // src.page_size
        col = (slot_idx % src.page_size) * src.num_bytes_per_token
        for k in range(src.num_bytes_per_token):
            flat_index = row * (src.page_size * src.num_bytes_per_token) + col + k
            flat[flat_index] = (slot_idx * 13 + k) & 0xFF
    sources = (src,)
    sources_ref = (
        RealKvSource(
            tensor=src.tensor.clone(),
            page_size=src.page_size,
            num_bytes_per_token=src.num_bytes_per_token,
            read_bytes=src.read_bytes,
        ),
    )

    write_plan = make_write_plan(
        write_offsets=[0, 2],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        device=_DEVICE,
    )
    fb_input_ids = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0, 5], dtype=torch.int32, device=_DEVICE)
    pseudo_tokens = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    pseudo_positions = torch.zeros(2, dtype=torch.int32, device=_DEVICE)
    log = FakeViolationLog.allocate(device=_DEVICE)
    canary_write_step_torch_reference(
        canary_buf=cuda_buf,
        plan=write_plan,
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
        real_kv_sources=sources,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )
    ref_buf.copy_(cuda_buf)

    plan_cuda = make_verify_plan(
        slot_indices=[0, 5],
        positions=[0, 1],
        prev_slot_indices=[-1, 0],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0, 5],
        positions=[0, 1],
        prev_slot_indices=[-1, 0],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=sources,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=RealKvHashMode.ALL,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_swa_translated_slot_indices() -> None:
    """SWA-translated slots already passed in plan; verify kernel does no further translation."""
    # SWA verify plans carry pre-translated slot indices — the verify kernel never sees the FULL slot
    # index again. We pre-stamp the SWA-side slot and feed it directly into the verify plan to assert no
    # extra translation happens kernel-side.
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=2,
            token=99,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    plan_cuda = make_verify_plan(
        slot_indices=[2], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[2], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_kernel_run_counter_per_call() -> None:
    """``kernel_run_counter`` increments by 1 per call, even when ``verify_num_valid == 0``."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    plan_cuda = make_verify_plan(
        slot_indices=[], positions=[], prev_slot_indices=[], capacity=4, device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[], positions=[], prev_slot_indices=[], capacity=4, device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    for _ in range(3):
        _run_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
            real_kv_sources_cuda=(),
            real_kv_sources_ref=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )

    assert int(cuda_log.kernel_run_counter[0].item()) == 3
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_slot_run_counter_per_entry() -> None:
    """``slot_run_counter`` accumulates ``verify_num_valid`` entries per call."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    slot_indices = [0, 1, 2, 3]
    tokens = [10, 11, 12, 13]
    positions = [0, 1, 2, 3]
    _stamp_chain(
        cuda_buf=cuda_buf,
        ref_buf=ref_buf,
        tokens=tokens,
        positions=positions,
        slot_indices=slot_indices,
    )

    plan_cuda = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=[-1, 0, 1, 2],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=[-1, 0, 1, 2],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.slot_run_counter[0].item()) == 4
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_violation_ring_fill_once_first_row() -> None:
    """First violation lands at ring[0]; subsequent violations advance ``violation_write_index``."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    # 3 chain-head entries with stored values that all yield POSITION mismatch (positions all 99).
    anchor_signed = chain_anchor_signed()
    for slot_idx in (0, 1, 2):
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=1,
                position=0,
                prev_hash=anchor_signed,
                real_kv_hash=0,
            )

    plan_cuda = make_verify_plan(
        slot_indices=[0, 1, 2],
        positions=[99, 99, 99],
        prev_slot_indices=[-1, -1, -1],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0, 1, 2],
        positions=[99, 99, 99],
        prev_slot_indices=[-1, -1, -1],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 3
    # row 0 is filled (slot 0 → POSITION bit).
    first_row_fail = int(cuda_log.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert first_row_fail & _FAIL_REASON_BIT_POSITION


def test_violation_ring_overflow_counter_still_increments() -> None:
    """Ring capacity exceeded → rows beyond are dropped but ``write_index`` still grows."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    anchor_signed = chain_anchor_signed()
    n_violations = 10
    slot_indices = list(range(n_violations))
    for slot_idx in slot_indices:
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=1,
                position=0,
                prev_hash=anchor_signed,
                real_kv_hash=0,
            )

    plan_cuda = make_verify_plan(
        slot_indices=slot_indices,
        positions=[99] * n_violations,
        prev_slot_indices=[-1] * n_violations,
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=slot_indices,
        positions=[99] * n_violations,
        prev_slot_indices=[-1] * n_violations,
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(capacity=4, device=_DEVICE)
    ref_log = FakeViolationLog.allocate(capacity=4, device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == n_violations
    assert int(ref_log.write_index[0].item()) == n_violations
    # Atomic-order may permute ring contents under overflow; only the write_index counter is
    # byte-equal — we relax the ring-contents check here.
    assert torch.equal(cuda_log.write_index, ref_log.write_index)


def test_kernel_kind_stamped_into_row() -> None:
    """Different ``CanaryLaunchTag`` values → violation row.kernel_kind reflects each."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=1,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    for tag in (CanaryLaunchTag.HEAD_K_FULL, CanaryLaunchTag.SWEEP_V_SWA):
        plan_cuda = make_verify_plan(
            slot_indices=[0], positions=[99], prev_slot_indices=[-1], device=_DEVICE
        )
        plan_ref = make_verify_plan(
            slot_indices=[0], positions=[99], prev_slot_indices=[-1], device=_DEVICE
        )
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)
        _run_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
            real_kv_sources_cuda=(),
            real_kv_sources_ref=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
            kernel_kind=tag,
        )
        kk = int(cuda_log.ring[0, _VIOLATION_FIELD_KERNEL_KIND].item())
        assert kk == int(tag)
        assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


def test_empty_plan_no_op() -> None:
    """``verify_num_valid = 0`` → no ring write, no slot_run_counter bump, only kernel_run_counter += 1."""
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    plan_cuda = make_verify_plan(
        slot_indices=[], positions=[], prev_slot_indices=[], capacity=4, device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[], positions=[], prev_slot_indices=[], capacity=4, device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0
    assert int(cuda_log.slot_run_counter[0].item()) == 0
    assert int(cuda_log.kernel_run_counter[0].item()) == 1
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


@pytest.mark.parametrize("hardcoded", [True])
def test_chain_link_byte_equal_5_step_hardcoded(hardcoded: bool) -> None:
    """5-step chain with hand-computed splitmix64 expected sequence; defends against ref + CUDA co-drift."""
    assert hardcoded
    tokens = [101, 202, 303, 404, 505]
    positions = [0, 1, 2, 3, 4]
    slot_indices = [0, 1, 2, 3, 4]
    real_kv_hashes = [0, 0, 0, 0, 0]

    # Step 1: compute the expected stored prev_hash sequence in pure Python via splitmix64.
    expected_prev_hashes_u64: list[int] = []
    running = splitmix64(CANARY_CHAIN_ANCHOR)
    for token, position, real_kv_hash in zip(tokens, positions, real_kv_hashes):
        expected_prev_hashes_u64.append(running)
        running = splitmix64_mix4(running, token, position, real_kv_hash)
    expected_prev_hashes_signed = [to_signed_int64(h) for h in expected_prev_hashes_u64]

    # Step 2: stamp each slot manually with the hardcoded expected prev_hash.
    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    for slot_idx, token, position, prev_hash in zip(
        slot_indices, tokens, positions, expected_prev_hashes_signed
    ):
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=prev_hash,
                real_kv_hash=0,
            )

    # Step 3: verify the 5-step chain — no violation expected and the ref vs CUDA state byte-equal.
    plan_cuda = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=[-1, 0, 1, 2, 3],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=[-1, 0, 1, 2, 3],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0

    # Step 4: also independently confirm the *stored* prev_hash at each slot matches the hardcoded sequence.
    for slot_idx, expected_signed in zip(slot_indices, expected_prev_hashes_signed):
        _, _, stored_prev_hash, _ = read_slot_fields(
            canary_buf=cuda_buf, slot_idx=slot_idx
        )
        assert stored_prev_hash == expected_signed
    assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)


@pytest.mark.parametrize("hardcoded", [True])
def test_chain_link_byte_equal_100_step_hardcoded(hardcoded: bool) -> None:
    assert hardcoded
    import json
    from pathlib import Path

    raw = json.loads(
        (Path(__file__).parent / "testdata" / "chain_100_steps.json").read_text()
    )
    assert len(raw["tokens"]) == 100
    assert len(raw["positions"]) == 100
    assert len(raw["real_kv_hashes"]) == 100
    assert len(raw["expected_prev_hashes"]) == 100

    tokens_int: list[int] = [int(v, 16) for v in raw["tokens"]]
    positions_int: list[int] = raw["positions"]
    real_kv_int: list[int] = [int(v, 16) for v in raw["real_kv_hashes"]]
    expected_u64: list[int] = [int(v, 16) for v in raw["expected_prev_hashes"]]
    expected_signed: list[int] = [to_signed_int64(v) for v in expected_u64]

    cuda_buf = make_canary_buf(num_slots=100, slot_stride_bytes=32, device=_DEVICE)
    ref_buf = cuda_buf.clone()

    for i in range(100):
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=i,
                token=tokens_int[i],
                position=positions_int[i],
                prev_hash=expected_signed[i],
                real_kv_hash=to_signed_int64(real_kv_int[i]),
            )

    slot_indices = list(range(100))
    prev_slot_indices = [-1] + list(range(99))

    plan_cuda = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions_int,
        prev_slot_indices=prev_slot_indices,
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=slot_indices,
        positions=positions_int,
        prev_slot_indices=prev_slot_indices,
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both_and_assert_state_equal(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    assert int(cuda_log.write_index[0].item()) == 0

    for i in range(100):
        _, _, stored_prev_hash, _ = read_slot_fields(canary_buf=cuda_buf, slot_idx=i)
        assert stored_prev_hash == expected_signed[i], (
            f"slot {i}: stored_prev_hash={stored_prev_hash:#x} expected={expected_signed[i]:#x}"
        )


def _hand_fold_bit(raw_bytes: bytes) -> int:
    """BIT-mode fold: parity of low bits across all read bytes, then splitmix64 mix into acc=0."""
    _u64 = (1 << 64) - 1
    parity = sum(b & 1 for b in raw_bytes) & 1
    source_hash = parity
    combined = 0 ^ source_hash
    x = combined & _u64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _u64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _u64
    return (x ^ (x >> 31)) & _u64


def _hand_fold_all(raw_bytes: bytes) -> int:
    """ALL-mode fold: pack bytes little-endian into 8-byte words, fold each via splitmix64, then mix into acc=0."""
    _u64 = (1 << 64) - 1

    def _sm64(v: int) -> int:
        v = v & _u64
        v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & _u64
        v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & _u64
        return (v ^ (v >> 31)) & _u64

    pad = (8 - len(raw_bytes) % 8) % 8
    padded = raw_bytes + bytes(pad)
    num_words = len(padded) // 8
    acc = 0
    for w in range(num_words):
        chunk = padded[w * 8 : (w + 1) * 8]
        word = sum(b << (8 * k) for k, b in enumerate(chunk))
        acc = _sm64(acc ^ word)
    source_hash = acc
    return _sm64(0 ^ source_hash)


@pytest.mark.parametrize("hardcoded", [True])
def test_real_kv_hash_bit_mode_hardcoded(hardcoded: bool) -> None:
    assert hardcoded
    # Step 1: build one RealKvSource with read_bytes=8 and a fixed byte pattern at slot 0.
    _PATTERN = bytes([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80])
    _EXPECTED_HASH: int = 0x5692161D100B05E5

    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    source_cuda = make_real_kv_source(
        num_slots=16, num_bytes_per_token=8, page_size=1, read_bytes=8, device=_DEVICE
    )
    source_cuda.tensor[0, :8] = torch.tensor(list(_PATTERN), dtype=torch.uint8)
    source_ref = RealKvSource(
        tensor=source_cuda.tensor.clone(),
        page_size=source_cuda.page_size,
        num_bytes_per_token=source_cuda.num_bytes_per_token,
        read_bytes=source_cuda.read_bytes,
    )

    # Step 2: verify hand-computed fold matches the hex literal.
    assert _hand_fold_bit(_PATTERN) == _EXPECTED_HASH

    # Step 3: stamp slot 0 with a chain-head entry whose real_kv_hash equals the expected value.
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=7,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=to_signed_int64(_EXPECTED_HASH),
        )

    # Step 4: 1-entry verify plan; no violation because stored matches recomputed.
    plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(source_cuda,),
        real_kv_sources_ref=(source_ref,),
        real_kv_hash_mode=RealKvHashMode.BIT,
    )

    assert int(cuda_log.write_index[0].item()) == 0

    # Step 5: mutate one byte in the source so the recomputed hash diverges from stored.
    source_cuda.tensor[0, 0] ^= 0xFF
    source_ref.tensor.copy_(source_cuda.tensor)

    plan_cuda2 = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref2 = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log2 = FakeViolationLog.allocate(device=_DEVICE)
    ref_log2 = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda2,
        plan_ref=plan_ref2,
        cuda_log=cuda_log2,
        ref_log=ref_log2,
        real_kv_sources_cuda=(source_cuda,),
        real_kv_sources_ref=(source_ref,),
        real_kv_hash_mode=RealKvHashMode.BIT,
    )

    fail_bits = int(cuda_log2.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_REAL_KV_HASH
    assert_canary_state_equal(log_a=cuda_log2, log_b=ref_log2)


@pytest.mark.parametrize("hardcoded", [True])
def test_real_kv_hash_all_mode_hardcoded(hardcoded: bool) -> None:
    assert hardcoded
    # Step 1: build one RealKvSource with read_bytes=8 and a fixed byte pattern at slot 0.
    _PATTERN = bytes([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80])
    _EXPECTED_HASH: int = 0xC4C41792E6578644

    cuda_buf, ref_buf = _setup_pair_with_canned_chain()
    source_cuda = make_real_kv_source(
        num_slots=16, num_bytes_per_token=8, page_size=1, read_bytes=8, device=_DEVICE
    )
    source_cuda.tensor[0, :8] = torch.tensor(list(_PATTERN), dtype=torch.uint8)
    source_ref = RealKvSource(
        tensor=source_cuda.tensor.clone(),
        page_size=source_cuda.page_size,
        num_bytes_per_token=source_cuda.num_bytes_per_token,
        read_bytes=source_cuda.read_bytes,
    )

    # Step 2: verify hand-computed fold matches the hex literal.
    assert _hand_fold_all(_PATTERN) == _EXPECTED_HASH

    # Step 3: stamp slot 0 with a chain-head entry whose real_kv_hash equals the expected value.
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=7,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=to_signed_int64(_EXPECTED_HASH),
        )

    # Step 4: 1-entry verify plan; no violation because stored matches recomputed.
    plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(source_cuda,),
        real_kv_sources_ref=(source_ref,),
        real_kv_hash_mode=RealKvHashMode.ALL,
    )

    assert int(cuda_log.write_index[0].item()) == 0

    # Step 5: mutate one byte in the source so the recomputed hash diverges from stored.
    source_cuda.tensor[0, 0] ^= 0xFF
    source_ref.tensor.copy_(source_cuda.tensor)

    plan_cuda2 = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref2 = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log2 = FakeViolationLog.allocate(device=_DEVICE)
    ref_log2 = FakeViolationLog.allocate(device=_DEVICE)

    _run_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda2,
        plan_ref=plan_ref2,
        cuda_log=cuda_log2,
        ref_log=ref_log2,
        real_kv_sources_cuda=(source_cuda,),
        real_kv_sources_ref=(source_ref,),
        real_kv_hash_mode=RealKvHashMode.ALL,
    )

    fail_bits = int(cuda_log2.ring[0, _VIOLATION_FIELD_FAIL_REASON_BITS].item())
    assert fail_bits & _FAIL_REASON_BIT_REAL_KV_HASH
    assert_canary_state_equal(log_a=cuda_log2, log_b=ref_log2)


def test_chain_advance_formula_matches_spec() -> None:
    """Ref impl agrees with Python-side ``splitmix64(prev XOR token XOR pos XOR real_kv_hash)`` formula."""
    # Step 1: 5 hardcoded 4-tuples covering positive / zero / large prev_hash values.
    cases = [
        (CANARY_CHAIN_ANCHOR, 0, 0, 0),
        (0x1234567890ABCDEF, 100, 5, 0xDEADBEEF),
        (0, 0xFFFF, 0x7FFFFFFF, 0xCAFEBABE),
        (0x123, 1, 1, 1),
        (0xFFFFFFFFFFFFFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    ]
    for prev_hash, token, position, real_kv_hash in cases:
        # Step 2: hand-compute the expected via Python splitmix64.
        expected = splitmix64(
            (prev_hash ^ token ^ position ^ real_kv_hash) & ((1 << 64) - 1)
        )
        # Step 3: compute through ref-impl helper splitmix64_mix4 (same logic, alternate entry point).
        from sglang.jit_kernel.kv_canary.verify_ref import _splitmix64_mix4_vec

        prev_t = torch.tensor([to_signed_int64(prev_hash)], dtype=torch.int64)
        token_t = torch.tensor([to_signed_int64(token)], dtype=torch.int64)
        pos_t = torch.tensor([to_signed_int64(position)], dtype=torch.int64)
        rkv_t = torch.tensor([to_signed_int64(real_kv_hash)], dtype=torch.int64)
        actual_signed = int(
            _splitmix64_mix4_vec(prev_t, token_t, pos_t, rkv_t)[0].item()
        )
        actual = actual_signed & ((1 << 64) - 1)
        assert actual == expected, (
            f"chain advance mismatch: prev={prev_hash:#x} token={token:#x} pos={position:#x} "
            f"rkv={real_kv_hash:#x} expected={expected:#x} actual={actual:#x}"
        )
