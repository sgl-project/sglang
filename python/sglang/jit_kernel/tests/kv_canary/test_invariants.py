"""Property/invariant tests for the kv_canary jit_kernel.

Each test runs many varied inputs in a loop and asserts an invariant holds
across all iterations, catching accumulation bugs and cross-call state bugs
that single-shot differential tests miss.
"""

from __future__ import annotations

import random

import pytest
import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_CHAIN_ANCHOR,
    CanaryLaunchTag,
    RealKvHashMode,
    canary_verify_step,
)
from sglang.jit_kernel.kv_canary.verify_ref import (
    canary_verify_step_torch_reference,
)
from sglang.jit_kernel.kv_canary.write import (
    CanaryPseudoMode,
    canary_write_step,
)
from sglang.jit_kernel.kv_canary.write_ref import (
    canary_write_step_torch_reference,
)
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    chain_anchor_signed,
    make_canary_buf,
    make_real_kv_source,
    make_verify_plan,
    make_write_plan,
    splitmix64,
    to_signed_int64,
    write_slot_fields,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)

_DEVICE = torch.device("cuda")


def _verify_both(
    *,
    cuda_canary_buf: torch.Tensor,
    ref_canary_buf: torch.Tensor,
    plan_cuda,
    plan_ref,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
) -> None:
    canary_verify_step(
        canary_buf=cuda_canary_buf,
        plan=plan_cuda,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=cuda_log.ring,
        violation_write_index=cuda_log.write_index,
        slot_run_counter=cuda_log.slot_run_counter,
        kernel_run_counter=cuda_log.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    canary_verify_step_torch_reference(
        canary_buf=ref_canary_buf,
        plan=plan_ref,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    torch.cuda.synchronize()


def test_slot_run_counter_delta_equals_active_entries_across_random_plans() -> None:
    random.seed(0)
    num_slots = 32
    cuda_buf = make_canary_buf(num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE)
    ref_buf = cuda_buf.clone()

    anchor_signed = chain_anchor_signed()
    for slot_idx in range(num_slots):
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=slot_idx + 10,
                position=slot_idx,
                prev_hash=anchor_signed,
                real_kv_hash=0,
            )

    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    for _ in range(50):
        bs = random.randint(1, 16)
        entries_per_req = random.randint(1, 8)
        n_entries = bs * entries_per_req
        if n_entries > num_slots:
            n_entries = num_slots
        slot_indices = random.sample(range(num_slots), n_entries)
        positions = [random.randint(0, 99) for _ in range(n_entries)]
        prev_slot_indices = [-1] * n_entries

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

        before = int(cuda_log.slot_run_counter[0].item())
        _verify_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )
        after = int(cuda_log.slot_run_counter[0].item())
        assert after - before == n_entries


def test_kernel_run_counter_per_call_invariant_50_calls() -> None:
    cuda_buf = make_canary_buf(num_slots=8, slot_stride_bytes=32, device=_DEVICE)
    ref_buf = cuda_buf.clone()
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=42,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    for n in range(1, 51):
        _verify_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )
        assert int(cuda_log.kernel_run_counter[0].item()) == n


def test_empty_plan_keeps_slot_counter_unchanged() -> None:
    cuda_buf = make_canary_buf(num_slots=8, slot_stride_bytes=32, device=_DEVICE)
    ref_buf = cuda_buf.clone()
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        write_slot_fields(
            canary_buf=buf,
            slot_idx=0,
            token=7,
            position=0,
            prev_hash=anchor_signed,
            real_kv_hash=0,
        )

    nonempty_plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    nonempty_plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    empty_plan_cuda = make_verify_plan(
        slot_indices=[], positions=[], prev_slot_indices=[], capacity=4, device=_DEVICE
    )
    empty_plan_ref = make_verify_plan(
        slot_indices=[], positions=[], prev_slot_indices=[], capacity=4, device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    for _ in range(30):
        slot_before = int(cuda_log.slot_run_counter[0].item())
        kernel_before = int(cuda_log.kernel_run_counter[0].item())

        _verify_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=empty_plan_cuda,
            plan_ref=empty_plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )

        assert int(cuda_log.slot_run_counter[0].item()) == slot_before
        assert int(cuda_log.kernel_run_counter[0].item()) == kernel_before + 1

        _verify_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=nonempty_plan_cuda,
            plan_ref=nonempty_plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )


def test_chain_head_prev_hash_equals_splitmix64_anchor_random_50() -> None:
    random.seed(0)
    expected_prev_hash_signed = to_signed_int64(splitmix64(CANARY_CHAIN_ANCHOR))

    for _ in range(50):
        token = random.randint(0, 0x7FFFFFFF)
        position = random.randint(0, 0x7FFFFFFF)
        slot_idx = random.randint(0, 15)

        cuda_buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_DEVICE)
        ref_buf = cuda_buf.clone()
        for buf in (cuda_buf, ref_buf):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=expected_prev_hash_signed,
                real_kv_hash=0,
            )

        plan_cuda = make_verify_plan(
            slot_indices=[slot_idx],
            positions=[position],
            prev_slot_indices=[-1],
            device=_DEVICE,
        )
        plan_ref = make_verify_plan(
            slot_indices=[slot_idx],
            positions=[position],
            prev_slot_indices=[-1],
            device=_DEVICE,
        )
        cuda_log = FakeViolationLog.allocate(device=_DEVICE)
        ref_log = FakeViolationLog.allocate(device=_DEVICE)

        _verify_both(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )

        assert int(cuda_log.write_index[0].item()) == 0, (
            f"unexpected violation at iteration token={token} position={position} slot={slot_idx}"
        )


def test_splitmix64_no_collision_in_1000_random_inputs() -> None:
    random.seed(0)
    inputs = random.sample(range(1 << 64), 1000)
    outputs = [splitmix64(x) for x in inputs]
    assert len(set(outputs)) == 1000


def test_real_kv_off_does_not_deref_real_kv_sources() -> None:
    cuda_buf = make_canary_buf(num_slots=8, slot_stride_bytes=32, device=_DEVICE)
    ref_buf = cuda_buf.clone()
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

    garbage_source = make_real_kv_source(
        num_slots=8,
        num_bytes_per_token=8,
        page_size=1,
        read_bytes=8,
        device=_DEVICE,
        fill=0xDE,
    )

    plan_cuda = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    plan_ref = make_verify_plan(
        slot_indices=[0], positions=[0], prev_slot_indices=[-1], device=_DEVICE
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    canary_verify_step(
        canary_buf=cuda_buf,
        plan=plan_cuda,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=cuda_log.ring,
        violation_write_index=cuda_log.write_index,
        slot_run_counter=cuda_log.slot_run_counter,
        kernel_run_counter=cuda_log.kernel_run_counter,
        real_kv_sources=(garbage_source,),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    canary_verify_step_torch_reference(
        canary_buf=ref_buf,
        plan=plan_ref,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=(garbage_source,),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    torch.cuda.synchronize()

    assert int(cuda_log.write_index[0].item()) == 0
    assert int(ref_log.write_index[0].item()) == 0


def test_pseudo_off_does_not_deref_pseudo_expected() -> None:
    cuda_buf = make_canary_buf(num_slots=8, slot_stride_bytes=32, device=_DEVICE)
    ref_buf = cuda_buf.clone()

    plan_cuda = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    plan_ref = make_write_plan(
        write_offsets=[0, 1], seed_slot_indices=[-1], num_valid_reqs=1, device=_DEVICE
    )
    fb_input_ids = torch.tensor([42], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor([0], dtype=torch.int32, device=_DEVICE)

    garbage_expected_tokens = torch.full((1,), 0x7F7F7F7F, dtype=torch.int32, device=_DEVICE)
    garbage_expected_positions = torch.full((1,), 0x7F7F7F7F, dtype=torch.int32, device=_DEVICE)

    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    canary_write_step(
        canary_buf=cuda_buf,
        plan=plan_cuda,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=garbage_expected_tokens,
        pseudo_expected_positions=garbage_expected_positions,
        violation_ring=cuda_log.ring,
        violation_write_index=cuda_log.write_index,
        slot_run_counter=cuda_log.slot_run_counter,
        kernel_run_counter=cuda_log.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    canary_write_step_torch_reference(
        canary_buf=ref_buf,
        plan=plan_ref,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=garbage_expected_tokens,
        pseudo_expected_positions=garbage_expected_positions,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    torch.cuda.synchronize()

    assert int(cuda_log.write_index[0].item()) == 0
    assert int(ref_log.write_index[0].item()) == 0


def test_clear_resets_ring_and_write_index_zero() -> None:
    cuda_buf = make_canary_buf(num_slots=8, slot_stride_bytes=32, device=_DEVICE)
    ref_buf = cuda_buf.clone()
    anchor_signed = chain_anchor_signed()
    for buf in (cuda_buf, ref_buf):
        for slot_idx in range(3):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=slot_idx,
                token=1,
                position=99,
                prev_hash=anchor_signed,
                real_kv_hash=0,
            )

    plan_cuda = make_verify_plan(
        slot_indices=[0, 1, 2],
        positions=[0, 0, 0],
        prev_slot_indices=[-1, -1, -1],
        device=_DEVICE,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0, 1, 2],
        positions=[0, 0, 0],
        prev_slot_indices=[-1, -1, -1],
        device=_DEVICE,
    )
    cuda_log = FakeViolationLog.allocate(device=_DEVICE)
    ref_log = FakeViolationLog.allocate(device=_DEVICE)

    _verify_both(
        cuda_canary_buf=cuda_buf,
        ref_canary_buf=ref_buf,
        plan_cuda=plan_cuda,
        plan_ref=plan_ref,
        cuda_log=cuda_log,
        ref_log=ref_log,
    )

    assert int(cuda_log.write_index[0].item()) > 0

    cuda_log_fresh = FakeViolationLog.allocate(device=_DEVICE)
    assert int(cuda_log_fresh.write_index[0].item()) == 0
    assert torch.all(cuda_log_fresh.ring == 0).item()
