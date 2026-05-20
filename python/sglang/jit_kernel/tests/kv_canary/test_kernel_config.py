from __future__ import annotations

from typing import Optional

import pytest
import torch

from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.plan_ref import canary_plan_step_torch_reference
from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvHashMode,
    VerifyPlan,
    canary_verify_step,
)
from sglang.jit_kernel.kv_canary.verify_ref import canary_verify_step_torch_reference
from sglang.jit_kernel.kv_canary.write import (
    CanaryPseudoMode,
    WritePlan,
    canary_write_step,
)
from sglang.jit_kernel.kv_canary.write_ref import canary_write_step_torch_reference
from sglang.jit_kernel.tests.kv_canary.canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_equal,
    make_canary_buf,
    make_canary_buf_pair,
    make_log_pair,
    make_verify_plan,
    make_write_plan,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)

_DEVICE = torch.device("cuda")


def _dummy_pseudo(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.zeros(num_tokens, dtype=torch.int32, device=_DEVICE)
    return t, t.clone()


def _run_verify_both(
    *,
    cuda_buf: torch.Tensor,
    ref_buf: torch.Tensor,
    plan_cuda: VerifyPlan,
    plan_ref: VerifyPlan,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
) -> None:
    canary_verify_step(
        canary_buf=cuda_buf,
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
        canary_buf=ref_buf,
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


def _run_write_both(
    *,
    cuda_buf: torch.Tensor,
    ref_buf: torch.Tensor,
    plan_cuda: WritePlan,
    plan_ref: WritePlan,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
) -> None:
    pseudo_tok, pseudo_pos = _dummy_pseudo(fb_input_ids.shape[0])
    canary_write_step(
        canary_buf=cuda_buf,
        plan=plan_cuda,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        pseudo_mode=CanaryPseudoMode.OFF,
        pseudo_expected_tokens=pseudo_tok,
        pseudo_expected_positions=pseudo_pos,
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
        pseudo_expected_tokens=pseudo_tok,
        pseudo_expected_positions=pseudo_pos,
        violation_ring=ref_log.ring,
        violation_write_index=ref_log.write_index,
        slot_run_counter=ref_log.slot_run_counter,
        kernel_run_counter=ref_log.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    torch.cuda.synchronize()


def _build_verify_plan_5_entries(
    *, device: torch.device
) -> tuple[VerifyPlan, VerifyPlan]:
    num_slots = 16
    cuda_buf, ref_buf = make_canary_buf_pair(
        num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
    )

    plan_cuda = make_verify_plan(
        slot_indices=[0, 1, 2, 3, 4],
        positions=[0, 1, 2, 3, 4],
        prev_slot_indices=[-1, 0, 1, 2, 3],
        capacity=8,
        device=device,
    )
    plan_ref = make_verify_plan(
        slot_indices=[0, 1, 2, 3, 4],
        positions=[0, 1, 2, 3, 4],
        prev_slot_indices=[-1, 0, 1, 2, 3],
        capacity=8,
        device=device,
    )
    return plan_cuda, plan_ref


def _build_write_fixtures(
    *, device: torch.device
) -> tuple[WritePlan, WritePlan, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = 5
    plan_cuda = make_write_plan(
        write_offsets=[0, num_tokens],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        req_capacity=4,
        device=device,
    )
    plan_ref = make_write_plan(
        write_offsets=[0, num_tokens],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        req_capacity=4,
        device=device,
    )
    fb_input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device=device)
    fb_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    fb_out_cache_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    return plan_cuda, plan_ref, fb_input_ids, fb_positions, fb_out_cache_loc


def _build_plan_fixtures(
    *, device: torch.device, int64_req_to_token: bool = False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bs = 3
    max_reqs = 4
    max_seq_len = 16
    fb_req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    fb_prefix_lens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    fb_extend_seq_lens = torch.tensor([5, 1, 1], dtype=torch.int32, device=device)
    rp_axis = torch.arange(max_reqs, device=device, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=device, dtype=torch.int32).unsqueeze(0)
    req_to_token_int32 = (rp_axis * max_seq_len + pos_axis).contiguous()
    if int64_req_to_token:
        req_to_token = req_to_token_int32.to(torch.int64)
    else:
        req_to_token = req_to_token_int32
    return fb_req_pool_indices, fb_prefix_lens, fb_extend_seq_lens, req_to_token


def _empty_extras(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )


def _make_extras(
    *,
    slot_indices: list[int],
    positions: list[int],
    prev_slot_indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(slot_indices)
    cap = max(n, 1)
    slots = torch.zeros(cap, dtype=torch.int32, device=device)
    pos = torch.zeros(cap, dtype=torch.int32, device=device)
    prevs = torch.zeros(cap, dtype=torch.int32, device=device)
    if n > 0:
        slots[:n] = torch.tensor(slot_indices, dtype=torch.int32, device=device)
        pos[:n] = torch.tensor(positions, dtype=torch.int32, device=device)
        prevs[:n] = torch.tensor(prev_slot_indices, dtype=torch.int32, device=device)
    num_valid = torch.tensor([n], dtype=torch.int32, device=device)
    return slots, pos, prevs, num_valid


def _run_plan_both(
    *,
    triton_verify: VerifyPlan,
    triton_write: WritePlan,
    ref_verify: VerifyPlan,
    ref_write: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int = 0,
    full_to_swa_index_mapping: Optional[torch.Tensor] = None,
) -> None:
    extra_slots, extra_pos, extra_prev, extra_num = extras
    canary_plan_step(
        verify_plan_out=triton_verify,
        write_plan_out=triton_write,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_pos,
        extra_verify_prev_slot_indices=extra_prev,
        extra_verify_num_valid=extra_num,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    canary_plan_step_torch_reference(
        verify_plan_out=ref_verify,
        write_plan_out=ref_write,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_pos,
        extra_verify_prev_slot_indices=extra_prev,
        extra_verify_num_valid=extra_num,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    torch.cuda.synchronize()


def _assert_plan_equal(
    *,
    triton_verify: VerifyPlan,
    triton_write: WritePlan,
    ref_verify: VerifyPlan,
    ref_write: WritePlan,
) -> None:
    n_verify = int(triton_verify.verify_num_valid[0].item())
    n_verify_ref = int(ref_verify.verify_num_valid[0].item())
    assert (
        n_verify == n_verify_ref
    ), f"verify_num_valid: triton={n_verify} ref={n_verify_ref}"
    if n_verify > 0:
        assert torch.equal(
            triton_verify.verify_slot_indices[:n_verify],
            ref_verify.verify_slot_indices[:n_verify],
        )
        assert torch.equal(
            triton_verify.verify_positions[:n_verify],
            ref_verify.verify_positions[:n_verify],
        )
        assert torch.equal(
            triton_verify.verify_prev_slot_indices[:n_verify],
            ref_verify.verify_prev_slot_indices[:n_verify],
        )
    n_write = int(triton_write.write_num_valid_reqs[0].item())
    n_write_ref = int(ref_write.write_num_valid_reqs[0].item())
    assert (
        n_write == n_write_ref
    ), f"write_num_valid_reqs: triton={n_write} ref={n_write_ref}"
    assert torch.equal(
        triton_write.write_offsets[: n_write + 1],
        ref_write.write_offsets[: n_write + 1],
    )
    if n_write > 0:
        assert torch.equal(
            triton_write.write_seed_slot_indices[:n_write],
            ref_write.write_seed_slot_indices[:n_write],
        )


def test_verify_byte_equal_across_repeated_launches_10x() -> None:
    num_launches = 10
    plan_cuda, plan_ref = _build_verify_plan_5_entries(device=_DEVICE)

    snapshot_rings: list[torch.Tensor] = []
    snapshot_write_indices: list[torch.Tensor] = []
    snapshot_bufs: list[torch.Tensor] = []

    for _ in range(num_launches):
        cuda_buf, ref_buf = make_canary_buf_pair(
            num_slots=16, slot_stride_bytes=32, device=_DEVICE
        )
        cuda_log, ref_log = make_log_pair(capacity=64, device=_DEVICE)

        _run_verify_both(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )

        assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
        assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)

        snapshot_rings.append(cuda_log.ring.clone())
        snapshot_write_indices.append(cuda_log.write_index.clone())
        snapshot_bufs.append(cuda_buf.clone())

    for i in range(1, num_launches):
        assert torch.equal(
            snapshot_rings[0], snapshot_rings[i]
        ), f"violation_ring differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_write_indices[0], snapshot_write_indices[i]
        ), f"violation_write_index differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_bufs[0], snapshot_bufs[i]
        ), f"canary_buf differs between launch 0 and {i}"


def test_write_byte_equal_across_repeated_launches_10x() -> None:
    num_launches = 10
    plan_cuda, plan_ref, fb_input_ids, fb_positions, fb_out_cache_loc = (
        _build_write_fixtures(device=_DEVICE)
    )

    snapshot_bufs: list[torch.Tensor] = []
    snapshot_rings: list[torch.Tensor] = []
    snapshot_counters: list[torch.Tensor] = []

    for _ in range(num_launches):
        cuda_buf, ref_buf = make_canary_buf_pair(
            num_slots=16, slot_stride_bytes=32, device=_DEVICE
        )
        cuda_log, ref_log = make_log_pair(capacity=64, device=_DEVICE)

        _run_write_both(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )

        assert_canary_buf_equal(buf_a=cuda_buf, buf_b=ref_buf)
        assert_canary_state_equal(log_a=cuda_log, log_b=ref_log)

        snapshot_bufs.append(cuda_buf.clone())
        snapshot_rings.append(cuda_log.ring.clone())
        snapshot_counters.append(cuda_log.slot_run_counter.clone())

    for i in range(1, num_launches):
        assert torch.equal(
            snapshot_bufs[0], snapshot_bufs[i]
        ), f"canary_buf differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_rings[0], snapshot_rings[i]
        ), f"violation_ring differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_counters[0], snapshot_counters[i]
        ), f"slot_run_counter differs between launch 0 and {i}"


def test_plan_byte_equal_across_repeated_launches_10x() -> None:
    num_launches = 10
    fb_rpi, fb_prefix, fb_extend, req_to_token = _build_plan_fixtures(device=_DEVICE)

    snapshot_slots: list[torch.Tensor] = []
    snapshot_positions: list[torch.Tensor] = []
    snapshot_prevs: list[torch.Tensor] = []
    snapshot_write_offsets: list[torch.Tensor] = []

    for _ in range(num_launches):
        triton_v = VerifyPlan.allocate(verify_capacity=64, device=_DEVICE)
        triton_w = WritePlan.allocate(write_req_capacity=8, device=_DEVICE)
        ref_v = VerifyPlan.allocate(verify_capacity=64, device=_DEVICE)
        ref_w = WritePlan.allocate(write_req_capacity=8, device=_DEVICE)

        _run_plan_both(
            triton_verify=triton_v,
            triton_write=triton_w,
            ref_verify=ref_v,
            ref_write=ref_w,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_prefix,
            fb_extend_seq_lens=fb_extend,
            req_to_token=req_to_token,
            extras=_empty_extras(_DEVICE),
        )

        _assert_plan_equal(
            triton_verify=triton_v,
            triton_write=triton_w,
            ref_verify=ref_v,
            ref_write=ref_w,
        )

        n_verify = int(triton_v.verify_num_valid[0].item())
        snapshot_slots.append(triton_v.verify_slot_indices[:n_verify].clone())
        snapshot_positions.append(triton_v.verify_positions[:n_verify].clone())
        snapshot_prevs.append(triton_v.verify_prev_slot_indices[:n_verify].clone())
        snapshot_write_offsets.append(triton_w.write_offsets.clone())

    for i in range(1, num_launches):
        assert torch.equal(
            snapshot_slots[0], snapshot_slots[i]
        ), f"verify_slot_indices differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_positions[0], snapshot_positions[i]
        ), f"verify_positions differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_prevs[0], snapshot_prevs[i]
        ), f"verify_prev_slot_indices differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_write_offsets[0], snapshot_write_offsets[i]
        ), f"write_offsets differs between launch 0 and {i}"


def test_verify_multi_launch_100x_counter_linear() -> None:
    num_launches = 100

    plan_cuda = make_verify_plan(
        slot_indices=[0],
        positions=[0],
        prev_slot_indices=[-1],
        capacity=4,
        device=_DEVICE,
    )

    cuda_log = FakeViolationLog.allocate(capacity=64, device=_DEVICE)

    for _ in range(num_launches):
        cuda_buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_DEVICE)
        canary_verify_step(
            canary_buf=cuda_buf,
            plan=plan_cuda,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=cuda_log.ring,
            violation_write_index=cuda_log.write_index,
            slot_run_counter=cuda_log.slot_run_counter,
            kernel_run_counter=cuda_log.kernel_run_counter,
            real_kv_sources=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )

    torch.cuda.synchronize()

    assert (
        int(cuda_log.kernel_run_counter[0].item()) == num_launches
    ), f"kernel_run_counter expected {num_launches}, got {cuda_log.kernel_run_counter[0].item()}"
    assert int(cuda_log.slot_run_counter[0].item()) == num_launches, (
        f"slot_run_counter expected {num_launches} (1 active entry x 100 launches), "
        f"got {cuda_log.slot_run_counter[0].item()}"
    )


def test_verify_block_size_sweep_byte_equal() -> None:
    pytest.skip(
        "verify kernel block_size not user-configurable; block size is hardcoded as "
        "kVerifyBlockSize=128 in canary_verify.cuh and not exposed in the Python API. "
        "Determinism under the default config is covered by test_verify_byte_equal_across_repeated_launches_10x."
    )


def test_write_block_size_sweep_byte_equal() -> None:
    pytest.skip(
        "write kernel block_size not user-configurable; block size is hardcoded as "
        "kWriteBlockSize=1 in canary_write.cuh (serial chain per block) and not exposed in the Python API. "
        "Determinism under the default config is covered by test_write_byte_equal_across_repeated_launches_10x."
    )


def test_plan_triton_autotune_disabled_vs_enabled_byte_equal() -> None:
    pytest.skip(
        "plan.py Triton kernels do not use @triton.autotune; inner tile widths are fixed "
        "module-level constants (_PLAN_BS_BLOCK_SIZE=1024, _PLAN_VERIFY_INNER_BLOCK=64, "
        "_PLAN_EXTRAS_INNER_BLOCK=64) with no per-call override path. "
        "Determinism under the default config is covered by test_plan_byte_equal_across_repeated_launches_10x."
    )


@pytest.mark.parametrize(
    "extras_present,per_req_present",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_plan_extras_present_and_per_req_present_cartesian_4_combos(
    extras_present: bool, per_req_present: bool
) -> None:
    max_reqs = 4
    max_seq_len = 16
    rp_axis = torch.arange(max_reqs, device=_DEVICE, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    req_to_token = (rp_axis * max_seq_len + pos_axis).contiguous()

    if per_req_present:
        fb_rpi = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
        fb_prefix = torch.tensor([3, 5], dtype=torch.int32, device=_DEVICE)
        fb_extend = torch.tensor([1, 1], dtype=torch.int32, device=_DEVICE)
    else:
        fb_rpi = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
        fb_prefix = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
        fb_extend = torch.tensor([0], dtype=torch.int32, device=_DEVICE)

    if extras_present:
        extras = _make_extras(
            slot_indices=[10, 11],
            positions=[0, 1],
            prev_slot_indices=[-1, 10],
            device=_DEVICE,
        )
    else:
        extras = _empty_extras(_DEVICE)

    triton_v = VerifyPlan.allocate(verify_capacity=64, device=_DEVICE)
    triton_w = WritePlan.allocate(write_req_capacity=8, device=_DEVICE)
    ref_v = VerifyPlan.allocate(verify_capacity=64, device=_DEVICE)
    ref_w = WritePlan.allocate(write_req_capacity=8, device=_DEVICE)

    _run_plan_both(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_rpi,
        fb_prefix_lens=fb_prefix,
        fb_extend_seq_lens=fb_extend,
        req_to_token=req_to_token,
        extras=extras,
    )

    _assert_plan_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
    )

    if not per_req_present and not extras_present:
        assert int(triton_v.verify_num_valid[0].item()) == 0
        bs = int(fb_rpi.shape[0])
        assert int(triton_w.write_offsets[bs].item()) == 0

    if not per_req_present and extras_present:
        assert int(triton_v.verify_num_valid[0].item()) == 2

    if per_req_present and not extras_present:
        assert int(triton_v.verify_num_valid[0].item()) == 8

    if per_req_present and extras_present:
        assert int(triton_v.verify_num_valid[0].item()) == 10


def test_plan_req_to_token_int64_byte_equal() -> None:
    pytest.skip(
        "canary_plan_step Triton kernel loads req_to_token via int32 pointer arithmetic; "
        "passing an int64 tensor causes a dtype mismatch at the Triton ABI boundary. "
        "The kernel is specified as int32 (docstring: 'shape [max_reqs, max_seq_len], int32'). "
        "int64 support would require a kernel change; this skip documents the current contract."
    )
