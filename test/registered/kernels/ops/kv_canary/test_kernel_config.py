from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.verify import (
    CanaryLaunchTag,
    VerifyOrWriteContext,
    VerifyPlan,
    launch_canary_verify_kernel,
)
from sglang.kernels.ops.kv_canary.write import WritePlan
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.kv_canary._canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_equal,
    make_canary_buf,
    make_canary_buf_pair,
    make_log_pair,
    make_verify_plan,
    make_verify_plan_pair,
    make_write_plan_pair,
    stamp_clean_chain,
)
from sglang.test.kernels.kv_canary._differential import (
    _assert_plans_byte_equal,
    _run_both_plan,
    _run_both_verify,
    _run_both_write,
)
from sglang.test.kernels.kv_canary._fixtures import (
    dummy_pseudo_tensors,
    empty_extras,
)

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

_DEVICE = torch.device("cuda")


def _build_verify_plan_5_entries(
    *, device: torch.device
) -> tuple[VerifyPlan, VerifyPlan]:
    num_slots = 16
    cuda_buf, ref_buf = make_canary_buf_pair(
        num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
    )

    plan_cuda, plan_ref = make_verify_plan_pair(
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
    plan_cuda, plan_ref = make_write_plan_pair(
        write_offsets=[0, num_tokens],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        req_capacity=4,
        device=device,
    )
    input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64, device=device)
    positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    out_cache_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    return plan_cuda, plan_ref, input_ids, positions, out_cache_loc


def _build_plan_fixtures(
    *, device: torch.device, int64_req_to_token: bool = False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bs = 3
    max_reqs = 4
    max_seq_len = 16
    req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
    prefix_lens = torch.tensor([0, 4, 8], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([5, 1, 1], dtype=torch.int64, device=device)
    rp_axis = torch.arange(max_reqs, device=device, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=device, dtype=torch.int32).unsqueeze(0)
    req_to_token_int32 = (rp_axis * max_seq_len + pos_axis).contiguous()
    if int64_req_to_token:
        req_to_token = req_to_token_int32.to(torch.int64)
    else:
        req_to_token = req_to_token_int32
    return req_pool_indices, prefix_lens, extend_seq_lens, req_to_token


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

        _run_both_verify(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
            real_kv_sources_cuda=(),
            real_kv_sources_ref=(),
            real_kv_hash_mode=consts.RealKvHashMode.NONE,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
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
    plan_cuda, plan_ref, input_ids, positions, out_cache_loc = _build_write_fixtures(
        device=_DEVICE
    )

    snapshot_bufs: list[torch.Tensor] = []
    snapshot_rings: list[torch.Tensor] = []
    snapshot_counters: list[torch.Tensor] = []

    for _ in range(num_launches):
        cuda_buf, ref_buf = make_canary_buf_pair(
            num_slots=16, slot_stride_bytes=32, device=_DEVICE
        )
        cuda_log, ref_log = make_log_pair(capacity=64, device=_DEVICE)
        pseudo_tok, pseudo_pos = dummy_pseudo_tensors(input_ids.shape[0])

        _run_both_write(
            cuda_canary_buf=cuda_buf,
            ref_canary_buf=ref_buf,
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            enable_write_verify_inputs=False,
            expected_input_tokens=pseudo_tok,
            expected_input_positions=pseudo_pos,
            cuda_log=cuda_log,
            ref_log=ref_log,
            real_kv_sources_cuda=(),
            real_kv_sources_ref=(),
            real_kv_hash_mode=consts.RealKvHashMode.NONE,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
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
    req_pool_indices, prefix_lens, extend_seq_lens, req_to_token = _build_plan_fixtures(
        device=_DEVICE
    )

    snapshot_slots: list[torch.Tensor] = []
    snapshot_positions: list[torch.Tensor] = []
    snapshot_prevs: list[torch.Tensor] = []
    snapshot_write_offsets: list[torch.Tensor] = []

    for _ in range(num_launches):
        triton_v = VerifyPlan.allocate(
            verify_capacity=64, device=_DEVICE
        ).zero_for_testing_()
        triton_w = WritePlan.allocate(
            write_req_capacity=8, device=_DEVICE
        ).zero_for_testing_()
        ref_v = VerifyPlan.allocate(
            verify_capacity=64, device=_DEVICE
        ).zero_for_testing_()
        ref_w = WritePlan.allocate(
            write_req_capacity=8, device=_DEVICE
        ).zero_for_testing_()

        _run_both_plan(
            triton_verify=triton_v,
            triton_write=triton_w,
            ref_verify=ref_v,
            ref_write=ref_w,
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=extend_seq_lens,
            req_to_token=req_to_token,
            extras=empty_extras(),
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )

        _assert_plans_byte_equal(
            triton_verify=triton_v,
            triton_write=triton_w,
            ref_verify=ref_v,
            ref_write=ref_w,
        )

        n_verify = int(triton_v.verify_num_valid[0].item())
        snapshot_slots.append(triton_v.verify_slot_indices[:n_verify].clone())
        snapshot_positions.append(triton_v.verify_expected_positions[:n_verify].clone())
        snapshot_prevs.append(triton_v.verify_prev_slot_indices[:n_verify].clone())
        snapshot_write_offsets.append(triton_w.write_offsets.clone())

    for i in range(1, num_launches):
        assert torch.equal(
            snapshot_slots[0], snapshot_slots[i]
        ), f"verify_slot_indices differs between launch 0 and {i}"
        assert torch.equal(
            snapshot_positions[0], snapshot_positions[i]
        ), f"verify_expected_positions differs between launch 0 and {i}"
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
        launch_canary_verify_kernel(
            context=VerifyOrWriteContext(
                canary_buf=cuda_buf,
                kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                violation_ring=cuda_log.ring,
                violation_write_index=cuda_log.write_index,
                slot_run_counter=cuda_log.slot_run_counter,
                kernel_run_counter=cuda_log.kernel_run_counter,
                enable_chain_position_assert=cuda_log.enable_chain_position_assert,
                real_kv_sources=(),
                real_kv_hash_mode=consts.RealKvHashMode.NONE,
            ),
            plan=plan_cuda,
            check_verify_expected_token=True,
        )

    torch.cuda.synchronize()

    assert (
        int(cuda_log.kernel_run_counter[0].item()) == num_launches
    ), f"kernel_run_counter expected {num_launches}, got {cuda_log.kernel_run_counter[0].item()}"
    assert int(cuda_log.slot_run_counter[0].item()) == num_launches, (
        f"slot_run_counter expected {num_launches} (1 active entry x 100 launches), "
        f"got {cuda_log.slot_run_counter[0].item()}"
    )


def test_verify_check_disabled_byte_equal() -> None:
    """check_verify_expected_token True vs False produce equivalent violation logs on a clean plan."""
    plan_true_cuda, plan_true_ref = _build_verify_plan_5_entries(device=_DEVICE)
    plan_false_cuda, plan_false_ref = _build_verify_plan_5_entries(device=_DEVICE)

    chain_slot_indices = [0, 1, 2, 3, 4]
    chain_tokens = [10, 20, 30, 40, 50]
    chain_positions = [0, 1, 2, 3, 4]

    cuda_buf_true, ref_buf_true = make_canary_buf_pair(
        num_slots=16, slot_stride_bytes=32, device=_DEVICE
    )
    stamp_clean_chain(
        cuda_buf=cuda_buf_true,
        ref_buf=ref_buf_true,
        slot_indices=chain_slot_indices,
        tokens=chain_tokens,
        positions=chain_positions,
    )
    cuda_buf_false, ref_buf_false = make_canary_buf_pair(
        num_slots=16, slot_stride_bytes=32, device=_DEVICE
    )
    stamp_clean_chain(
        cuda_buf=cuda_buf_false,
        ref_buf=ref_buf_false,
        slot_indices=chain_slot_indices,
        tokens=chain_tokens,
        positions=chain_positions,
    )
    cuda_log_true, ref_log_true = make_log_pair(capacity=64, device=_DEVICE)
    cuda_log_false, ref_log_false = make_log_pair(capacity=64, device=_DEVICE)

    _run_both_verify(
        cuda_canary_buf=cuda_buf_true,
        ref_canary_buf=ref_buf_true,
        plan_cuda=plan_true_cuda,
        plan_ref=plan_true_ref,
        cuda_log=cuda_log_true,
        ref_log=ref_log_true,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=consts.RealKvHashMode.NONE,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        check_verify_expected_token=True,
    )
    _run_both_verify(
        cuda_canary_buf=cuda_buf_false,
        ref_canary_buf=ref_buf_false,
        plan_cuda=plan_false_cuda,
        plan_ref=plan_false_ref,
        cuda_log=cuda_log_false,
        ref_log=ref_log_false,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=consts.RealKvHashMode.NONE,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        check_verify_expected_token=False,
    )

    assert int(cuda_log_true.write_index[0].item()) == 0
    assert int(cuda_log_false.write_index[0].item()) == 0
    assert torch.equal(cuda_log_true.ring, cuda_log_false.ring)
    assert torch.equal(cuda_log_true.write_index, cuda_log_false.write_index)
    assert torch.equal(cuda_log_true.slot_run_counter, cuda_log_false.slot_run_counter)
    assert torch.equal(
        cuda_log_true.kernel_run_counter, cuda_log_false.kernel_run_counter
    )


@pytest.mark.parametrize("per_req_present", [False, True])
def test_plan_per_req_present_or_absent(per_req_present: bool) -> None:
    max_reqs = 4
    max_seq_len = 16
    rp_axis = torch.arange(max_reqs, device=_DEVICE, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    req_to_token = (rp_axis * max_seq_len + pos_axis).contiguous()

    if per_req_present:
        req_pool_indices = torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE)
        prefix_lens = torch.tensor([3, 5], dtype=torch.int64, device=_DEVICE)
        extend_seq_lens = torch.tensor([1, 1], dtype=torch.int64, device=_DEVICE)
    else:
        req_pool_indices = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
        prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
        extend_seq_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)

    triton_v = VerifyPlan.allocate(
        verify_capacity=64, device=_DEVICE
    ).zero_for_testing_()
    triton_w = WritePlan.allocate(
        write_req_capacity=8, device=_DEVICE
    ).zero_for_testing_()
    ref_v = VerifyPlan.allocate(verify_capacity=64, device=_DEVICE).zero_for_testing_()
    ref_w = WritePlan.allocate(write_req_capacity=8, device=_DEVICE).zero_for_testing_()

    _run_both_plan(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        extras=empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    _assert_plans_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
    )

    if not per_req_present:
        assert int(triton_v.verify_num_valid[0].item()) == 0
        bs = int(req_pool_indices.shape[0])
        assert int(triton_w.write_offsets[bs].item()) == 0

    if per_req_present:
        assert int(triton_v.verify_num_valid[0].item()) == 8


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
