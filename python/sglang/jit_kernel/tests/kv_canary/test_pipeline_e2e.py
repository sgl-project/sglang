"""E2E differential test: real plan → write → verify pipeline vs torch-reference pipeline, byte-equal."""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.plan_ref import canary_plan_step_torch_reference
from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
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
    make_real_kv_sources,
    write_slot_fields,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _build_req_to_token(*, max_reqs: int, max_seq_len: int) -> torch.Tensor:
    rp_axis = torch.arange(max_reqs, device=_DEVICE, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    return (rp_axis * max_seq_len + pos_axis).contiguous()


def _empty_extras() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
    )


def _run_pipeline_real(
    *,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    req_to_token: torch.Tensor,
    canary_buf: torch.Tensor,
    log: FakeViolationLog,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    kernel_kind: CanaryLaunchTag,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan]:
    extra_slots, extra_positions, extra_prev_slots, extra_num_valid = extras
    plan_v = VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE)
    plan_w = WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE)
    canary_plan_step(
        verify_plan_out=plan_v,
        write_plan_out=plan_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_positions,
        extra_verify_prev_slot_indices=extra_prev_slots,
        extra_verify_num_valid=extra_num_valid,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    canary_write_step(
        canary_buf=canary_buf,
        plan=plan_w,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        violation_ring=log.ring,
        violation_write_index=log.write_index,
        slot_run_counter=log.slot_run_counter,
        kernel_run_counter=log.kernel_run_counter,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    canary_verify_step(
        canary_buf=canary_buf,
        plan=plan_v,
        kernel_kind=kernel_kind,
        violation_ring=log.ring,
        violation_write_index=log.write_index,
        slot_run_counter=log.slot_run_counter,
        kernel_run_counter=log.kernel_run_counter,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    torch.cuda.synchronize()
    return plan_v, plan_w


def _run_pipeline_ref(
    *,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    req_to_token: torch.Tensor,
    canary_buf: torch.Tensor,
    log: FakeViolationLog,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    kernel_kind: CanaryLaunchTag,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: RealKvHashMode,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan]:
    extra_slots, extra_positions, extra_prev_slots, extra_num_valid = extras
    plan_v = VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE)
    plan_w = WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE)
    canary_plan_step_torch_reference(
        verify_plan_out=plan_v,
        write_plan_out=plan_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_positions,
        extra_verify_prev_slot_indices=extra_prev_slots,
        extra_verify_num_valid=extra_num_valid,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    canary_write_step_torch_reference(
        canary_buf=canary_buf,
        plan=plan_w,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        violation_ring=log.ring,
        violation_write_index=log.write_index,
        slot_run_counter=log.slot_run_counter,
        kernel_run_counter=log.kernel_run_counter,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    canary_verify_step_torch_reference(
        canary_buf=canary_buf,
        plan=plan_v,
        kernel_kind=kernel_kind,
        violation_ring=log.ring,
        violation_write_index=log.write_index,
        slot_run_counter=log.slot_run_counter,
        kernel_run_counter=log.kernel_run_counter,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    return plan_v, plan_w


def _run_both_and_assert_pipeline_equal(
    *,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    req_to_token: torch.Tensor,
    num_slots: int,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int = 0,
    full_to_swa_index_mapping: Optional[torch.Tensor] = None,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
    pseudo_mode: CanaryPseudoMode = CanaryPseudoMode.OFF,
    pseudo_expected_tokens: Optional[torch.Tensor] = None,
    pseudo_expected_positions: Optional[torch.Tensor] = None,
    real_kv_sources_real: tuple[RealKvSource, ...] = (),
    real_kv_sources_ref: tuple[RealKvSource, ...] = (),
    real_kv_hash_mode: RealKvHashMode = RealKvHashMode.OFF,
    ring_capacity: int = 64,
    verify_capacity: int = 256,
    write_req_capacity: int = 16,
    assert_ring_equal: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    FakeViolationLog,
    FakeViolationLog,
    VerifyPlan,
    WritePlan,
    VerifyPlan,
    WritePlan,
]:
    total_tokens = int(fb_input_ids.shape[0])
    if pseudo_expected_tokens is None:
        pseudo_expected_tokens = torch.zeros(
            total_tokens, dtype=torch.int32, device=_DEVICE
        )
    if pseudo_expected_positions is None:
        pseudo_expected_positions = torch.zeros(
            total_tokens, dtype=torch.int32, device=_DEVICE
        )

    buf_real = make_canary_buf(num_slots=num_slots, device=_DEVICE)
    buf_ref = make_canary_buf(num_slots=num_slots, device=_DEVICE)
    log_real = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)
    log_ref = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)

    plan_v_real, plan_w_real = _run_pipeline_real(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        canary_buf=buf_real,
        log=log_real,
        extras=extras,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        real_kv_sources=real_kv_sources_real,
        real_kv_hash_mode=real_kv_hash_mode,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
    )
    plan_v_ref, plan_w_ref = _run_pipeline_ref(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        canary_buf=buf_ref,
        log=log_ref,
        extras=extras,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        kernel_kind=kernel_kind,
        pseudo_mode=pseudo_mode,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        real_kv_sources=real_kv_sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
    )

    assert_canary_buf_equal(buf_a=buf_real, buf_b=buf_ref)
    if assert_ring_equal:
        assert_canary_state_equal(log_a=log_real, log_b=log_ref)
    else:
        assert torch.equal(log_real.write_index, log_ref.write_index)
        assert torch.equal(log_real.slot_run_counter, log_ref.slot_run_counter)
        assert torch.equal(log_real.kernel_run_counter, log_ref.kernel_run_counter)

    return (
        buf_real,
        buf_ref,
        log_real,
        log_ref,
        plan_v_real,
        plan_w_real,
        plan_v_ref,
        plan_w_ref,
    )


def _clone_real_kv_sources(
    sources: tuple[RealKvSource, ...],
) -> tuple[RealKvSource, ...]:
    return tuple(
        RealKvSource(
            tensor=src.tensor.clone(),
            page_size=src.page_size,
            num_bytes_per_token=src.num_bytes_per_token,
            read_bytes=src.read_bytes,
        )
        for src in sources
    )


def test_pipeline_basic_5_step_single_req() -> None:
    """Single req, prefix_len=0, extend_seq_len=5: basic plan→write→verify byte-equal."""
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([5], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            1 * max_seq_len + 0,
            1 * max_seq_len + 1,
            1 * max_seq_len + 2,
            1 * max_seq_len + 3,
            1 * max_seq_len + 4,
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )

    _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=64,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )


def test_pipeline_multi_req_mixed_extend_decode() -> None:
    """bs=3: pure extend req, decode req (prefix+1 extend), and padding sentinel row."""
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=8, max_seq_len=max_seq_len)
    fb_req_pool_indices = torch.tensor([1, 2, 0], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0, 5, 0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([4, 1, 0], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.tensor([11, 12, 13, 14, 21], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            1 * max_seq_len + 0,
            1 * max_seq_len + 1,
            1 * max_seq_len + 2,
            1 * max_seq_len + 3,
            2 * max_seq_len + 5,
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )

    _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=128,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        write_req_capacity=4,
    )


def test_pipeline_swa_window() -> None:
    """SWA window=4, prefix_len=6: verify covers window [2,6), write covers extend tokens."""
    max_seq_len = 16
    num_slots_full = 64
    max_reqs = 4
    req_to_token = _build_req_to_token(max_reqs=max_reqs, max_seq_len=max_seq_len)

    swa_window_size = 4
    full_pool_size = max_reqs * max_seq_len

    full_to_swa_index_mapping = torch.arange(
        full_pool_size + 1, dtype=torch.int32, device=_DEVICE
    )

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([6], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([2], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.tensor([100, 101], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([6, 7], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            full_to_swa_index_mapping[1 * max_seq_len + 6].item(),
            full_to_swa_index_mapping[1 * max_seq_len + 7].item(),
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )

    _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=num_slots_full,
        extras=_empty_extras(),
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )


def test_pipeline_sweep_no_write() -> None:
    """All extend_seq_lens=0: write_step is no-op, verify sweeps prefix, buf unchanged."""
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.zeros(1, dtype=torch.int32, device=_DEVICE)

    _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=64,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )


@pytest.mark.parametrize(
    "real_kv_hash_mode", [RealKvHashMode.OFF, RealKvHashMode.BIT, RealKvHashMode.ALL]
)
def test_pipeline_real_kv_mode(real_kv_hash_mode: RealKvHashMode) -> None:
    """real_kv_hash_mode OFF/BIT/ALL: real and ref use cloned sources to prevent ALL-mode hash aliasing."""
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.tensor([5, 6, 7], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [1 * max_seq_len + 0, 1 * max_seq_len + 1, 1 * max_seq_len + 2],
        dtype=torch.int32,
        device=_DEVICE,
    )

    sources_real = make_real_kv_sources(count=2, num_slots=64, device=_DEVICE)
    sources_ref = _clone_real_kv_sources(sources_real)

    _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=64,
        extras=_empty_extras(),
        real_kv_sources_real=sources_real,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
    )


def test_pipeline_pseudo_mode_on_match() -> None:
    """pseudo_mode=ON, expected==actual: zero violations, buf byte-equal."""
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([4], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            1 * max_seq_len + 0,
            1 * max_seq_len + 1,
            1 * max_seq_len + 2,
            1 * max_seq_len + 3,
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )
    pseudo_expected_tokens = fb_input_ids.clone()
    pseudo_expected_positions = fb_positions.clone()

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=64,
        extras=_empty_extras(),
        pseudo_mode=CanaryPseudoMode.ON,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
    )

    assert int(log_real.write_index[0].item()) == 0
    assert int(log_ref.write_index[0].item()) == 0


def test_pipeline_pseudo_mode_on_token_mismatch_then_verify_clean() -> None:
    """pseudo_mode=ON, expected tokens all wrong: write records N violations, verify does not cascade CHAIN_HASH."""
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    n_tokens = 3
    fb_input_ids = torch.tensor([10, 20, 30], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [1 * max_seq_len + 0, 1 * max_seq_len + 1, 1 * max_seq_len + 2],
        dtype=torch.int32,
        device=_DEVICE,
    )
    pseudo_expected_tokens = torch.tensor(
        [99, 99, 99], dtype=torch.int32, device=_DEVICE
    )
    pseudo_expected_positions = fb_positions.clone()

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=64,
        extras=_empty_extras(),
        pseudo_mode=CanaryPseudoMode.ON,
        pseudo_expected_tokens=pseudo_expected_tokens,
        pseudo_expected_positions=pseudo_expected_positions,
        ring_capacity=64,
    )

    write_violations = int(log_real.write_index[0].item())
    assert (
        write_violations == n_tokens
    ), f"expected {n_tokens} write violations, got {write_violations}"


def test_pipeline_empty_batch() -> None:
    """bs=1 with req_pool_idx=0 (padding): write and verify are no-op, kernel_run_counter == 2 (write+verify)."""
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    fb_req_pool_indices = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.zeros(1, dtype=torch.int32, device=_DEVICE)

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=64,
        extras=_empty_extras(),
    )

    assert int(log_real.kernel_run_counter[0].item()) == 2
    assert int(log_ref.kernel_run_counter[0].item()) == 2
    assert int(log_real.write_index[0].item()) == 0


def test_pipeline_negative_slot_swa_out_of_window() -> None:
    """SWA: some fb_out_cache_loc entries map to -1 (out-of-window); write_step skips them, buf unchanged."""
    max_seq_len = 16
    max_reqs = 4
    full_pool_size = max_reqs * max_seq_len
    req_to_token = _build_req_to_token(max_reqs=max_reqs, max_seq_len=max_seq_len)
    swa_window_size = 4

    full_to_swa_index_mapping = torch.arange(
        full_pool_size + 1, dtype=torch.int32, device=_DEVICE
    )
    full_to_swa_index_mapping[1 * max_seq_len + 6] = -1
    full_to_swa_index_mapping[1 * max_seq_len + 7] = -1

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([6], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([4], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.tensor([100, 101, 102, 103], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([6, 7, 8, 9], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [-1, -1, 1 * max_seq_len + 8, 1 * max_seq_len + 9],
        dtype=torch.int32,
        device=_DEVICE,
    )

    _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=128,
        extras=_empty_extras(),
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )


def test_pipeline_ring_overflow_via_real_plan() -> None:
    """Verify detects >capacity violations when prev_hash is pre-corrupted; write_index byte-equal, ring relaxed."""
    max_seq_len = 16
    max_reqs = 4
    req_to_token = _build_req_to_token(max_reqs=max_reqs, max_seq_len=max_seq_len)
    n_slots = 8

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([n_slots], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.zeros(1, dtype=torch.int32, device=_DEVICE)

    num_slots = max_reqs * max_seq_len

    # Step 1: pre-pollute canary_buf slots [0..n_slots) with wrong prev_hash so verify fires n_slots violations.
    buf_real = make_canary_buf(num_slots=num_slots, device=_DEVICE)
    buf_ref = make_canary_buf(num_slots=num_slots, device=_DEVICE)
    for slot_idx in range(n_slots):
        full_slot = 1 * max_seq_len + slot_idx
        for buf in (buf_real, buf_ref):
            write_slot_fields(
                canary_buf=buf,
                slot_idx=full_slot,
                token=slot_idx + 1,
                position=slot_idx,
                prev_hash=0x1234_DEAD_BEEF_0000 + slot_idx,
                real_kv_hash=0,
            )

    # Step 2: run real pipeline (plan + no write + verify); overflow ring capacity=4 with all n_slots violations.
    ring_capacity = 4
    log_real = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)
    log_ref = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)
    extras = _empty_extras()
    extra_slots, extra_positions, extra_prev_slots, extra_num_valid = extras

    plan_v_real = VerifyPlan.allocate(verify_capacity=256, device=_DEVICE)
    plan_w_real = WritePlan.allocate(write_req_capacity=4, device=_DEVICE)
    plan_v_ref = VerifyPlan.allocate(verify_capacity=256, device=_DEVICE)
    plan_w_ref = WritePlan.allocate(write_req_capacity=4, device=_DEVICE)

    canary_plan_step(
        verify_plan_out=plan_v_real,
        write_plan_out=plan_w_real,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_positions,
        extra_verify_prev_slot_indices=extra_prev_slots,
        extra_verify_num_valid=extra_num_valid,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )
    canary_plan_step_torch_reference(
        verify_plan_out=plan_v_ref,
        write_plan_out=plan_w_ref,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_positions,
        extra_verify_prev_slot_indices=extra_prev_slots,
        extra_verify_num_valid=extra_num_valid,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    canary_verify_step(
        canary_buf=buf_real,
        plan=plan_v_real,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=log_real.ring,
        violation_write_index=log_real.write_index,
        slot_run_counter=log_real.slot_run_counter,
        kernel_run_counter=log_real.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )
    torch.cuda.synchronize()

    canary_verify_step_torch_reference(
        canary_buf=buf_ref,
        plan=plan_v_ref,
        kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=log_ref.ring,
        violation_write_index=log_ref.write_index,
        slot_run_counter=log_ref.slot_run_counter,
        kernel_run_counter=log_ref.kernel_run_counter,
        real_kv_sources=(),
        real_kv_hash_mode=RealKvHashMode.OFF,
    )

    # Step 3: write_index byte-equal; ring contents relaxed (atomic order not guaranteed under overflow).
    assert torch.equal(log_real.write_index, log_ref.write_index)
    assert int(log_real.write_index[0].item()) == n_slots


@pytest.mark.parametrize(
    "kernel_kind", [CanaryLaunchTag.HEAD_K_FULL, CanaryLaunchTag.SWEEP_V_SWA]
)
def test_pipeline_kernel_kind_propagates(kernel_kind: CanaryLaunchTag) -> None:
    """Different CanaryLaunchTag values: violation ring's kernel_kind field matches on both sides."""

    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_input_ids = torch.tensor([7], dtype=torch.int32, device=_DEVICE)
    fb_positions = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [1 * max_seq_len + 0], dtype=torch.int32, device=_DEVICE
    )

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        req_to_token=req_to_token,
        num_slots=64,
        extras=_empty_extras(),
        kernel_kind=kernel_kind,
    )

    assert int(log_real.write_index[0].item()) == 0
    assert int(log_ref.write_index[0].item()) == 0
