from __future__ import annotations

from typing import Any, Optional

import pytest
import torch

from sglang.kernels.jit.tests.kv_canary._canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_equal,
    make_canary_buf,
    make_real_kv_sources,
    stamp_clean_chain,
    write_slot_fields,
)
from sglang.kernels.jit.tests.kv_canary._fixtures import (
    clone_real_kv_sources,
    empty_extras,
    make_req_to_token,
)
from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.plan import launch_canary_plan_kernels
from sglang.kernels.ops.kv_canary.plan_ref import (
    launch_canary_plan_kernels_torch_reference,
)
from sglang.kernels.ops.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
    launch_canary_verify_kernel,
)
from sglang.kernels.ops.kv_canary.verify_ref import (
    launch_canary_verify_kernel_torch_reference,
)
from sglang.kernels.ops.kv_canary.write import WritePlan, launch_canary_write_kernel
from sglang.kernels.ops.kv_canary.write_ref import (
    launch_canary_write_kernel_torch_reference,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")


def _run_pipeline(
    *,
    real: bool,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_to_token: torch.Tensor,
    canary_buf: torch.Tensor,
    log: FakeViolationLog,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    kernel_kind: CanaryLaunchTag,
    enable_write_verify_inputs: bool,
    expected_input_tokens: torch.Tensor,
    expected_input_positions: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: consts.RealKvHashMode,
    verify_capacity: int,
    write_req_capacity: int,
    req_to_verify_expected_tokens: Optional[torch.Tensor] = None,
    req_to_verify_expected_tokens_valid_lens: Optional[torch.Tensor] = None,
    kv_token_id_vs_position_offset: int = 0,
    check_verify_expected_token: bool = True,
) -> tuple[VerifyPlan, WritePlan]:
    _ = extras
    plan_v = VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE)
    plan_w = WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE)

    # Existing pipeline tests that supply a pool but no per-req lens want "bound by full
    # row width" semantics. Synthesise that bound here so callers don't have to.
    if (
        req_to_verify_expected_tokens is not None
        and req_to_verify_expected_tokens_valid_lens is None
    ):
        req_to_verify_expected_tokens_valid_lens = torch.full(
            (int(req_pool_indices.shape[0]),),
            int(req_to_verify_expected_tokens.shape[1]),
            dtype=torch.int64,
            device=req_pool_indices.device,
        )

    plan_fn = (
        launch_canary_plan_kernels
        if real
        else launch_canary_plan_kernels_torch_reference
    )
    plan_fn(
        verify_plan_out=plan_v,
        write_plan_out=plan_w,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_capacity=verify_capacity,
        req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        req_to_verify_expected_tokens_valid_lens=req_to_verify_expected_tokens_valid_lens,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )

    if real:
        context = VerifyOrWriteContext(
            canary_buf=canary_buf,
            kernel_kind=kernel_kind,
            violation_ring=log.ring,
            violation_write_index=log.write_index,
            slot_run_counter=log.slot_run_counter,
            kernel_run_counter=log.kernel_run_counter,
            enable_chain_position_assert=log.enable_chain_position_assert,
            real_kv_sources=real_kv_sources,
            real_kv_hash_mode=real_kv_hash_mode,
        )
        launch_canary_write_kernel(
            context=context,
            plan=plan_w,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            enable_write_input_assert=enable_write_verify_inputs,
            expected_input_tokens=expected_input_tokens,
            expected_input_positions=expected_input_positions,
        )
        launch_canary_verify_kernel(
            context=context,
            plan=plan_v,
            check_verify_expected_token=check_verify_expected_token,
        )
        torch.cuda.synchronize()
    else:
        launch_canary_write_kernel_torch_reference(
            context=VerifyOrWriteContext(
                canary_buf=canary_buf,
                kernel_kind=kernel_kind,
                violation_ring=log.ring,
                violation_write_index=log.write_index,
                slot_run_counter=log.slot_run_counter,
                kernel_run_counter=log.kernel_run_counter,
                enable_chain_position_assert=log.enable_chain_position_assert,
                real_kv_sources=real_kv_sources,
                real_kv_hash_mode=real_kv_hash_mode,
            ),
            plan=plan_w,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            enable_write_input_assert=enable_write_verify_inputs,
            expected_input_tokens=expected_input_tokens,
            expected_input_positions=expected_input_positions,
        )
        launch_canary_verify_kernel_torch_reference(
            context=VerifyOrWriteContext(
                canary_buf=canary_buf,
                kernel_kind=kernel_kind,
                violation_ring=log.ring,
                violation_write_index=log.write_index,
                slot_run_counter=log.slot_run_counter,
                kernel_run_counter=log.kernel_run_counter,
                enable_chain_position_assert=log.enable_chain_position_assert,
                real_kv_sources=real_kv_sources,
                real_kv_hash_mode=real_kv_hash_mode,
            ),
            plan=plan_v,
            check_verify_expected_token=check_verify_expected_token,
        )

    return plan_v, plan_w


def _run_both_and_assert_pipeline_equal(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_to_token: torch.Tensor,
    num_slots: int,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int = 0,
    full_to_swa_index_mapping: Optional[torch.Tensor] = None,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
    enable_write_verify_inputs: bool = False,
    expected_input_tokens: Optional[torch.Tensor] = None,
    expected_input_positions: Optional[torch.Tensor] = None,
    real_kv_sources_real: tuple[RealKvSource, ...] = (),
    real_kv_sources_ref: tuple[RealKvSource, ...] = (),
    real_kv_hash_mode: consts.RealKvHashMode = consts.RealKvHashMode.NONE,
    ring_capacity: int = 64,
    verify_capacity: int = 256,
    write_req_capacity: int = 16,
    assert_ring_equal: bool = True,
    initial_canary_buf: Optional[torch.Tensor] = None,
    req_to_verify_expected_tokens: Optional[torch.Tensor] = None,
    kv_token_id_vs_position_offset: int = 0,
    check_verify_expected_token: bool = True,
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
    # The kernel rejects non-None expected_* tensors when enable_write_verify_inputs=False
    # (sanity check to catch caller bugs), so only synthesise zero placeholders in the
    # branch that will actually assert against them.
    if enable_write_verify_inputs:
        total_tokens = int(input_ids.shape[0])
        if expected_input_tokens is None:
            expected_input_tokens = torch.zeros(
                total_tokens, dtype=torch.int64, device=_DEVICE
            )
        if expected_input_positions is None:
            expected_input_positions = torch.zeros(
                total_tokens, dtype=torch.int64, device=_DEVICE
            )

    if initial_canary_buf is None:
        buf_real = make_canary_buf(num_slots=num_slots, device=_DEVICE)
    else:
        buf_real = initial_canary_buf.clone()
    buf_ref = buf_real.clone()
    log_real = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)
    log_ref = FakeViolationLog.allocate(capacity=ring_capacity, device=_DEVICE)

    shared: dict[str, Any] = dict(
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        kernel_kind=kernel_kind,
        enable_write_verify_inputs=enable_write_verify_inputs,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
        real_kv_hash_mode=real_kv_hash_mode,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
        req_to_verify_expected_tokens=req_to_verify_expected_tokens,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
        check_verify_expected_token=check_verify_expected_token,
    )

    plan_v_real, plan_w_real = _run_pipeline(
        real=True,
        canary_buf=buf_real,
        log=log_real,
        real_kv_sources=real_kv_sources_real,
        **shared,
    )
    plan_v_ref, plan_w_ref = _run_pipeline(
        real=False,
        canary_buf=buf_ref,
        log=log_ref,
        real_kv_sources=real_kv_sources_ref,
        **shared,
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


def _t(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=_DEVICE)


def _linear_r2t(*, max_reqs: int = 4, max_seq_len: int = 16) -> torch.Tensor:
    return make_req_to_token(
        kind="linear", max_reqs=max_reqs, max_seq_len=max_seq_len, device=_DEVICE
    )


def _zero_no_write_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(input_ids, positions, out_cache_loc)`` zero placeholders for extend_seq_lens=0 tests."""
    zeros = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    return zeros.clone(), zeros.clone(), zeros.clone()


def _contiguous_out_cache_loc(
    *, req_pool_idx: int, start: int, count: int, max_seq_len: int = 16
) -> torch.Tensor:
    return _t([req_pool_idx * max_seq_len + start + i for i in range(count)])


def _stamp_linear_prefix(
    *,
    initial_buf: torch.Tensor,
    initial_ref: torch.Tensor,
    req_pool_idx: int,
    prefix_len: int,
    tokens: list[int],
    max_seq_len: int = 16,
) -> None:
    """Stamp clean chain for slots ``[rp*max_seq_len + 0 .. + prefix_len)`` at positions ``0..prefix_len``."""
    stamp_clean_chain(
        cuda_buf=initial_buf,
        ref_buf=initial_ref,
        slot_indices=[req_pool_idx * max_seq_len + pos for pos in range(prefix_len)],
        tokens=tokens,
        positions=list(range(prefix_len)),
    )


def test_pipeline_basic_5_step_single_req() -> None:
    """Single req, prefix_len=0, extend_seq_len=5: basic plan→write→verify byte-equal."""
    _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([0]),
        extend_seq_lens=_t([5]),
        input_ids=_t([10, 20, 30, 40, 50]),
        positions=_t([0, 1, 2, 3, 4]),
        out_cache_loc=_contiguous_out_cache_loc(req_pool_idx=1, start=0, count=5),
        req_to_token=_linear_r2t(),
        num_slots=64,
        extras=empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )


def test_pipeline_multi_req_mixed_extend_decode() -> None:
    """bs=3: pure extend req, decode req (prefix+1 extend), and padding sentinel row."""
    max_seq_len = 16
    _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1, 2, 0]),
        prefix_lens=_t([0, 5, 0]),
        extend_seq_lens=_t([4, 1, 0]),
        input_ids=_t([11, 12, 13, 14, 21]),
        positions=_t([0, 1, 2, 3, 5]),
        out_cache_loc=_t(
            [
                1 * max_seq_len + 0,
                1 * max_seq_len + 1,
                1 * max_seq_len + 2,
                1 * max_seq_len + 3,
                2 * max_seq_len + 5,
            ]
        ),
        req_to_token=_linear_r2t(max_reqs=8, max_seq_len=max_seq_len),
        num_slots=128,
        extras=empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        write_req_capacity=4,
    )


def test_pipeline_swa_window() -> None:
    """SWA window=4, prefix_len=6: verify covers window [2,6), write covers extend tokens."""
    max_seq_len = 16
    max_reqs = 4
    full_to_swa_index_mapping = torch.arange(
        max_reqs * max_seq_len + 1, dtype=torch.int64, device=_DEVICE
    )

    _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([6]),
        extend_seq_lens=_t([2]),
        input_ids=_t([100, 101]),
        positions=_t([6, 7]),
        out_cache_loc=_t(
            [
                full_to_swa_index_mapping[1 * max_seq_len + 6].item(),
                full_to_swa_index_mapping[1 * max_seq_len + 7].item(),
            ]
        ),
        req_to_token=_linear_r2t(max_reqs=max_reqs, max_seq_len=max_seq_len),
        num_slots=64,
        extras=empty_extras(),
        swa_window_size=4,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )


def test_pipeline_sweep_no_write() -> None:
    """All extend_seq_lens=0: write_step is no-op, verify sweeps prefix, buf unchanged."""
    prefix_len = 4
    input_ids, positions, out_cache_loc = _zero_no_write_inputs()

    initial_buf = make_canary_buf(num_slots=64, device=_DEVICE)
    initial_ref = initial_buf.clone()
    _stamp_linear_prefix(
        initial_buf=initial_buf,
        initial_ref=initial_ref,
        req_pool_idx=1,
        prefix_len=prefix_len,
        tokens=[100 + pos for pos in range(prefix_len)],
    )

    buf_real, buf_ref, log_real, log_ref, plan_v_real, plan_w_real, _, _ = (
        _run_both_and_assert_pipeline_equal(
            req_pool_indices=_t([1]),
            prefix_lens=_t([prefix_len]),
            extend_seq_lens=_t([0]),
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            req_to_token=_linear_r2t(),
            num_slots=64,
            extras=empty_extras(),
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            initial_canary_buf=initial_buf,
        )
    )

    assert int(plan_v_real.verify_num_valid[0].item()) == prefix_len
    assert int(plan_w_real.write_num_valid_reqs[0].item()) == 1
    assert int(plan_w_real.write_offsets[1].item()) == 0
    assert torch.equal(buf_real, initial_buf)
    assert torch.equal(buf_ref, initial_buf)
    assert int(log_real.write_index[0].item()) == 0
    assert int(log_ref.write_index[0].item()) == 0
    assert int(log_real.slot_run_counter[0].item()) == prefix_len
    assert int(log_ref.slot_run_counter[0].item()) == prefix_len


@pytest.mark.parametrize(
    "real_kv_hash_mode",
    [
        consts.RealKvHashMode.NONE,
        consts.RealKvHashMode.PARTIAL,
        consts.RealKvHashMode.ALL,
    ],
)
def test_pipeline_real_kv_mode(real_kv_hash_mode: consts.RealKvHashMode) -> None:
    """real_kv_hash_mode OFF/PARTIAL/ALL: real and ref use cloned sources to prevent ALL-mode hash aliasing."""
    sources_real = make_real_kv_sources(count=2, num_slots=64, device=_DEVICE)
    sources_ref = clone_real_kv_sources(sources_real)

    _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([0]),
        extend_seq_lens=_t([3]),
        input_ids=_t([5, 6, 7]),
        positions=_t([0, 1, 2]),
        out_cache_loc=_contiguous_out_cache_loc(req_pool_idx=1, start=0, count=3),
        req_to_token=_linear_r2t(),
        num_slots=64,
        extras=empty_extras(),
        real_kv_sources_real=sources_real,
        real_kv_sources_ref=sources_ref,
        real_kv_hash_mode=real_kv_hash_mode,
    )


def test_pipeline_pseudo_mode_on_match() -> None:
    """enable_write_verify_inputs=ON, expected==actual: zero violations, buf byte-equal."""
    input_ids = _t([1, 2, 3, 4])
    positions = _t([0, 1, 2, 3])

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([0]),
        extend_seq_lens=_t([4]),
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=_contiguous_out_cache_loc(req_pool_idx=1, start=0, count=4),
        req_to_token=_linear_r2t(),
        num_slots=64,
        extras=empty_extras(),
        enable_write_verify_inputs=True,
        expected_input_tokens=input_ids.clone(),
        expected_input_positions=positions.clone(),
    )

    assert int(log_real.write_index[0].item()) == 0
    assert int(log_ref.write_index[0].item()) == 0


def test_pipeline_pseudo_mode_on_token_mismatch_then_verify_clean() -> None:
    """enable_write_verify_inputs=ON, expected tokens all wrong: write records N violations."""
    n_tokens = 3
    positions = _t([0, 1, 2])

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([0]),
        extend_seq_lens=_t([3]),
        input_ids=_t([10, 20, 30]),
        positions=positions,
        out_cache_loc=_contiguous_out_cache_loc(req_pool_idx=1, start=0, count=3),
        req_to_token=_linear_r2t(),
        num_slots=64,
        extras=empty_extras(),
        enable_write_verify_inputs=True,
        expected_input_tokens=_t([99, 99, 99]),
        expected_input_positions=positions.clone(),
        ring_capacity=64,
    )

    write_violations = int(log_real.write_index[0].item())
    assert (
        write_violations == n_tokens
    ), f"expected {n_tokens} write violations, got {write_violations}"


def test_pipeline_empty_batch() -> None:
    """bs=1 with req_pool_idx=0 (padding): write and verify are no-op, kernel_run_counter == 2 (write+verify)."""
    input_ids, positions, out_cache_loc = _zero_no_write_inputs()

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([0]),
        prefix_lens=_t([0]),
        extend_seq_lens=_t([0]),
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        req_to_token=_linear_r2t(),
        num_slots=64,
        extras=empty_extras(),
    )

    assert int(log_real.kernel_run_counter[0].item()) == 2
    assert int(log_ref.kernel_run_counter[0].item()) == 2
    assert int(log_real.write_index[0].item()) == 0


def test_pipeline_negative_slot_swa_out_of_window() -> None:
    """SWA: some out_cache_loc entries map to -1 (out-of-window); write_step skips them, buf unchanged."""
    max_seq_len = 16
    max_reqs = 4

    full_to_swa_index_mapping = torch.arange(
        max_reqs * max_seq_len + 1, dtype=torch.int64, device=_DEVICE
    )
    full_to_swa_index_mapping[1 * max_seq_len + 6] = -1
    full_to_swa_index_mapping[1 * max_seq_len + 7] = -1

    _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([6]),
        extend_seq_lens=_t([4]),
        input_ids=_t([100, 101, 102, 103]),
        positions=_t([6, 7, 8, 9]),
        out_cache_loc=_t([-1, -1, 1 * max_seq_len + 8, 1 * max_seq_len + 9]),
        req_to_token=_linear_r2t(max_reqs=max_reqs, max_seq_len=max_seq_len),
        num_slots=128,
        extras=empty_extras(),
        swa_window_size=4,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )


def test_pipeline_ring_overflow_via_real_plan() -> None:
    """Verify detects >capacity violations when prev_hash is pre-corrupted; write_index byte-equal, ring relaxed."""
    max_seq_len = 16
    max_reqs = 4
    req_to_token = _linear_r2t(max_reqs=max_reqs, max_seq_len=max_seq_len)
    n_slots = 8

    req_pool_indices = _t([1])
    prefix_lens = _t([n_slots])
    extend_seq_lens = _t([0])
    input_ids, positions, out_cache_loc = _zero_no_write_inputs()

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
    plan_v_real = VerifyPlan.allocate(verify_capacity=256, device=_DEVICE)
    plan_w_real = WritePlan.allocate(write_req_capacity=4, device=_DEVICE)
    plan_v_ref = VerifyPlan.allocate(verify_capacity=256, device=_DEVICE)
    plan_w_ref = WritePlan.allocate(write_req_capacity=4, device=_DEVICE)

    launch_canary_plan_kernels(
        verify_plan_out=plan_v_real,
        write_plan_out=plan_w_real,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        verify_capacity=int(plan_v_real.verify_slot_indices.shape[0]),
        req_to_verify_expected_tokens=None,
        req_to_verify_expected_tokens_valid_lens=None,
        kv_token_id_vs_position_offset=0,
    )
    launch_canary_plan_kernels_torch_reference(
        verify_plan_out=plan_v_ref,
        write_plan_out=plan_w_ref,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        verify_capacity=int(plan_v_ref.verify_slot_indices.shape[0]),
        req_to_verify_expected_tokens=None,
        req_to_verify_expected_tokens_valid_lens=None,
        kv_token_id_vs_position_offset=0,
    )

    launch_canary_verify_kernel(
        context=VerifyOrWriteContext(
            canary_buf=buf_real,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=log_real.ring,
            violation_write_index=log_real.write_index,
            slot_run_counter=log_real.slot_run_counter,
            kernel_run_counter=log_real.kernel_run_counter,
            enable_chain_position_assert=log_real.enable_chain_position_assert,
            real_kv_sources=(),
            real_kv_hash_mode=consts.RealKvHashMode.NONE,
        ),
        plan=plan_v_real,
        check_verify_expected_token=True,
    )
    torch.cuda.synchronize()

    launch_canary_verify_kernel_torch_reference(
        context=VerifyOrWriteContext(
            canary_buf=buf_ref,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=log_ref.ring,
            violation_write_index=log_ref.write_index,
            slot_run_counter=log_ref.slot_run_counter,
            kernel_run_counter=log_ref.kernel_run_counter,
            enable_chain_position_assert=log_ref.enable_chain_position_assert,
            real_kv_sources=(),
            real_kv_hash_mode=consts.RealKvHashMode.NONE,
        ),
        plan=plan_v_ref,
        check_verify_expected_token=True,
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
    input_ids, positions, out_cache_loc = _zero_no_write_inputs()

    initial_buf = make_canary_buf(num_slots=64, device=_DEVICE)
    write_slot_fields(
        canary_buf=initial_buf,
        slot_idx=1 * max_seq_len,
        token=7,
        position=99,
        prev_hash=0,
        real_kv_hash=0,
    )

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([1]),
        extend_seq_lens=_t([0]),
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        req_to_token=_linear_r2t(max_seq_len=max_seq_len),
        num_slots=64,
        extras=empty_extras(),
        kernel_kind=kernel_kind,
        initial_canary_buf=initial_buf,
    )

    assert int(log_real.write_index[0].item()) == 1
    assert int(log_ref.write_index[0].item()) == 1
    assert int(log_real.ring[0, consts.VIOLATION_FIELD_KERNEL_KIND].item()) == int(
        kernel_kind
    )
    assert int(log_ref.ring[0, consts.VIOLATION_FIELD_KERNEL_KIND].item()) == int(
        kernel_kind
    )


def test_pipeline_token_mismatch_detected_via_pool() -> None:
    """plan-pool gather + verify-token check: stamped wrong token id raises VERIFY_TOKEN_MISMATCH."""
    max_seq_len = 16
    max_reqs = 4
    prefix_len = 4
    input_ids, positions, out_cache_loc = _zero_no_write_inputs()

    expected_tokens = [1000 + pos for pos in range(prefix_len)]
    pool = torch.full((max_reqs, max_seq_len), -999, dtype=torch.int32, device=_DEVICE)
    for pos, token in enumerate(expected_tokens):
        pool[1, pos] = token

    stored_tokens = [token + 1 for token in expected_tokens]

    initial_buf = make_canary_buf(num_slots=64, device=_DEVICE)
    initial_ref = initial_buf.clone()
    _stamp_linear_prefix(
        initial_buf=initial_buf,
        initial_ref=initial_ref,
        req_pool_idx=1,
        prefix_len=prefix_len,
        tokens=stored_tokens,
        max_seq_len=max_seq_len,
    )

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([prefix_len]),
        extend_seq_lens=_t([0]),
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        req_to_token=_linear_r2t(max_reqs=max_reqs, max_seq_len=max_seq_len),
        num_slots=64,
        extras=empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        initial_canary_buf=initial_buf,
        req_to_verify_expected_tokens=pool,
        kv_token_id_vs_position_offset=0,
        check_verify_expected_token=True,
    )

    assert int(log_real.write_index[0].item()) == prefix_len
    assert int(log_ref.write_index[0].item()) == prefix_len
    # Ring rows may land in any order; collect stored/expected pairs and compare as sets.
    observed_pairs: set[tuple[int, int]] = set()
    for row_idx in range(prefix_len):
        fail_bits = int(
            log_real.ring[row_idx, consts.VIOLATION_FIELD_FAIL_REASON_BITS].item()
        )
        assert fail_bits & int(
            consts.FailReason.VERIFY_TOKEN_MISMATCH
        ), f"row {row_idx}: VERIFY_TOKEN_MISMATCH bit missing in {fail_bits:#b}"
        stored = int(log_real.ring[row_idx, consts.VIOLATION_FIELD_STORED_TOKEN].item())
        expected = int(
            log_real.ring[row_idx, consts.VIOLATION_FIELD_EXPECTED_TOKEN].item()
        )
        observed_pairs.add((stored, expected))
    expected_pairs = {(stored_tokens[i], expected_tokens[i]) for i in range(prefix_len)}
    assert observed_pairs == expected_pairs


def test_pipeline_eagle_offset_plus_1_byte_equal() -> None:
    """plan-pool + offset=+1 full pipeline: stamped tokens match pool[rp, pos+1], no violations CUDA vs ref byte-equal."""
    max_seq_len = 16
    max_reqs = 4
    prefix_len = 4
    input_ids, positions, out_cache_loc = _zero_no_write_inputs()

    stored_tokens = [2000 + pos for pos in range(prefix_len)]
    pool = torch.full((max_reqs, max_seq_len), -999, dtype=torch.int32, device=_DEVICE)
    for pos in range(prefix_len):
        # offset=+1 means kernel gathers from pool[rp, pos + 1], so place stored_tokens[pos] there.
        pool[1, pos + 1] = stored_tokens[pos]

    initial_buf = make_canary_buf(num_slots=64, device=_DEVICE)
    initial_ref = initial_buf.clone()
    _stamp_linear_prefix(
        initial_buf=initial_buf,
        initial_ref=initial_ref,
        req_pool_idx=1,
        prefix_len=prefix_len,
        tokens=stored_tokens,
        max_seq_len=max_seq_len,
    )

    _, _, log_real, log_ref, _, _, _, _ = _run_both_and_assert_pipeline_equal(
        req_pool_indices=_t([1]),
        prefix_lens=_t([prefix_len]),
        extend_seq_lens=_t([0]),
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        req_to_token=_linear_r2t(max_reqs=max_reqs, max_seq_len=max_seq_len),
        num_slots=64,
        extras=empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        initial_canary_buf=initial_buf,
        req_to_verify_expected_tokens=pool,
        kv_token_id_vs_position_offset=1,
        check_verify_expected_token=True,
    )

    assert int(log_real.write_index[0].item()) == 0
    assert int(log_ref.write_index[0].item()) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
