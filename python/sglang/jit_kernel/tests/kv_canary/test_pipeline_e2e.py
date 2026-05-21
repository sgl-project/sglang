from __future__ import annotations

from typing import Any, Callable, Optional

import pytest
import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.plan_ref import canary_plan_step_torch_reference
from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvSource,
    VerifyPlan,
    canary_verify_step,
)
from sglang.jit_kernel.kv_canary.verify_ref import canary_verify_step_torch_reference
from sglang.jit_kernel.kv_canary.write import WritePlan, canary_write_step
from sglang.jit_kernel.kv_canary.write_ref import canary_write_step_torch_reference
from sglang.jit_kernel.tests.kv_canary._canary_helpers import (
    FakeViolationLog,
    assert_canary_buf_equal,
    assert_canary_state_equal,
    make_canary_buf,
    make_real_kv_sources,
    stamp_clean_chain,
    write_slot_fields,
)
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    _empty_extras,
    clone_real_kv_sources,
    make_req_to_token,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _run_pipeline(
    *,
    plan_fn: Callable[..., None],
    write_fn: Callable[..., None],
    verify_fn: Callable[..., None],
    synchronize: bool,
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
    enable_write_verify_inputs: bool,
    expected_input_tokens: torch.Tensor,
    expected_input_positions: torch.Tensor,
    real_kv_sources: tuple[RealKvSource, ...],
    real_kv_hash_mode: consts.RealKvHashMode,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan]:
    _ = extras
    plan_v = VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE)
    plan_w = WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE)

    plan_fn(
        verify_plan_out=plan_v,
        write_plan_out=plan_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        verify_capacity=verify_capacity,
    )
    write_fn(
        canary_buf=canary_buf,
        plan=plan_w,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
        kernel_kind=kernel_kind,
        enable_write_verify_inputs=enable_write_verify_inputs,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
        violation_ring=log.ring,
        violation_write_index=log.write_index,
        slot_run_counter=log.slot_run_counter,
        kernel_run_counter=log.kernel_run_counter,
        real_kv_sources=real_kv_sources,
        real_kv_hash_mode=real_kv_hash_mode,
    )
    verify_fn(
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

    if synchronize:
        torch.cuda.synchronize()
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
    enable_write_verify_inputs: bool = False,
    expected_input_tokens: Optional[torch.Tensor] = None,
    expected_input_positions: Optional[torch.Tensor] = None,
    real_kv_sources_real: tuple[RealKvSource, ...] = (),
    real_kv_sources_ref: tuple[RealKvSource, ...] = (),
    real_kv_hash_mode: consts.RealKvHashMode = consts.RealKvHashMode.OFF,
    ring_capacity: int = 64,
    verify_capacity: int = 256,
    write_req_capacity: int = 16,
    assert_ring_equal: bool = True,
    initial_canary_buf: Optional[torch.Tensor] = None,
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
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
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
    )

    plan_v_real, plan_w_real = _run_pipeline(
        plan_fn=canary_plan_step,
        write_fn=canary_write_step,
        verify_fn=canary_verify_step,
        synchronize=True,
        canary_buf=buf_real,
        log=log_real,
        real_kv_sources=real_kv_sources_real,
        **shared,
    )
    plan_v_ref, plan_w_ref = _run_pipeline(
        plan_fn=canary_plan_step_torch_reference,
        write_fn=canary_write_step_torch_reference,
        verify_fn=canary_verify_step_torch_reference,
        synchronize=False,
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


def test_pipeline_basic_5_step_single_req() -> None:
    """Single req, prefix_len=0, extend_seq_len=5: basic plan→write→verify byte-equal."""
    max_seq_len = 16
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
    )
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([5], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            1 * max_seq_len + 0,
            1 * max_seq_len + 1,
            1 * max_seq_len + 2,
            1 * max_seq_len + 3,
            1 * max_seq_len + 4,
        ],
        dtype=torch.int64,
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
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=8, max_seq_len=max_seq_len, device=_DEVICE
    )
    fb_req_pool_indices = torch.tensor([1, 2, 0], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0, 5, 0], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([4, 1, 0], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.tensor([11, 12, 13, 14, 21], dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            1 * max_seq_len + 0,
            1 * max_seq_len + 1,
            1 * max_seq_len + 2,
            1 * max_seq_len + 3,
            2 * max_seq_len + 5,
        ],
        dtype=torch.int64,
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
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=max_reqs, max_seq_len=max_seq_len, device=_DEVICE
    )

    swa_window_size = 4
    full_pool_size = max_reqs * max_seq_len

    full_to_swa_index_mapping = torch.arange(
        full_pool_size + 1, dtype=torch.int64, device=_DEVICE
    )

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([6], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([2], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.tensor([100, 101], dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.tensor([6, 7], dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            full_to_swa_index_mapping[1 * max_seq_len + 6].item(),
            full_to_swa_index_mapping[1 * max_seq_len + 7].item(),
        ],
        dtype=torch.int64,
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
    prefix_len = 4
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
    )

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([prefix_len], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.zeros(1, dtype=torch.int64, device=_DEVICE)

    initial_buf = make_canary_buf(num_slots=64, device=_DEVICE)
    initial_ref = initial_buf.clone()
    prefix_slots = [1 * max_seq_len + pos for pos in range(prefix_len)]
    stamp_clean_chain(
        cuda_buf=initial_buf,
        ref_buf=initial_ref,
        slot_indices=prefix_slots,
        tokens=[100 + pos for pos in range(prefix_len)],
        positions=list(range(prefix_len)),
    )

    buf_real, buf_ref, log_real, log_ref, plan_v_real, plan_w_real, _, _ = (
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
            initial_canary_buf=initial_buf,
        )
    )

    assert int(plan_v_real.verify_num_valid[0].item()) == prefix_len
    assert int(plan_w_real.write_num_valid_reqs[0].item()) == 0
    assert torch.equal(buf_real, initial_buf)
    assert torch.equal(buf_ref, initial_buf)
    assert int(log_real.write_index[0].item()) == 0
    assert int(log_ref.write_index[0].item()) == 0
    assert int(log_real.slot_run_counter[0].item()) == prefix_len
    assert int(log_ref.slot_run_counter[0].item()) == prefix_len


@pytest.mark.parametrize(
    "real_kv_hash_mode",
    [
        consts.RealKvHashMode.OFF,
        consts.RealKvHashMode.PARTIAL,
        consts.RealKvHashMode.ALL,
    ],
)
def test_pipeline_real_kv_mode(real_kv_hash_mode: consts.RealKvHashMode) -> None:
    """real_kv_hash_mode OFF/PARTIAL/ALL: real and ref use cloned sources to prevent ALL-mode hash aliasing."""
    max_seq_len = 16
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
    )
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([3], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.tensor([5, 6, 7], dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [1 * max_seq_len + 0, 1 * max_seq_len + 1, 1 * max_seq_len + 2],
        dtype=torch.int64,
        device=_DEVICE,
    )

    sources_real = make_real_kv_sources(count=2, num_slots=64, device=_DEVICE)
    sources_ref = clone_real_kv_sources(sources_real)

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
    """enable_write_verify_inputs=ON, expected==actual: zero violations, buf byte-equal."""
    max_seq_len = 16
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
    )
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([4], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [
            1 * max_seq_len + 0,
            1 * max_seq_len + 1,
            1 * max_seq_len + 2,
            1 * max_seq_len + 3,
        ],
        dtype=torch.int64,
        device=_DEVICE,
    )
    expected_input_tokens = fb_input_ids.clone()
    expected_input_positions = fb_positions.clone()

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
        enable_write_verify_inputs=True,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
    )

    assert int(log_real.write_index[0].item()) == 0
    assert int(log_ref.write_index[0].item()) == 0


def test_pipeline_pseudo_mode_on_token_mismatch_then_verify_clean() -> None:
    """enable_write_verify_inputs=ON, expected tokens all wrong: write records N violations, verify does not cascade CHAIN_HASH."""
    max_seq_len = 16
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
    )
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([3], dtype=torch.int64, device=_DEVICE)
    n_tokens = 3
    fb_input_ids = torch.tensor([10, 20, 30], dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.tensor([0, 1, 2], dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [1 * max_seq_len + 0, 1 * max_seq_len + 1, 1 * max_seq_len + 2],
        dtype=torch.int64,
        device=_DEVICE,
    )
    expected_input_tokens = torch.tensor(
        [99, 99, 99], dtype=torch.int64, device=_DEVICE
    )
    expected_input_positions = fb_positions.clone()

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
        enable_write_verify_inputs=True,
        expected_input_tokens=expected_input_tokens,
        expected_input_positions=expected_input_positions,
        ring_capacity=64,
    )

    write_violations = int(log_real.write_index[0].item())
    assert (
        write_violations == n_tokens
    ), f"expected {n_tokens} write violations, got {write_violations}"


def test_pipeline_empty_batch() -> None:
    """bs=1 with req_pool_idx=0 (padding): write and verify are no-op, kernel_run_counter == 2 (write+verify)."""
    max_seq_len = 16
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
    )
    fb_req_pool_indices = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.zeros(1, dtype=torch.int64, device=_DEVICE)

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
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=max_reqs, max_seq_len=max_seq_len, device=_DEVICE
    )
    swa_window_size = 4

    full_to_swa_index_mapping = torch.arange(
        full_pool_size + 1, dtype=torch.int64, device=_DEVICE
    )
    full_to_swa_index_mapping[1 * max_seq_len + 6] = -1
    full_to_swa_index_mapping[1 * max_seq_len + 7] = -1

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([6], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([4], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.tensor([100, 101, 102, 103], dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.tensor([6, 7, 8, 9], dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.tensor(
        [-1, -1, 1 * max_seq_len + 8, 1 * max_seq_len + 9],
        dtype=torch.int64,
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
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=max_reqs, max_seq_len=max_seq_len, device=_DEVICE
    )
    n_slots = 8

    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([n_slots], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.zeros(1, dtype=torch.int64, device=_DEVICE)

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

    canary_plan_step(
        verify_plan_out=plan_v_real,
        write_plan_out=plan_w_real,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        verify_capacity=int(plan_v_real.verify_slot_indices.shape[0]),
    )
    canary_plan_step_torch_reference(
        verify_plan_out=plan_v_ref,
        write_plan_out=plan_w_ref,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        verify_capacity=int(plan_v_ref.verify_slot_indices.shape[0]),
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
        real_kv_hash_mode=consts.RealKvHashMode.OFF,
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
        real_kv_hash_mode=consts.RealKvHashMode.OFF,
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
    req_to_token = make_req_to_token(
        kind="linear", max_reqs=4, max_seq_len=max_seq_len, device=_DEVICE
    )
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_prefix_lens = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
    fb_input_ids = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_positions = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    fb_out_cache_loc = torch.zeros(1, dtype=torch.int64, device=_DEVICE)

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
