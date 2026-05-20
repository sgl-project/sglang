"""Differential test: Triton canary_plan_step vs the torch reference, byte-equal."""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.plan_ref import (
    canary_plan_step_torch_reference,
)
from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _allocate_plan_pair(
    *,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan, VerifyPlan, WritePlan]:
    """Allocate (triton_verify, triton_write, ref_verify, ref_write) plan tensors."""
    return (
        VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE),
        WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE),
        VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE),
        WritePlan.allocate(write_req_capacity=write_req_capacity, device=_DEVICE),
    )


def _build_req_to_token(*, max_reqs: int, max_seq_len: int) -> torch.Tensor:
    """Construct a deterministic [max_reqs, max_seq_len] req_to_token table.

    Slot index = rp * max_seq_len + pos so every (rp, pos) maps to a distinct slot, which lets per-entry
    assertions reason about which req contributed which slot.
    """
    rp_axis = torch.arange(max_reqs, device=_DEVICE, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    return (rp_axis * max_seq_len + pos_axis).contiguous()


def _empty_extras() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four zero-filled length-1 int32 tensors representing an "extras absent" payload."""
    return (
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        torch.zeros(1, dtype=torch.int32, device=_DEVICE),
    )


def _make_extras(
    *,
    slot_indices: list[int],
    positions: list[int],
    prev_slot_indices: list[int],
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(slot_indices)
    slots = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    pos = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    prevs = torch.zeros(capacity, dtype=torch.int32, device=_DEVICE)
    if n > 0:
        slots[:n] = torch.tensor(slot_indices, dtype=torch.int32, device=_DEVICE)
        pos[:n] = torch.tensor(positions, dtype=torch.int32, device=_DEVICE)
        prevs[:n] = torch.tensor(prev_slot_indices, dtype=torch.int32, device=_DEVICE)
    num_valid = torch.tensor([n], dtype=torch.int32, device=_DEVICE)
    return slots, pos, prevs, num_valid


def _run_both(
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
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> None:
    extra_slots, extra_positions, extra_prev_slots, extra_num_valid = extras
    canary_plan_step(
        verify_plan_out=triton_verify,
        write_plan_out=triton_write,
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
    canary_plan_step_torch_reference(
        verify_plan_out=ref_verify,
        write_plan_out=ref_write,
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
    torch.cuda.synchronize()


def _run_both_and_assert_byte_equal(
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
    swa_window_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    active_verify_entries: Optional[int] = None,
    active_write_reqs: Optional[int] = None,
) -> None:
    _run_both(
        triton_verify=triton_verify,
        triton_write=triton_write,
        ref_verify=ref_verify,
        ref_write=ref_write,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    _assert_plans_byte_equal(
        triton_verify=triton_verify,
        triton_write=triton_write,
        ref_verify=ref_verify,
        ref_write=ref_write,
        active_verify_entries=active_verify_entries,
        active_write_reqs=active_write_reqs,
    )


def _assert_plans_byte_equal(
    *,
    triton_verify: VerifyPlan,
    triton_write: WritePlan,
    ref_verify: VerifyPlan,
    ref_write: WritePlan,
    active_verify_entries: Optional[int] = None,
    active_write_reqs: Optional[int] = None,
) -> None:
    """Byte-equal check on (Triton vs ref) plan outputs.

    Optional ``active_verify_entries`` / ``active_write_reqs`` truncate the comparison to the meaningful
    prefix; tail entries past the active count are kernel-undefined and need not match byte-equal.
    """
    n_verify = (
        active_verify_entries
        if active_verify_entries is not None
        else int(triton_verify.verify_num_valid[0].item())
    )
    n_verify_ref = int(ref_verify.verify_num_valid[0].item())
    assert (
        n_verify == n_verify_ref
    ), f"verify_num_valid diverged: triton={n_verify} ref={n_verify_ref}"
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

    n_write = (
        active_write_reqs
        if active_write_reqs is not None
        else int(triton_write.write_num_valid_reqs[0].item())
    )
    n_write_ref = int(ref_write.write_num_valid_reqs[0].item())
    assert (
        n_write == n_write_ref
    ), f"write_num_valid_reqs diverged: triton={n_write} ref={n_write_ref}"
    assert torch.equal(
        triton_write.write_offsets[: n_write + 1],
        ref_write.write_offsets[: n_write + 1],
    )
    if n_write > 0:
        assert torch.equal(
            triton_write.write_seed_slot_indices[:n_write],
            ref_write.write_seed_slot_indices[:n_write],
        )


def test_single_req_extend_basic() -> None:
    """bs=1, prefix=0, extend=5 → verify entries empty; write_offsets[0:2] = [0, 5]; seed = -1."""
    # Step 1: build a one-req batch with no prefix and 5 extend tokens.
    fb_req_pool_indices = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([5], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # Step 2: prefix=0 → no verify entries; seed = -1 because prefix==0.
    assert int(triton_v.verify_num_valid[0].item()) == 0
    assert int(triton_w.write_num_valid_reqs[0].item()) == 1
    assert int(triton_w.write_offsets[0].item()) == 0
    assert int(triton_w.write_offsets[1].item()) == 5
    assert int(triton_w.write_seed_slot_indices[0].item()) == -1


def test_single_req_decode() -> None:
    """extend=1, prefix=K → write_seed_slot = req_to_token[rp, K-1]; verify covers all K prefix tokens."""
    fb_req_pool_indices = torch.tensor([2], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([7], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # Step: verify covers positions [0..7); seed slot for req rp=2 is at position 6 = rp * max_seq_len + 6.
    assert int(triton_v.verify_num_valid[0].item()) == 7
    assert int(triton_w.write_seed_slot_indices[0].item()) == 2 * max_seq_len + 6


def test_multi_req_mixed_extend_decode() -> None:
    """bs=3 mixed extend/decode → write_offsets cumsum is byte-equal across Triton + ref."""
    fb_req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0, 4, 10], dtype=torch.int32, device=_DEVICE)
    # req0: prefill extend=8; req1: decode extend=1; req2: decode extend=1.
    fb_extend_seq_lens = torch.tensor([8, 1, 1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=8
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # Step: write_offsets exclusive cumsum on extend_seq_lens.
    expected_write_offsets = [0, 8, 9, 10]
    for i, value in enumerate(expected_write_offsets):
        assert int(triton_w.write_offsets[i].item()) == value
    # Verify count = 0 + 4 + 10 = 14.
    assert int(triton_v.verify_num_valid[0].item()) == 14


def test_prefix_zero_seed_is_minus_one() -> None:
    """prefix=0 → seed_slot_idx = -1 (no predecessor to anchor on)."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    assert int(triton_w.write_seed_slot_indices[0].item()) == -1


def test_padding_rows_contribute_zero() -> None:
    """``fb_req_pool_indices[r] == 0`` rows → no verify entry, no write entry, seed = -1."""
    # Step: bs=3 with row 1 marked as padding (rpi=0).
    fb_req_pool_indices = torch.tensor([1, 0, 2], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([5, 99, 3], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1, 99, 1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # verify count = 5 (req0) + 0 (padding) + 3 (req2) = 8.
    assert int(triton_v.verify_num_valid[0].item()) == 8
    # write_offsets cumsum: [0, 1, 1, 2] — padding row contributes 0.
    expected_write_offsets = [0, 1, 1, 2]
    for i, value in enumerate(expected_write_offsets):
        assert int(triton_w.write_offsets[i].item()) == value
    # Padding row's seed must be -1.
    assert int(triton_w.write_seed_slot_indices[1].item()) == -1


def test_swa_window_clip_prefix_less_than_window() -> None:
    """SWA: prefix=3 < window=128 → window_start=0, verify covers 3 entries (no clip)."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=512)
    # Identity LUT keeps slot indices unchanged after SWA translation.
    full_pool_size = 4 * 512
    lut = torch.arange(full_pool_size + 1, dtype=torch.int32, device=_DEVICE)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=256, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=128,
        full_to_swa_index_mapping=lut,
    )

    assert int(triton_v.verify_num_valid[0].item()) == 3


def test_swa_window_clip_prefix_gt_window() -> None:
    """SWA: prefix=200 > window=128 → window_start=72, verify covers 128 entries."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([200], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=512)
    full_pool_size = 4 * 512
    lut = torch.arange(full_pool_size + 1, dtype=torch.int32, device=_DEVICE)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=512, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=128,
        full_to_swa_index_mapping=lut,
    )

    assert int(triton_v.verify_num_valid[0].item()) == 128
    # First verify entry should be at position 72.
    assert int(triton_v.verify_positions[0].item()) == 72


def test_swa_lut_translates_verify_slots() -> None:
    """FULL slot → SWA slot translation is performed inside the plan kernel for verify_slot_indices."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    full_pool_size = 4 * max_seq_len
    # Build a LUT that maps FULL slot S → SWA slot (S + 100) for every S; chosen so we can distinguish a
    # translated value from a raw full slot.
    lut = (
        torch.arange(full_pool_size + 1, dtype=torch.int32, device=_DEVICE) + 100
    ).contiguous()

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=128,
        full_to_swa_index_mapping=lut,
    )

    # FULL slot for (rp=1, pos=0) = 1 * max_seq_len + 0 = 16; expected SWA slot = 16 + 100 = 116.
    assert int(triton_v.verify_slot_indices[0].item()) == 1 * max_seq_len + 0 + 100
    assert int(triton_v.verify_slot_indices[1].item()) == 1 * max_seq_len + 1 + 100
    assert int(triton_v.verify_slot_indices[2].item()) == 1 * max_seq_len + 2 + 100


def test_swa_lut_translates_seed_slot() -> None:
    """write_seed_slot_indices is also SWA-translated inside the plan kernel."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    full_pool_size = 4 * max_seq_len
    lut = (
        torch.arange(full_pool_size + 1, dtype=torch.int32, device=_DEVICE) + 100
    ).contiguous()

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=128,
        full_to_swa_index_mapping=lut,
    )

    # FULL slot at (rp=1, pos=2) = 1 * max_seq_len + 2 = 18; expected SWA seed = 18 + 100 = 118.
    assert int(triton_w.write_seed_slot_indices[0].item()) == 1 * max_seq_len + 2 + 100


def test_prev_slot_minus_one_at_chain_head() -> None:
    """pos=0 entry → verify_prev_slot_indices == -1 (chain head)."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # First entry has pos=0 → prev_slot = -1.
    assert int(triton_v.verify_prev_slot_indices[0].item()) == -1


def test_prev_slot_is_self_minus_one() -> None:
    """pos>0 entry → prev = req_to_token[rp, pos-1] (SWA-translated when SWA enabled)."""
    fb_req_pool_indices = torch.tensor([2], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([4], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    max_seq_len = 16
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # entry[1] is pos=1: prev_slot = req_to_token[2, 0] = 2 * max_seq_len + 0 = 32.
    assert int(triton_v.verify_prev_slot_indices[1].item()) == 2 * max_seq_len + 0
    # entry[2] is pos=2: prev_slot = req_to_token[2, 1] = 2 * max_seq_len + 1 = 33.
    assert int(triton_v.verify_prev_slot_indices[2].item()) == 2 * max_seq_len + 1


def test_extra_verify_entries_appended_after_per_req() -> None:
    """Extras land at ``verify_offsets[bs]`` (after the per-req-derived entries)."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)
    extras = _make_extras(
        slot_indices=[77, 78],
        positions=[5, 6],
        prev_slot_indices=[-1, 77],
        capacity=8,
    )

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # verify_num_valid = per_req(3) + extras(2) = 5; extras land at indices 3, 4.
    assert int(triton_v.verify_num_valid[0].item()) == 5
    assert int(triton_v.verify_slot_indices[3].item()) == 77
    assert int(triton_v.verify_slot_indices[4].item()) == 78
    assert int(triton_v.verify_positions[3].item()) == 5
    assert int(triton_v.verify_positions[4].item()) == 6
    assert int(triton_v.verify_prev_slot_indices[3].item()) == -1
    assert int(triton_v.verify_prev_slot_indices[4].item()) == 77


def test_extra_verify_num_valid_zero_no_op() -> None:
    """``extra_verify_num_valid == 0`` → extras tail is untouched; only per-req entries materialize."""
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([3], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)
    # Extras buffer is non-empty but extra_verify_num_valid = 0; kernel must not append anything.
    extras = _make_extras(
        slot_indices=[999, 999],
        positions=[999, 999],
        prev_slot_indices=[999, 999],
        capacity=8,
    )
    extras[3].zero_()

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    assert int(triton_v.verify_num_valid[0].item()) == 3


def test_sweep_caller_writes_dummy_write_plan() -> None:
    """``extend_seq_lens`` all zero → ``write_num_valid_reqs == 0``; VerifyPlan still populated."""
    fb_req_pool_indices = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([4, 6], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([0, 0], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # VerifyPlan covers 4+6 = 10 entries; write counts are zero.
    assert int(triton_v.verify_num_valid[0].item()) == 10
    # write_offsets cumsum of zeros stays zero across the active prefix.
    assert int(triton_w.write_offsets[0].item()) == 0
    assert int(triton_w.write_offsets[1].item()) == 0
    assert int(triton_w.write_offsets[2].item()) == 0
    # Seeds for write-empty reqs must be -1 per plan semantics.
    assert int(triton_w.write_seed_slot_indices[0].item()) == -1
    assert int(triton_w.write_seed_slot_indices[1].item()) == -1


def test_verify_num_valid_aggregate() -> None:
    """``verify_num_valid == sum(per-req verify_count) + extra_num_valid``."""
    fb_req_pool_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([2, 5, 4], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1, 1, 1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)
    extras = _make_extras(
        slot_indices=[80, 81, 82],
        positions=[10, 11, 12],
        prev_slot_indices=[-1, 80, 81],
        capacity=8,
    )

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # Aggregate = 2 + 5 + 4 + 3 = 14.
    assert int(triton_v.verify_num_valid[0].item()) == 14


def test_verify_covers_all_tokens_no_skip() -> None:
    """FULL group + bs=4 → verify_num_valid == Σ(prefix_lens) — every prefix token verified."""
    # Step: 4 reqs with mixed prefix and extend; FULL group means no SWA window clip.
    prefix_values = [0, 3, 7, 12]
    extend_values = [4, 1, 1, 1]
    fb_req_pool_indices = torch.tensor([1, 2, 3, 1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor(prefix_values, dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor(extend_values, dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=32)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=128, write_req_capacity=8
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    assert int(triton_v.verify_num_valid[0].item()) == sum(prefix_values)


def test_plan_verify_positions_strictly_increment_per_req() -> None:
    """Per req, verify_positions[verify_offsets[r]:verify_offsets[r+1]] == [window_start..prefix-1]."""
    fb_req_pool_indices = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([5, 8], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1, 1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=32)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # Req 0: positions [0..5); Req 1: positions [0..8).
    req0_positions = triton_v.verify_positions[:5].cpu().tolist()
    req1_positions = triton_v.verify_positions[5:13].cpu().tolist()
    assert req0_positions == [0, 1, 2, 3, 4]
    assert req1_positions == [0, 1, 2, 3, 4, 5, 6, 7]


def test_verify_covers_all_tokens_in_swa_window() -> None:
    """SWA group with window=128 + bs=4 → verify_num_valid == Σ min(prefix_lens[r], 128)."""
    window = 128
    prefix_values = [50, 128, 200, 1024]
    fb_req_pool_indices = torch.tensor([1, 2, 3, 1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor(prefix_values, dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1, 1, 1, 1], dtype=torch.int32, device=_DEVICE)
    max_seq_len = 2048
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)
    full_pool_size = 4 * max_seq_len
    lut = torch.arange(full_pool_size + 1, dtype=torch.int32, device=_DEVICE)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=1024, write_req_capacity=8
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=window,
        full_to_swa_index_mapping=lut,
    )

    expected_total = sum(min(p, window) for p in prefix_values)
    assert int(triton_v.verify_num_valid[0].item()) == expected_total


def test_write_num_valid_reqs_excludes_padding() -> None:
    """Padding rows (rpi == 0) at the tail must not be counted toward the active write-req count."""
    # bs=4, last two rows are padding.
    fb_req_pool_indices = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([3, 5, 99, 99], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1, 1, 99, 99], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=16)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=8
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # Padding rows must contribute 0 to write_offsets cumsum: [0, 1, 2, 2, 2].
    expected_write_offsets = [0, 1, 2, 2, 2]
    for i, value in enumerate(expected_write_offsets):
        assert int(triton_w.write_offsets[i].item()) == value


def test_byte_equal_python_reference() -> None:
    """End-to-end Triton vs Python ref byte-equal across a representative bs=4 case (no SWA)."""
    fb_req_pool_indices = torch.tensor([1, 2, 3, 1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0, 3, 8, 15], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([4, 1, 1, 1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=32)
    extras = _make_extras(
        slot_indices=[50, 51],
        positions=[100, 101],
        prev_slot_indices=[-1, 50],
        capacity=8,
    )

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=128, write_req_capacity=8
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=extras,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )



@pytest.mark.parametrize("hardcoded", [True])
def test_byte_equal_python_reference_hardcoded(hardcoded: bool) -> None:
    """bs=3, three prefix combinations → hand-computed verify_offsets / write_offsets / seed slots."""
    assert hardcoded

    # Step 1: pin (prefix, extend) per req.
    prefixes = [0, 4, 7]
    extends = [3, 1, 1]
    rps = [1, 2, 3]
    max_seq_len = 16

    fb_req_pool_indices = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor(prefixes, dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor(extends, dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64, write_req_capacity=4
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )

    # Step 2: hand-compute expected write_offsets (exclusive cumsum of extends) and verify_num_valid.
    expected_write_offsets = [0, 3, 4, 5]
    expected_verify_num_valid = sum(prefixes)
    expected_seeds = [
        -1,  # prefix=0 → no predecessor
        rps[1] * max_seq_len + (prefixes[1] - 1),
        rps[2] * max_seq_len + (prefixes[2] - 1),
    ]

    for i, value in enumerate(expected_write_offsets):
        assert (
            int(triton_w.write_offsets[i].item()) == value
        ), f"write_offsets[{i}] expected {value} got {int(triton_w.write_offsets[i].item())}"
    assert (
        int(triton_v.verify_num_valid[0].item()) == expected_verify_num_valid
    ), f"verify_num_valid expected {expected_verify_num_valid}"
    for i, expected_seed in enumerate(expected_seeds):
        assert (
            int(triton_w.write_seed_slot_indices[i].item()) == expected_seed
        ), f"write_seed_slot_indices[{i}] expected {expected_seed}"

    # Also confirm Triton == ref byte-equal.


@pytest.mark.parametrize(
    "bs",
    [1, 31, 32, 33, 128],
)
def test_bs_boundary_byte_equal_sweep(bs: int) -> None:
    """Sweep bs boundary values around Triton block boundaries; assert Triton vs ref byte-equal."""
    req_pool_indices = list(range(1, bs + 1))
    prefix_lens = [10] * bs
    extend_seq_lens = [1] * bs
    max_seq_len = 32

    fb_req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=bs + 1, max_seq_len=max_seq_len)

    total_verify = sum(min(p, max_seq_len) for p in prefix_lens)
    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=max(total_verify + 64, 256),
        write_req_capacity=bs + 4,
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )


@pytest.mark.parametrize(
    "prefix_val",
    [0, 1, 127, 128, 129, 4096],
)
def test_prefix_lens_boundary_byte_equal_sweep(prefix_val: int) -> None:
    """Sweep prefix_lens boundary values; assert Triton vs ref byte-equal."""
    max_seq_len = max(prefix_val + 4, 256)
    fb_req_pool_indices = torch.tensor([1, 2], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([prefix_val, 10], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([1, 1], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)

    total_verify = prefix_val + 10
    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=max(total_verify + 64, 256),
        write_req_capacity=8,
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )


@pytest.mark.parametrize(
    "extend_val",
    [1, 128, 4096],
)
def test_extend_seq_lens_boundary_byte_equal_sweep(extend_val: int) -> None:
    """Sweep extend_seq_lens boundary values; assert Triton vs ref byte-equal."""
    max_seq_len = max(extend_val + 4, 64)
    fb_req_pool_indices = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor([0], dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor([extend_val], dtype=torch.int32, device=_DEVICE)
    req_to_token = _build_req_to_token(max_reqs=4, max_seq_len=max_seq_len)

    triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
        verify_capacity=64,
        write_req_capacity=4,
    )
    _run_both_and_assert_byte_equal(
        triton_verify=triton_v,
        triton_write=triton_w,
        ref_verify=ref_v,
        ref_write=ref_w,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extras=_empty_extras(),
        swa_window_size=0,
        full_to_swa_index_mapping=None,
    )


import random as _random


def _build_random_plan_inputs(
    rng: _random.Random,
    *,
    bs: int,
    max_seq_len: int,
    max_prefix: int,
    max_extend: int,
    padding_fraction: float = 0.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Build random (fb_req_pool_indices, fb_prefix_lens, fb_extend_seq_lens, req_to_token, max_rp)."""
    max_rp = bs + 2
    req_pool_indices: list[int] = []
    prefix_lens: list[int] = []
    extend_lens: list[int] = []
    for row in range(bs):
        if row > 0 and rng.random() < padding_fraction:
            req_pool_indices.append(0)
            prefix_lens.append(0)
            extend_lens.append(0)
        else:
            rp = rng.randint(1, max_rp - 1)
            req_pool_indices.append(rp)
            pfx = rng.randint(0, max_prefix)
            ext = rng.randint(1, max_extend)
            prefix_lens.append(pfx)
            extend_lens.append(ext)
    fb_req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=_DEVICE)
    fb_prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
    rp_axis = torch.arange(max_rp, device=_DEVICE, dtype=torch.int32).unsqueeze(1)
    pos_axis = torch.arange(max_seq_len, device=_DEVICE, dtype=torch.int32).unsqueeze(0)
    req_to_token = (rp_axis * max_seq_len + pos_axis).contiguous()
    return fb_req_pool_indices, fb_prefix_lens, fb_extend_seq_lens, req_to_token, max_rp


def test_plan_pure_random_fuzz_byte_equal() -> None:
    """100 fully-random plan inputs — Triton vs ref must be byte-equal on every iteration."""
    rng = _random.Random(0)
    for iteration in range(100):
        bs = rng.randint(1, 16)
        max_seq_len = rng.randint(16, 128)
        max_prefix = max_seq_len - 1
        max_extend = rng.randint(1, 16)
        swa_enabled = rng.random() < 0.4
        swa_window_size = rng.randint(4, max_seq_len) if swa_enabled else 0

        fb_rpi, fb_pfx, fb_ext, req_to_token, max_rp = _build_random_plan_inputs(
            rng,
            bs=bs,
            max_seq_len=max_seq_len,
            max_prefix=max_prefix,
            max_extend=max_extend,
            padding_fraction=0.1,
        )

        extra_count = rng.randint(0, 4)
        if extra_count > 0:
            extra_slots_list = rng.sample(range(500, 600), extra_count)
            extra_positions_list = sorted(rng.sample(range(200, 300), extra_count))
            extra_prevs_list = [-1] + extra_slots_list[: extra_count - 1]
            extras = _make_extras(
                slot_indices=extra_slots_list,
                positions=extra_positions_list,
                prev_slot_indices=extra_prevs_list,
                capacity=extra_count + 2,
            )
        else:
            extras = _empty_extras()

        total_prefix = int(fb_pfx.sum().item())
        verify_capacity = max(total_prefix + extra_count + 64, 128)
        write_req_capacity = bs + 4

        full_to_swa_lut: Optional[torch.Tensor]
        if swa_window_size > 0:
            pool_size = max_rp * max_seq_len
            full_to_swa_lut = torch.arange(pool_size + 1, dtype=torch.int32, device=_DEVICE)
        else:
            full_to_swa_lut = None

        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=extras,
                swa_window_size=swa_window_size,
                full_to_swa_index_mapping=full_to_swa_lut,
            )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} "
                f"bs={bs} max_seq_len={max_seq_len} swa_window_size={swa_window_size} "
                f"extra_count={extra_count} "
                f"fb_rpi={fb_rpi.tolist()} fb_pfx={fb_pfx.tolist()} fb_ext={fb_ext.tolist()}"
            ) from exc


def test_plan_random_extend_only() -> None:
    """25 random extend-only batches (all prefix_lens=0) — byte-equal and write_offsets == cumsum(ext)."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 12)
        max_seq_len = 32
        extend_lens = [rng.randint(1, 12) for _ in range(bs)]
        rps = [rng.randint(1, bs + 1) for _ in range(bs)]

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.zeros(bs, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=64,
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert int(triton_v.verify_num_valid[0].item()) == 0, (
                f"iteration={iteration}: extend-only batch should have 0 verify entries"
            )
            cumsum = 0
            for i, ext in enumerate(extend_lens):
                assert int(triton_w.write_offsets[i].item()) == cumsum, (
                    f"iteration={iteration} i={i}: write_offsets mismatch"
                )
                cumsum += ext
            assert int(triton_w.write_offsets[bs].item()) == cumsum
            for i in range(bs):
                assert int(triton_w.write_seed_slot_indices[i].item()) == -1, (
                    f"iteration={iteration} i={i}: extend-only seed should be -1"
                )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} extend_lens={extend_lens} rps={rps}"
            ) from exc


def test_plan_random_decode_only() -> None:
    """25 random decode-only batches (all extend_lens=1, prefix>0) — byte-equal and seeds != -1."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 12)
        max_seq_len = 32
        prefix_lens = [rng.randint(1, 20) for _ in range(bs)]
        rps = list(range(1, bs + 1))

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.ones(bs, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        total_verify = sum(prefix_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=total_verify + 32,
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert int(triton_v.verify_num_valid[0].item()) == total_verify, (
                f"iteration={iteration}: verify_num_valid mismatch"
            )
            for i, (rp, pfx) in enumerate(zip(rps, prefix_lens)):
                expected_seed = rp * max_seq_len + (pfx - 1)
                assert int(triton_w.write_seed_slot_indices[i].item()) == expected_seed, (
                    f"iteration={iteration} i={i}: seed mismatch"
                )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} prefix_lens={prefix_lens} rps={rps}"
            ) from exc


def test_plan_random_mixed_extend_decode() -> None:
    """25 random mixed batches — byte-equal; verify_num_valid == sum(prefix_lens)."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(2, 10)
        max_seq_len = 32
        prefix_lens = [rng.randint(0, 15) for _ in range(bs)]
        extend_lens = [rng.randint(1, 8) for _ in range(bs)]
        rps = list(range(1, bs + 1))

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        total_verify = sum(prefix_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=max(total_verify + 32, 64),
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert int(triton_v.verify_num_valid[0].item()) == total_verify, (
                f"iteration={iteration}: verify_num_valid mismatch expected={total_verify}"
            )
            cumsum = 0
            for i, ext in enumerate(extend_lens):
                assert int(triton_w.write_offsets[i].item()) == cumsum, (
                    f"iteration={iteration} i={i}: write_offsets[{i}] expected={cumsum}"
                )
                cumsum += ext
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} prefix_lens={prefix_lens} extend_lens={extend_lens}"
            ) from exc


def test_plan_random_swa_clip_window_boundary() -> None:
    """25 random SWA batches — verify_num_valid == sum(min(pfx, window)); byte-equal."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 8)
        max_seq_len = 256
        window = rng.choice([16, 32, 64, 128])
        prefix_lens = [rng.randint(0, max_seq_len - 1) for _ in range(bs)]
        rps = list(range(1, bs + 1))

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.ones(bs, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)
        pool_size = (bs + 2) * max_seq_len
        lut = torch.arange(pool_size + 1, dtype=torch.int32, device=_DEVICE)

        expected_verify = sum(min(pfx, window) for pfx in prefix_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=max(expected_verify + 64, 128),
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=window,
                full_to_swa_index_mapping=lut,
            )
            assert int(triton_v.verify_num_valid[0].item()) == expected_verify, (
                f"iteration={iteration} window={window} prefix_lens={prefix_lens} "
                f"expected_verify={expected_verify}"
            )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} window={window} prefix_lens={prefix_lens}"
            ) from exc


def test_plan_random_sweep_extras_only() -> None:
    """25 random batches with extras and no per-req verify — byte-equal; extras appear at tail."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(1, 8)
        max_seq_len = 16
        rps = list(range(1, bs + 1))
        extend_lens = [rng.randint(1, 4) for _ in range(bs)]

        fb_rpi = torch.tensor(rps, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.zeros(bs, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=bs + 2, max_seq_len=max_seq_len)

        n_extras = rng.randint(1, 6)
        extra_slots = sorted(rng.sample(range(1000, 1100), n_extras))
        extra_positions = list(range(n_extras))
        extra_prevs = [-1] + extra_slots[: n_extras - 1]
        extras = _make_extras(
            slot_indices=extra_slots,
            positions=extra_positions,
            prev_slot_indices=extra_prevs,
            capacity=n_extras + 2,
        )

        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=n_extras + 32,
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=extras,
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert int(triton_v.verify_num_valid[0].item()) == n_extras, (
                f"iteration={iteration}: verify_num_valid expected={n_extras}"
            )
            for k, slot in enumerate(extra_slots):
                assert int(triton_v.verify_slot_indices[k].item()) == slot, (
                    f"iteration={iteration} k={k}: slot mismatch"
                )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} n_extras={n_extras} extra_slots={extra_slots}"
            ) from exc


def test_plan_random_padding_rows_mixed() -> None:
    """25 random batches with padding rows (rpi==0) mixed in — byte-equal; padding contributes nothing."""
    rng = _random.Random(0)
    for iteration in range(25):
        bs = rng.randint(3, 12)
        max_seq_len = 24
        max_rp = bs + 2

        req_pool_indices: list[int] = []
        prefix_lens: list[int] = []
        extend_lens: list[int] = []
        for row in range(bs):
            is_pad = row > 0 and rng.random() < 0.25
            if is_pad:
                req_pool_indices.append(0)
                prefix_lens.append(0)
                extend_lens.append(0)
            else:
                req_pool_indices.append(rng.randint(1, max_rp - 1))
                prefix_lens.append(rng.randint(0, 10))
                extend_lens.append(rng.randint(1, 6))

        fb_rpi = torch.tensor(req_pool_indices, dtype=torch.int32, device=_DEVICE)
        fb_pfx = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
        fb_ext = torch.tensor(extend_lens, dtype=torch.int32, device=_DEVICE)
        req_to_token = _build_req_to_token(max_reqs=max_rp, max_seq_len=max_seq_len)

        total_verify = sum(prefix_lens)
        total_extend = sum(extend_lens)
        triton_v, triton_w, ref_v, ref_w = _allocate_plan_pair(
            verify_capacity=max(total_verify + 32, 64),
            write_req_capacity=bs + 4,
        )
        try:
            _run_both_and_assert_byte_equal(
                triton_verify=triton_v,
                triton_write=triton_w,
                ref_verify=ref_v,
                ref_write=ref_w,
                fb_req_pool_indices=fb_rpi,
                fb_prefix_lens=fb_pfx,
                fb_extend_seq_lens=fb_ext,
                req_to_token=req_to_token,
                extras=_empty_extras(),
                swa_window_size=0,
                full_to_swa_index_mapping=None,
            )
            assert int(triton_v.verify_num_valid[0].item()) == total_verify, (
                f"iteration={iteration}: verify_num_valid expected={total_verify}"
            )
            assert int(triton_w.write_offsets[bs].item()) == total_extend, (
                f"iteration={iteration}: write_offsets[{bs}] expected={total_extend}"
            )
        except AssertionError as exc:
            raise AssertionError(
                f"iteration={iteration} bs={bs} "
                f"req_pool_indices={req_pool_indices} prefix_lens={prefix_lens} extend_lens={extend_lens}"
            ) from exc
