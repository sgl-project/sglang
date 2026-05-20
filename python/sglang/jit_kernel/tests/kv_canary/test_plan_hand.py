"""Hand-written differential tests: Triton canary_plan_step vs the torch reference, byte-equal."""

from __future__ import annotations

import random

import pytest
import torch

from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.plan_ref import canary_plan_step_torch_reference
from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.jit_kernel.tests.kv_canary._differential import (
    _run_both_and_assert_plan_byte_equal as _run_both_and_assert_byte_equal,
)
from sglang.jit_kernel.tests.kv_canary._fixtures import (
    _allocate_plan_pair,
    _build_req_to_token,
    _empty_extras,
    _make_extras,
    derive_plan_capacity,
    make_lut,
    make_req_to_token,
)
from sglang.jit_kernel.tests.kv_canary._invariants import assert_all_plan_invariants
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


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

    fb_req_pool_indices = torch.tensor(
        req_pool_indices, dtype=torch.int32, device=_DEVICE
    )
    fb_prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=_DEVICE)
    fb_extend_seq_lens = torch.tensor(
        extend_seq_lens, dtype=torch.int32, device=_DEVICE
    )
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


def _alloc_for_inputs(
    *,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    extras_count: int,
    swa_window_size: int,
) -> tuple[int, int]:
    bs = int(fb_req_pool_indices.shape[0])
    rpi_cpu = fb_req_pool_indices.detach().cpu().tolist()
    pfx_cpu = fb_prefix_lens.detach().cpu().tolist()
    ext_cpu = fb_extend_seq_lens.detach().cpu().tolist()
    total_verify = 0
    for rpi, pfx in zip(rpi_cpu, pfx_cpu):
        if rpi == 0:
            continue
        if swa_window_size > 0:
            window_start = max(0, pfx - swa_window_size)
            total_verify += max(0, pfx - window_start)
        else:
            total_verify += max(0, pfx)
    return derive_plan_capacity(
        kind="loose", total_verify=total_verify, extras_count=extras_count, bs=bs
    )


def _run_label(
    *,
    label: str,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    extras: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swa_window_size: int,
    full_to_swa_index_mapping: torch.Tensor | None,
    verify_capacity: int,
    write_req_capacity: int,
) -> tuple[VerifyPlan, WritePlan]:
    verify_plan = VerifyPlan.allocate(verify_capacity=verify_capacity, device=_DEVICE)
    write_plan = WritePlan.allocate(
        write_req_capacity=write_req_capacity, device=_DEVICE
    )
    runner = canary_plan_step if label == "real" else canary_plan_step_torch_reference
    extra_slots, extra_positions, extra_prevs, extra_num_valid = extras
    runner(
        verify_plan_out=verify_plan,
        write_plan_out=write_plan,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        extra_verify_slot_indices=extra_slots,
        extra_verify_positions=extra_positions,
        extra_verify_prev_slot_indices=extra_prevs,
        extra_verify_num_valid=extra_num_valid,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
    )
    torch.cuda.synchronize()
    return verify_plan, write_plan


def test_seed_translated_through_permuted_lut() -> None:
    """Permuted LUT: seed slot is the LUT-lookup of req_to_token[rp, prefix-1], NOT identity."""
    rng = random.Random(42)
    max_seq_len = 16
    max_reqs = 4
    pool_size = max_reqs * max_seq_len
    lut = make_lut(kind="permutation", pool_size=pool_size, device=_DEVICE, rng=rng)
    rtt = _build_req_to_token(max_reqs=max_reqs, max_seq_len=max_seq_len)

    rp = 2
    prefix = 5
    fb_rpi = torch.tensor([rp], dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor([prefix], dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor([1], dtype=torch.int32, device=_DEVICE)

    full_seed_slot = rp * max_seq_len + (prefix - 1)
    expected_seed = int(lut[full_seed_slot].item())

    extras = _empty_extras()
    verify_capacity, write_req_capacity = _alloc_for_inputs(
        fb_req_pool_indices=fb_rpi,
        fb_prefix_lens=fb_pfx,
        fb_extend_seq_lens=fb_ext,
        extras_count=0,
        swa_window_size=max_seq_len,
    )
    plans: dict[str, tuple[VerifyPlan, WritePlan]] = {}
    for label in ("real", "ref"):
        plans[label] = _run_label(
            label=label,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            req_to_token=rtt,
            extras=extras,
            swa_window_size=max_seq_len,
            full_to_swa_index_mapping=lut,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        v_plan, w_plan = plans[label]
        actual_seed = int(w_plan.write_seed_slot_indices[0].item())
        assert (
            actual_seed == expected_seed
        ), f"[{label}] permuted-LUT seed expected {expected_seed} got {actual_seed}"


def test_per_req_slot_when_req_to_token_is_sparse() -> None:
    """sparse_permuted rtt: verify_slot_indices read directly from the constructed table."""
    rng = random.Random(7)
    max_seq_len = 8
    max_reqs = 3
    rtt = make_req_to_token(
        kind="sparse_permuted",
        max_reqs=max_reqs,
        max_seq_len=max_seq_len,
        device=_DEVICE,
        rng=rng,
    )
    rp = 1
    prefix = 4
    fb_rpi = torch.tensor([rp], dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor([prefix], dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    extras = _empty_extras()
    verify_capacity, write_req_capacity = _alloc_for_inputs(
        fb_req_pool_indices=fb_rpi,
        fb_prefix_lens=fb_pfx,
        fb_extend_seq_lens=fb_ext,
        extras_count=0,
        swa_window_size=0,
    )

    expected_slots = [int(rtt[rp, pos].item()) for pos in range(prefix)]

    for label in ("real", "ref"):
        v_plan, _ = _run_label(
            label=label,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            req_to_token=rtt,
            extras=extras,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        actual_slots = v_plan.verify_slot_indices[:prefix].detach().cpu().tolist()
        assert (
            actual_slots == expected_slots
        ), f"[{label}] sparse-rtt slots expected {expected_slots} got {actual_slots}"


def test_extras_capacity_just_fits() -> None:
    """tight_match capacity: all extras land at the tail with no drop."""
    rp = 1
    prefix = 4
    fb_rpi = torch.tensor([rp], dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor([prefix], dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    rtt = _build_req_to_token(max_reqs=4, max_seq_len=16)

    extras_slot_list = [500, 501, 502]
    extras_position_list = [10, 11, 12]
    extras_prev_list = [-1, 500, 501]
    extras_count = len(extras_slot_list)
    extras = _make_extras(
        slot_indices=extras_slot_list,
        positions=extras_position_list,
        prev_slot_indices=extras_prev_list,
        capacity=extras_count,
    )

    total_verify = prefix
    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind="tight_match",
        total_verify=total_verify,
        extras_count=extras_count,
        bs=1,
    )

    for label in ("real", "ref"):
        v_plan, w_plan = _run_label(
            label=label,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            req_to_token=rtt,
            extras=extras,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        n = int(v_plan.verify_num_valid[0].item())
        assert n == total_verify + extras_count, f"[{label}] num_valid {n}"
        tail = (
            v_plan.verify_slot_indices[prefix : prefix + extras_count]
            .detach()
            .cpu()
            .tolist()
        )
        assert tail == extras_slot_list, f"[{label}] extras tail {tail}"


def test_extras_capacity_undershoot_by_one() -> None:
    """under_by_one capacity: the last extra is capped/dropped consistently across real and ref."""
    rp = 1
    prefix = 3
    fb_rpi = torch.tensor([rp], dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor([prefix], dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    rtt = _build_req_to_token(max_reqs=4, max_seq_len=16)

    extras_slot_list = [600, 601, 602]
    extras_position_list = [20, 21, 22]
    extras_prev_list = [-1, 600, 601]
    extras_count = len(extras_slot_list)
    extras = _make_extras(
        slot_indices=extras_slot_list,
        positions=extras_position_list,
        prev_slot_indices=extras_prev_list,
        capacity=extras_count,
    )

    total_verify = prefix
    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind="under_by_one",
        total_verify=total_verify,
        extras_count=extras_count,
        bs=1,
    )

    real_v, _ = _run_label(
        label="real",
        fb_req_pool_indices=fb_rpi,
        fb_prefix_lens=fb_pfx,
        fb_extend_seq_lens=fb_ext,
        req_to_token=rtt,
        extras=extras,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
    )
    ref_v, _ = _run_label(
        label="ref",
        fb_req_pool_indices=fb_rpi,
        fb_prefix_lens=fb_pfx,
        fb_extend_seq_lens=fb_ext,
        req_to_token=rtt,
        extras=extras,
        swa_window_size=0,
        full_to_swa_index_mapping=None,
        verify_capacity=verify_capacity,
        write_req_capacity=write_req_capacity,
    )
    n_real = int(real_v.verify_num_valid[0].item())
    n_ref = int(ref_v.verify_num_valid[0].item())
    assert n_real == n_ref, f"real {n_real} vs ref {n_ref} diverged under cap"
    assert (
        n_real <= verify_capacity
    ), f"real n_valid {n_real} exceeded cap {verify_capacity}"


def test_swa_window_head_prev_slot_is_real_predecessor() -> None:
    """SWA window with non-zero window_start: head entry's prev_slot != -1; it is the real predecessor."""
    rng = random.Random(13)
    max_seq_len = 256
    max_reqs = 2
    pool_size = max_reqs * max_seq_len
    swa_window_size = 128
    prefix = 200
    rp = 1
    lut = make_lut(kind="permutation", pool_size=pool_size, device=_DEVICE, rng=rng)
    rtt = _build_req_to_token(max_reqs=max_reqs, max_seq_len=max_seq_len)

    fb_rpi = torch.tensor([rp], dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor([prefix], dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    extras = _empty_extras()

    window_start = prefix - swa_window_size
    full_prev_slot = int(rtt[rp, window_start - 1].item())
    expected_prev = int(lut[full_prev_slot].item())

    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind="loose", total_verify=swa_window_size, extras_count=0, bs=1
    )

    for label in ("real", "ref"):
        v_plan, _ = _run_label(
            label=label,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            req_to_token=rtt,
            extras=extras,
            swa_window_size=swa_window_size,
            full_to_swa_index_mapping=lut,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        actual_prev = int(v_plan.verify_prev_slot_indices[0].item())
        assert (
            actual_prev != -1
        ), f"[{label}] SWA window head must have real predecessor, got -1"
        assert (
            actual_prev == expected_prev
        ), f"[{label}] expected prev={expected_prev} got {actual_prev}"


def test_replay_same_inputs_yields_same_outputs() -> None:
    """Two consecutive runs on identical inputs produce byte-equal plans (kernel is pure)."""
    fb_rpi = torch.tensor([1, 2, 3], dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor([4, 7, 2], dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor([2, 1, 3], dtype=torch.int32, device=_DEVICE)
    rtt = _build_req_to_token(max_reqs=8, max_seq_len=16)
    extras = _empty_extras()
    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind="loose", total_verify=13, extras_count=0, bs=3
    )

    for label in ("real", "ref"):
        run1_v, run1_w = _run_label(
            label=label,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            req_to_token=rtt,
            extras=extras,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        run2_v, run2_w = _run_label(
            label=label,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            req_to_token=rtt,
            extras=extras,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        assert torch.equal(
            run1_v.verify_slot_indices, run2_v.verify_slot_indices
        ), label
        assert torch.equal(run1_v.verify_positions, run2_v.verify_positions), label
        assert torch.equal(
            run1_v.verify_prev_slot_indices, run2_v.verify_prev_slot_indices
        ), label
        assert torch.equal(run1_v.verify_num_valid, run2_v.verify_num_valid), label
        assert torch.equal(run1_w.write_offsets, run2_w.write_offsets), label
        assert torch.equal(
            run1_w.write_seed_slot_indices, run2_w.write_seed_slot_indices
        ), label
        assert torch.equal(
            run1_w.write_num_valid_reqs, run2_w.write_num_valid_reqs
        ), label


def test_padding_row_with_garbage_prefix_does_not_oob() -> None:
    """rpi==0 padding row with absurd prefix_lens must not OOB-read req_to_token (row is skipped)."""
    fb_rpi = torch.tensor([1, 0, 2], dtype=torch.int32, device=_DEVICE)
    fb_pfx = torch.tensor([5, 99999, 3], dtype=torch.int32, device=_DEVICE)
    fb_ext = torch.tensor([1, 99999, 1], dtype=torch.int32, device=_DEVICE)
    rtt = _build_req_to_token(max_reqs=4, max_seq_len=16)
    extras = _empty_extras()
    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind="loose", total_verify=8, extras_count=0, bs=3
    )

    for label in ("real", "ref"):
        v_plan, w_plan = _run_label(
            label=label,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            req_to_token=rtt,
            extras=extras,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
            write_req_capacity=write_req_capacity,
        )
        assert int(v_plan.verify_num_valid[0].item()) == 8, label
        assert (
            int(w_plan.write_seed_slot_indices[1].item()) == -1
        ), f"[{label}] padding row seed must be -1"
        assert_all_plan_invariants(
            verify_plan=v_plan,
            write_plan=w_plan,
            fb_req_pool_indices=fb_rpi,
            fb_prefix_lens=fb_pfx,
            fb_extend_seq_lens=fb_ext,
            swa_window_size=0,
            extras_slot_indices=extras[0],
            extras_positions=extras[1],
            extras_prev_slot_indices=extras[2],
            extras_count=0,
        )


def test_shrink_bs_clears_stale_write_offsets() -> None:
    """Reusing a WritePlan with smaller bs: write_offsets beyond new bs must be zeroed by the kernel."""
    rtt = _build_req_to_token(max_reqs=16, max_seq_len=16)
    extras = _empty_extras()
    verify_capacity, write_req_capacity = derive_plan_capacity(
        kind="loose", total_verify=80, extras_count=0, bs=8
    )

    for label in ("real", "ref"):
        big_rpi = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=_DEVICE
        )
        big_pfx = torch.tensor([10] * 8, dtype=torch.int32, device=_DEVICE)
        big_ext = torch.tensor([1] * 8, dtype=torch.int32, device=_DEVICE)
        small_rpi = torch.tensor([1, 2, 3], dtype=torch.int32, device=_DEVICE)
        small_pfx = torch.tensor([5, 5, 5], dtype=torch.int32, device=_DEVICE)
        small_ext = torch.tensor([1, 1, 1], dtype=torch.int32, device=_DEVICE)

        verify_plan = VerifyPlan.allocate(
            verify_capacity=verify_capacity, device=_DEVICE
        )
        write_plan = WritePlan.allocate(
            write_req_capacity=write_req_capacity, device=_DEVICE
        )
        runner = (
            canary_plan_step if label == "real" else canary_plan_step_torch_reference
        )
        runner(
            verify_plan_out=verify_plan,
            write_plan_out=write_plan,
            fb_req_pool_indices=big_rpi,
            fb_prefix_lens=big_pfx,
            fb_extend_seq_lens=big_ext,
            req_to_token=rtt,
            extra_verify_slot_indices=extras[0],
            extra_verify_positions=extras[1],
            extra_verify_prev_slot_indices=extras[2],
            extra_verify_num_valid=extras[3],
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        torch.cuda.synchronize()
        runner(
            verify_plan_out=verify_plan,
            write_plan_out=write_plan,
            fb_req_pool_indices=small_rpi,
            fb_prefix_lens=small_pfx,
            fb_extend_seq_lens=small_ext,
            req_to_token=rtt,
            extra_verify_slot_indices=extras[0],
            extra_verify_positions=extras[1],
            extra_verify_prev_slot_indices=extras[2],
            extra_verify_num_valid=extras[3],
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        torch.cuda.synchronize()
        n_active = int(write_plan.write_num_valid_reqs[0].item())
        tail_offsets = (
            write_plan.write_offsets[n_active + 1 : 8].detach().cpu().tolist()
        )
        assert all(
            v == 0 for v in tail_offsets
        ), f"[{label}] stale write_offsets tail not cleared: {tail_offsets}"
