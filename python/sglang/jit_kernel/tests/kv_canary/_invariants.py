"""Ref/real-independent invariant assertions for kv_canary kernel tests.

Each invariant only looks at the kernel's inputs and outputs (shape relationships, monotonicity, tail
positions, etc.) — it must never re-implement the reference algorithm. Hand and fuzz tests both call
into this module so a single contract violation surfaces consistently.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import (
    _VIOLATION_FIELD_KERNEL_KIND,
    _VIOLATION_FIELD_SLOT_IDX,
    CanaryLaunchTag,
    VerifyPlan,
)
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode, WritePlan
from sglang.jit_kernel.tests.kv_canary.canary_helpers import FakeViolationLog

# Plan invariants


def assert_write_offsets_monotone(write_plan: WritePlan) -> None:
    n_active = int(write_plan.write_num_valid_reqs[0].item())
    if n_active < 0:
        raise AssertionError(f"write_num_valid_reqs negative: {n_active}")
    offsets = write_plan.write_offsets[: n_active + 1].detach().cpu().tolist()
    for i in range(len(offsets) - 1):
        assert (
            offsets[i] <= offsets[i + 1]
        ), f"write_offsets non-monotone at {i}: {offsets[i]} > {offsets[i + 1]}"


def assert_write_offsets_total_matches_active_extend_sum(
    *,
    write_plan: WritePlan,
    fb_extend_seq_lens: torch.Tensor,
    fb_req_pool_indices: torch.Tensor,
) -> None:
    n_active = int(write_plan.write_num_valid_reqs[0].item())
    total = int(write_plan.write_offsets[n_active].item())
    rpi_cpu = fb_req_pool_indices.detach().cpu().tolist()
    ext_cpu = fb_extend_seq_lens.detach().cpu().tolist()
    expected_total = sum(ext for rpi, ext in zip(rpi_cpu, ext_cpu) if rpi != 0)
    assert (
        total == expected_total
    ), f"write_offsets total {total} != active extend sum {expected_total}"


def assert_extras_land_at_tail(
    *,
    verify_plan: VerifyPlan,
    derived_verify_count: int,
    extras_slot_indices: torch.Tensor,
    extras_positions: torch.Tensor,
    extras_prev_slot_indices: torch.Tensor,
    extras_count: int,
) -> None:
    if extras_count == 0:
        return
    tail_start = derived_verify_count
    tail_end = derived_verify_count + extras_count
    n_valid = int(verify_plan.verify_num_valid[0].item())
    assert (
        tail_end <= n_valid
    ), f"extras tail {tail_end} exceeds verify_num_valid {n_valid}"
    plan_slots = verify_plan.verify_slot_indices[tail_start:tail_end]
    plan_positions = verify_plan.verify_positions[tail_start:tail_end]
    plan_prevs = verify_plan.verify_prev_slot_indices[tail_start:tail_end]
    assert torch.equal(plan_slots, extras_slot_indices[:extras_count])
    assert torch.equal(plan_positions, extras_positions[:extras_count])
    assert torch.equal(plan_prevs, extras_prev_slot_indices[:extras_count])


def assert_padding_row_seed_is_minus_one(
    *,
    write_plan: WritePlan,
    fb_req_pool_indices: torch.Tensor,
) -> None:
    n_active = int(write_plan.write_num_valid_reqs[0].item())
    if n_active == 0:
        return
    rpi_cpu = fb_req_pool_indices.detach().cpu().tolist()
    seeds_cpu = write_plan.write_seed_slot_indices[:n_active].detach().cpu().tolist()
    for r in range(min(n_active, len(rpi_cpu))):
        if rpi_cpu[r] == 0:
            assert seeds_cpu[r] == -1, f"padding row {r} has seed {seeds_cpu[r]} != -1"


def assert_prev_slot_minus_one_iff_chain_head(
    *,
    verify_plan: VerifyPlan,
    swa_window_size: int,
    derived_verify_count: int,
) -> None:
    if derived_verify_count == 0:
        return
    positions_cpu = (
        verify_plan.verify_positions[:derived_verify_count].detach().cpu().tolist()
    )
    prevs_cpu = (
        verify_plan.verify_prev_slot_indices[:derived_verify_count]
        .detach()
        .cpu()
        .tolist()
    )
    for i, (pos, prev) in enumerate(zip(positions_cpu, prevs_cpu)):
        if pos == 0:
            assert prev == -1, f"entry {i} at position 0 must have prev=-1, got {prev}"
        else:
            if swa_window_size == 0:
                assert (
                    prev != -1
                ), f"FULL entry {i} at position {pos} must have prev != -1, got {prev}"


def assert_verify_num_valid_equals_derived_plus_extras(
    *,
    verify_plan: VerifyPlan,
    fb_prefix_lens: torch.Tensor,
    fb_req_pool_indices: torch.Tensor,
    swa_window_size: int,
    extras_count: int,
) -> int:
    rpi_cpu = fb_req_pool_indices.detach().cpu().tolist()
    pfx_cpu = fb_prefix_lens.detach().cpu().tolist()
    derived = 0
    for rpi, pfx in zip(rpi_cpu, pfx_cpu):
        if rpi == 0:
            continue
        if swa_window_size > 0:
            window_start = max(0, pfx - swa_window_size)
            derived += max(0, pfx - window_start)
        else:
            derived += max(0, pfx)
    expected = derived + extras_count
    actual = int(verify_plan.verify_num_valid[0].item())
    assert (
        actual == expected
    ), f"verify_num_valid {actual} != derived {derived} + extras {extras_count} = {expected}"
    return derived


def assert_all_plan_invariants(
    *,
    verify_plan: VerifyPlan,
    write_plan: WritePlan,
    fb_req_pool_indices: torch.Tensor,
    fb_prefix_lens: torch.Tensor,
    fb_extend_seq_lens: torch.Tensor,
    swa_window_size: int,
    extras_slot_indices: torch.Tensor,
    extras_positions: torch.Tensor,
    extras_prev_slot_indices: torch.Tensor,
    extras_count: int,
) -> None:
    assert_write_offsets_monotone(write_plan)
    assert_write_offsets_total_matches_active_extend_sum(
        write_plan=write_plan,
        fb_extend_seq_lens=fb_extend_seq_lens,
        fb_req_pool_indices=fb_req_pool_indices,
    )
    derived = assert_verify_num_valid_equals_derived_plus_extras(
        verify_plan=verify_plan,
        fb_prefix_lens=fb_prefix_lens,
        fb_req_pool_indices=fb_req_pool_indices,
        swa_window_size=swa_window_size,
        extras_count=extras_count,
    )
    assert_extras_land_at_tail(
        verify_plan=verify_plan,
        derived_verify_count=derived,
        extras_slot_indices=extras_slot_indices,
        extras_positions=extras_positions,
        extras_prev_slot_indices=extras_prev_slot_indices,
        extras_count=extras_count,
    )
    assert_padding_row_seed_is_minus_one(
        write_plan=write_plan,
        fb_req_pool_indices=fb_req_pool_indices,
    )
    assert_prev_slot_minus_one_iff_chain_head(
        verify_plan=verify_plan,
        swa_window_size=swa_window_size,
        derived_verify_count=derived,
    )


# Verify invariants


def assert_canary_buf_unchanged(
    *,
    canary_buf_before: torch.Tensor,
    canary_buf_after: torch.Tensor,
) -> None:
    assert torch.equal(
        canary_buf_before, canary_buf_after
    ), "verify kernel mutated canary_buf (must be read-only)"


def assert_violation_count_le_active_entries(
    *,
    log_after: FakeViolationLog,
    log_before: FakeViolationLog,
    plan: VerifyPlan,
) -> None:
    delta = int(log_after.write_index[0].item()) - int(log_before.write_index[0].item())
    n_active = int(plan.verify_num_valid[0].item())
    assert (
        0 <= delta <= n_active
    ), f"violation_write_index delta {delta} out of [0, {n_active}]"


def assert_violation_rows_have_valid_slot_and_kernel_kind(
    *,
    log_after: FakeViolationLog,
    log_before: FakeViolationLog,
    plan: VerifyPlan,
    kernel_kind: CanaryLaunchTag,
) -> None:
    write_idx_after = int(log_after.write_index[0].item())
    write_idx_before = int(log_before.write_index[0].item())
    if write_idx_after == write_idx_before:
        return
    ring_capacity = log_after.ring.shape[0]
    visible_start = write_idx_before
    visible_end = min(write_idx_after, ring_capacity)
    if visible_end <= visible_start:
        return
    n_active = int(plan.verify_num_valid[0].item())
    plan_slots = set(plan.verify_slot_indices[:n_active].detach().cpu().tolist())
    rows = log_after.ring[visible_start:visible_end].detach().cpu()
    for i in range(rows.shape[0]):
        kind = int(rows[i, _VIOLATION_FIELD_KERNEL_KIND].item())
        assert kind == int(
            kernel_kind
        ), f"row {visible_start + i} kernel_kind {kind} != expected {int(kernel_kind)}"
        slot = int(rows[i, _VIOLATION_FIELD_SLOT_IDX].item())
        assert (
            slot in plan_slots
        ), f"row {visible_start + i} slot {slot} not in plan_slots"


def assert_slot_run_counter_incremented_by_active_entries(
    *,
    log_before: FakeViolationLog,
    log_after: FakeViolationLog,
    plan: VerifyPlan,
) -> None:
    n_active = int(plan.verify_num_valid[0].item())
    delta = int(log_after.slot_run_counter[0].item()) - int(
        log_before.slot_run_counter[0].item()
    )
    assert (
        delta == n_active
    ), f"slot_run_counter delta {delta} != active entries {n_active}"


def assert_kernel_run_counter_incremented_by_one(
    *,
    log_before: FakeViolationLog,
    log_after: FakeViolationLog,
) -> None:
    delta = int(log_after.kernel_run_counter[0].item()) - int(
        log_before.kernel_run_counter[0].item()
    )
    assert delta == 1, f"kernel_run_counter delta {delta} != 1"


def assert_all_verify_invariants(
    *,
    canary_buf_before: torch.Tensor,
    canary_buf_after: torch.Tensor,
    log_before: FakeViolationLog,
    log_after: FakeViolationLog,
    plan: VerifyPlan,
    kernel_kind: CanaryLaunchTag,
) -> None:
    assert_canary_buf_unchanged(
        canary_buf_before=canary_buf_before, canary_buf_after=canary_buf_after
    )
    assert_violation_count_le_active_entries(
        log_after=log_after, log_before=log_before, plan=plan
    )
    assert_violation_rows_have_valid_slot_and_kernel_kind(
        log_after=log_after,
        log_before=log_before,
        plan=plan,
        kernel_kind=kernel_kind,
    )
    assert_slot_run_counter_incremented_by_active_entries(
        log_before=log_before, log_after=log_after, plan=plan
    )
    assert_kernel_run_counter_incremented_by_one(
        log_before=log_before, log_after=log_after
    )


# Write invariants


def assert_written_slots_token_position_match_input(
    *,
    canary_buf_after: torch.Tensor,
    plan: WritePlan,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
) -> None:
    n_active = int(plan.write_num_valid_reqs[0].item())
    if n_active == 0:
        return
    offsets = plan.write_offsets[: n_active + 1].detach().cpu().tolist()
    total = offsets[n_active]
    slots_cpu = fb_out_cache_loc[:total].detach().cpu().tolist()
    tokens_cpu = fb_input_ids[:total].detach().cpu().tolist()
    pos_cpu = fb_positions[:total].detach().cpu().tolist()
    view = canary_buf_after.view(torch.int64)
    for i in range(total):
        slot = slots_cpu[i]
        if slot < 0:
            continue
        stored_token = int(view[slot, 0].item())
        stored_position = int(view[slot, 1].item())
        assert (
            stored_token == tokens_cpu[i]
        ), f"slot {slot}: stored token {stored_token} != input {tokens_cpu[i]}"
        assert (
            stored_position == pos_cpu[i]
        ), f"slot {slot}: stored position {stored_position} != input {pos_cpu[i]}"


def assert_slot_minus_one_skipped(
    *,
    canary_buf_before: torch.Tensor,
    canary_buf_after: torch.Tensor,
    plan: WritePlan,
    fb_out_cache_loc: torch.Tensor,
) -> None:
    n_active = int(plan.write_num_valid_reqs[0].item())
    if n_active == 0:
        return
    total = int(plan.write_offsets[n_active].item())
    slots_cpu = fb_out_cache_loc[:total].detach().cpu().tolist()
    written_slots = {s for s in slots_cpu if s >= 0}
    view_before = canary_buf_before.view(torch.int64)
    view_after = canary_buf_after.view(torch.int64)
    num_slots = canary_buf_after.shape[0]
    for slot in range(num_slots):
        if slot in written_slots:
            continue
        assert torch.equal(
            view_before[slot], view_after[slot]
        ), f"slot {slot} not in fb_out_cache_loc but canary_buf changed"


def assert_pseudo_violation_only_on_mismatch(
    *,
    pseudo_mode: CanaryPseudoMode,
    log_before: FakeViolationLog,
    log_after: FakeViolationLog,
    pseudo_expected_tokens: torch.Tensor,
    pseudo_expected_positions: torch.Tensor,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    plan: WritePlan,
) -> None:
    delta = int(log_after.write_index[0].item()) - int(log_before.write_index[0].item())
    if pseudo_mode == CanaryPseudoMode.OFF:
        assert delta == 0, f"pseudo_mode=OFF must produce no violations, got {delta}"
        return
    n_active = int(plan.write_num_valid_reqs[0].item())
    if n_active == 0:
        assert delta == 0, f"empty plan produced {delta} violations"
        return
    total = int(plan.write_offsets[n_active].item())
    tok = fb_input_ids[:total].detach().cpu().tolist()
    pos = fb_positions[:total].detach().cpu().tolist()
    exp_tok = pseudo_expected_tokens[:total].detach().cpu().tolist()
    exp_pos = pseudo_expected_positions[:total].detach().cpu().tolist()
    slots_cpu = fb_out_cache_loc[:total].detach().cpu().tolist()
    mismatch_entries = sum(
        1
        for i in range(total)
        if slots_cpu[i] >= 0 and (tok[i] != exp_tok[i] or pos[i] != exp_pos[i])
    )
    no_mismatch = mismatch_entries == 0
    if no_mismatch:
        assert (
            delta == 0
        ), f"pseudo_mode=ON with no mismatch produced {delta} violations"


def assert_write_slot_run_counter_incremented(
    *,
    log_before: FakeViolationLog,
    log_after: FakeViolationLog,
    plan: WritePlan,
    fb_out_cache_loc: torch.Tensor,
) -> None:
    n_active = int(plan.write_num_valid_reqs[0].item())
    if n_active == 0:
        delta = int(log_after.slot_run_counter[0].item()) - int(
            log_before.slot_run_counter[0].item()
        )
        assert delta == 0, f"empty plan incremented slot_run_counter by {delta}"
        return
    total = int(plan.write_offsets[n_active].item())
    delta = int(log_after.slot_run_counter[0].item()) - int(
        log_before.slot_run_counter[0].item()
    )
    assert (
        delta == total
    ), f"slot_run_counter delta {delta} != total write entries {total}"


def assert_write_kernel_run_counter_incremented_by_one(
    *,
    log_before: FakeViolationLog,
    log_after: FakeViolationLog,
) -> None:
    delta = int(log_after.kernel_run_counter[0].item()) - int(
        log_before.kernel_run_counter[0].item()
    )
    assert delta == 1, f"kernel_run_counter delta {delta} != 1"


def assert_all_write_invariants(
    *,
    canary_buf_before: torch.Tensor,
    canary_buf_after: torch.Tensor,
    plan: WritePlan,
    fb_input_ids: torch.Tensor,
    fb_positions: torch.Tensor,
    fb_out_cache_loc: torch.Tensor,
    pseudo_mode: CanaryPseudoMode,
    pseudo_expected_tokens: Optional[torch.Tensor],
    pseudo_expected_positions: Optional[torch.Tensor],
    log_before: FakeViolationLog,
    log_after: FakeViolationLog,
) -> None:
    assert_written_slots_token_position_match_input(
        canary_buf_after=canary_buf_after,
        plan=plan,
        fb_input_ids=fb_input_ids,
        fb_positions=fb_positions,
        fb_out_cache_loc=fb_out_cache_loc,
    )
    assert_slot_minus_one_skipped(
        canary_buf_before=canary_buf_before,
        canary_buf_after=canary_buf_after,
        plan=plan,
        fb_out_cache_loc=fb_out_cache_loc,
    )
    if (
        pseudo_mode == CanaryPseudoMode.ON
        and pseudo_expected_tokens is not None
        and pseudo_expected_positions is not None
    ):
        assert_pseudo_violation_only_on_mismatch(
            pseudo_mode=pseudo_mode,
            log_before=log_before,
            log_after=log_after,
            pseudo_expected_tokens=pseudo_expected_tokens,
            pseudo_expected_positions=pseudo_expected_positions,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            plan=plan,
        )
    elif pseudo_mode == CanaryPseudoMode.OFF:
        assert_pseudo_violation_only_on_mismatch(
            pseudo_mode=pseudo_mode,
            log_before=log_before,
            log_after=log_after,
            pseudo_expected_tokens=fb_input_ids,
            pseudo_expected_positions=fb_positions,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            plan=plan,
        )
    assert_write_slot_run_counter_incremented(
        log_before=log_before,
        log_after=log_after,
        plan=plan,
        fb_out_cache_loc=fb_out_cache_loc,
    )
    assert_write_kernel_run_counter_incremented_by_one(
        log_before=log_before, log_after=log_after
    )
