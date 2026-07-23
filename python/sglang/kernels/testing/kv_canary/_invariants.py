"""Ref/real-independent invariant assertions for kv_canary kernel tests.

Each invariant only looks at the kernel's inputs and outputs (shape relationships, monotonicity, tail
positions, etc.) — it must never re-implement the reference algorithm. Hand and fuzz tests both call
into this module so a single contract violation surfaces consistently.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.kv_canary import consts
from sglang.kernels.ops.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.kernels.ops.kv_canary.write import WritePlan
from sglang.kernels.testing.kv_canary._canary_helpers import FakeViolationLog


class PlanInvariants:
    @staticmethod
    def assert_all(
        *,
        verify_plan: VerifyPlan,
        write_plan: WritePlan,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        swa_window_size: int,
        extras_slot_indices: torch.Tensor,
        extras_positions: torch.Tensor,
        extras_prev_slot_indices: torch.Tensor,
        extras_count: int,
    ) -> None:
        PlanInvariants._assert_write_offsets_monotone(write_plan)
        PlanInvariants._assert_write_offsets_total_matches_active_extend_sum(
            write_plan=write_plan,
            extend_seq_lens=extend_seq_lens,
            req_pool_indices=req_pool_indices,
        )
        derived = PlanInvariants._assert_verify_num_valid_equals_derived_plus_extras(
            verify_plan=verify_plan,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            swa_window_size=swa_window_size,
            extras_count=extras_count,
        )
        PlanInvariants._assert_padding_row_seed_is_minus_one(
            write_plan=write_plan,
            req_pool_indices=req_pool_indices,
        )
        # In overflow (derived + extras > verify_capacity) the plan kernel disables
        # verify (enable=0) and the verify entries buffer is partially populated; the
        # downstream entry-shape invariants only hold when the kernel actually emitted
        # the full set, so guard them on enable=1.
        verify_enabled = int(verify_plan.enable[0].item()) == 1
        if verify_enabled:
            PlanInvariants._assert_extras_land_at_tail(
                verify_plan=verify_plan,
                derived_verify_count=derived,
                extras_slot_indices=extras_slot_indices,
                extras_positions=extras_positions,
                extras_prev_slot_indices=extras_prev_slot_indices,
                extras_count=extras_count,
            )
            PlanInvariants._assert_prev_slot_minus_one_iff_chain_head(
                verify_plan=verify_plan,
                swa_window_size=swa_window_size,
                derived_verify_count=derived,
            )

    @staticmethod
    def _assert_write_offsets_monotone(write_plan: WritePlan) -> None:
        n_active = int(write_plan.write_num_valid_reqs[0].item())
        if n_active < 0:
            raise AssertionError(f"write_num_valid_reqs negative: {n_active}")
        offsets = write_plan.write_offsets[: n_active + 1].detach().cpu().tolist()
        for i in range(len(offsets) - 1):
            assert (
                offsets[i] <= offsets[i + 1]
            ), f"write_offsets non-monotone at {i}: {offsets[i]} > {offsets[i + 1]}"

    @staticmethod
    def _assert_write_offsets_total_matches_active_extend_sum(
        *,
        write_plan: WritePlan,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        n_active = int(write_plan.write_num_valid_reqs[0].item())
        total = int(write_plan.write_offsets[n_active].item())
        rpi_cpu = req_pool_indices.detach().cpu().tolist()
        ext_cpu = extend_seq_lens.detach().cpu().tolist()
        expected_total = sum(ext for rpi, ext in zip(rpi_cpu, ext_cpu) if rpi != 0)
        assert (
            total == expected_total
        ), f"write_offsets total {total} != active extend sum {expected_total}"

    @staticmethod
    def _assert_extras_land_at_tail(
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
        plan_positions = verify_plan.verify_expected_positions[tail_start:tail_end]
        plan_prevs = verify_plan.verify_prev_slot_indices[tail_start:tail_end]
        assert torch.equal(plan_slots, extras_slot_indices[:extras_count])
        assert torch.equal(plan_positions, extras_positions[:extras_count])
        assert torch.equal(plan_prevs, extras_prev_slot_indices[:extras_count])

    @staticmethod
    def _assert_padding_row_seed_is_minus_one(
        *,
        write_plan: WritePlan,
        req_pool_indices: torch.Tensor,
    ) -> None:
        n_active = int(write_plan.write_num_valid_reqs[0].item())
        if n_active == 0:
            return
        rpi_cpu = req_pool_indices.detach().cpu().tolist()
        seeds_cpu = (
            write_plan.write_seed_slot_indices[:n_active].detach().cpu().tolist()
        )
        for r in range(min(n_active, len(rpi_cpu))):
            if rpi_cpu[r] == 0:
                assert (
                    seeds_cpu[r] == -1
                ), f"padding row {r} has seed {seeds_cpu[r]} != -1"

    @staticmethod
    def _assert_prev_slot_minus_one_iff_chain_head(
        *,
        verify_plan: VerifyPlan,
        swa_window_size: int,
        derived_verify_count: int,
    ) -> None:
        if derived_verify_count == 0:
            return
        positions_cpu = (
            verify_plan.verify_expected_positions[:derived_verify_count]
            .detach()
            .cpu()
            .tolist()
        )
        prevs_cpu = (
            verify_plan.verify_prev_slot_indices[:derived_verify_count]
            .detach()
            .cpu()
            .tolist()
        )
        for i, (pos, prev) in enumerate(zip(positions_cpu, prevs_cpu)):
            if pos == 0:
                assert (
                    prev == -1
                ), f"entry {i} at position 0 must have prev=-1, got {prev}"
            else:
                if swa_window_size == 0:
                    assert (
                        prev != -1
                    ), f"FULL entry {i} at position {pos} must have prev != -1, got {prev}"

    @staticmethod
    def _assert_verify_num_valid_equals_derived_plus_extras(
        *,
        verify_plan: VerifyPlan,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        swa_window_size: int,
        extras_count: int,
    ) -> int:
        rpi_cpu = req_pool_indices.detach().cpu().tolist()
        pfx_cpu = prefix_lens.detach().cpu().tolist()
        derived = 0
        for rpi, pfx in zip(rpi_cpu, pfx_cpu):
            if rpi == 0:
                continue
            if swa_window_size > 0:
                window_start = max(0, pfx - swa_window_size)
                derived += max(0, pfx - window_start)
            else:
                derived += max(0, pfx)
        # The plan kernel clamps verify_num_valid to verify_capacity and turns enable
        # off when (derived + extras) overflows the slot indices buffer. The invariant
        # must match that: on overflow the kernel records the capacity, on no overflow
        # it records the exact derived total (so the verify kernel scans every row).
        verify_capacity = int(verify_plan.verify_slot_indices.shape[0])
        expected_unclamped = derived + extras_count
        expected = min(expected_unclamped, verify_capacity)
        overflow = expected_unclamped > verify_capacity
        actual = int(verify_plan.verify_num_valid[0].item())
        assert actual == expected, (
            f"verify_num_valid {actual} != min(derived {derived} + extras {extras_count}, "
            f"verify_capacity {verify_capacity}) = {expected}"
        )
        enable = int(verify_plan.enable[0].item())
        expected_enable = 0 if overflow else 1
        assert enable == expected_enable, (
            f"verify_plan.enable {enable} != expected {expected_enable} "
            f"(overflow={overflow}; derived+extras={expected_unclamped}, "
            f"verify_capacity={verify_capacity})"
        )
        return derived


class VerifyInvariants:
    @staticmethod
    def assert_all(
        *,
        canary_buf_before: torch.Tensor,
        canary_buf_after: torch.Tensor,
        log_before: FakeViolationLog,
        log_after: FakeViolationLog,
        plan: VerifyPlan,
        kernel_kind: CanaryLaunchTag,
    ) -> None:
        VerifyInvariants._assert_canary_buf_unchanged(
            canary_buf_before=canary_buf_before, canary_buf_after=canary_buf_after
        )
        VerifyInvariants._assert_violation_count_le_active_entries(
            log_after=log_after, log_before=log_before, plan=plan
        )
        VerifyInvariants._assert_violation_rows_have_valid_slot_and_kernel_kind(
            log_after=log_after,
            log_before=log_before,
            plan=plan,
            kernel_kind=kernel_kind,
        )
        VerifyInvariants._assert_slot_run_counter_incremented_by_active_entries(
            log_before=log_before, log_after=log_after, plan=plan
        )
        VerifyInvariants._assert_kernel_run_counter_incremented_by_one(
            log_before=log_before, log_after=log_after
        )

    @staticmethod
    def _assert_canary_buf_unchanged(
        *,
        canary_buf_before: torch.Tensor,
        canary_buf_after: torch.Tensor,
    ) -> None:
        assert torch.equal(
            canary_buf_before, canary_buf_after
        ), "verify kernel mutated canary_buf (must be read-only)"

    @staticmethod
    def _assert_violation_count_le_active_entries(
        *,
        log_after: FakeViolationLog,
        log_before: FakeViolationLog,
        plan: VerifyPlan,
    ) -> None:
        delta = int(log_after.write_index[0].item()) - int(
            log_before.write_index[0].item()
        )
        n_active = int(plan.verify_num_valid[0].item())
        assert (
            0 <= delta <= n_active
        ), f"violation_write_index delta {delta} out of [0, {n_active}]"

    @staticmethod
    def _assert_violation_rows_have_valid_slot_and_kernel_kind(
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
            kind = int(rows[i, consts.VIOLATION_FIELD_KERNEL_KIND].item())
            assert kind == int(
                kernel_kind
            ), f"row {visible_start + i} kernel_kind {kind} != expected {int(kernel_kind)}"
            slot = int(rows[i, consts.VIOLATION_FIELD_SLOT_IDX].item())
            assert (
                slot in plan_slots
            ), f"row {visible_start + i} slot {slot} not in plan_slots"

    @staticmethod
    def _assert_slot_run_counter_incremented_by_active_entries(
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

    @staticmethod
    def _assert_kernel_run_counter_incremented_by_one(
        *,
        log_before: FakeViolationLog,
        log_after: FakeViolationLog,
    ) -> None:
        delta = int(log_after.kernel_run_counter[0].item()) - int(
            log_before.kernel_run_counter[0].item()
        )
        assert delta == 1, f"kernel_run_counter delta {delta} != 1"


class WriteInvariants:
    @staticmethod
    def assert_all(
        *,
        canary_buf_before: torch.Tensor,
        canary_buf_after: torch.Tensor,
        plan: WritePlan,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        enable_write_verify_inputs: bool,
        expected_input_tokens: Optional[torch.Tensor],
        expected_input_positions: Optional[torch.Tensor],
        log_before: FakeViolationLog,
        log_after: FakeViolationLog,
    ) -> None:
        WriteInvariants._assert_written_slots_token_position_match_input(
            canary_buf_after=canary_buf_after,
            plan=plan,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
        )
        WriteInvariants._assert_slot_minus_one_skipped(
            canary_buf_before=canary_buf_before,
            canary_buf_after=canary_buf_after,
            plan=plan,
            out_cache_loc=out_cache_loc,
        )
        WriteInvariants._assert_pseudo_violation_only_on_mismatch(
            enable_write_verify_inputs=enable_write_verify_inputs,
            log_before=log_before,
            log_after=log_after,
            expected_input_tokens=expected_input_tokens,
            expected_input_positions=expected_input_positions,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            plan=plan,
        )
        WriteInvariants._assert_write_slot_run_counter_incremented(
            log_before=log_before,
            log_after=log_after,
            plan=plan,
            out_cache_loc=out_cache_loc,
        )
        WriteInvariants._assert_write_kernel_run_counter_incremented_by_one(
            log_before=log_before, log_after=log_after
        )

    @staticmethod
    def _assert_written_slots_token_position_match_input(
        *,
        canary_buf_after: torch.Tensor,
        plan: WritePlan,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> None:
        n_active = int(plan.write_num_valid_reqs[0].item())
        if n_active == 0:
            return
        offsets = plan.write_offsets[: n_active + 1].detach().cpu().tolist()
        total = offsets[n_active]
        slots_cpu = out_cache_loc[:total].detach().cpu().tolist()
        tokens_cpu = input_ids[:total].detach().cpu().tolist()
        pos_cpu = positions[:total].detach().cpu().tolist()
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

    @staticmethod
    def _assert_slot_minus_one_skipped(
        *,
        canary_buf_before: torch.Tensor,
        canary_buf_after: torch.Tensor,
        plan: WritePlan,
        out_cache_loc: torch.Tensor,
    ) -> None:
        n_active = int(plan.write_num_valid_reqs[0].item())
        if n_active == 0:
            return
        total = int(plan.write_offsets[n_active].item())
        slots_cpu = out_cache_loc[:total].detach().cpu().tolist()
        written_slots = {s for s in slots_cpu if s >= 0}
        view_before = canary_buf_before.view(torch.int64)
        view_after = canary_buf_after.view(torch.int64)
        num_slots = canary_buf_after.shape[0]
        for slot in range(num_slots):
            if slot in written_slots:
                continue
            assert torch.equal(
                view_before[slot], view_after[slot]
            ), f"slot {slot} not in out_cache_loc but canary_buf changed"

    @staticmethod
    def _assert_pseudo_violation_only_on_mismatch(
        *,
        enable_write_verify_inputs: bool,
        log_before: FakeViolationLog,
        log_after: FakeViolationLog,
        expected_input_tokens: Optional[torch.Tensor],
        expected_input_positions: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        plan: WritePlan,
    ) -> None:
        delta = int(log_after.write_index[0].item()) - int(
            log_before.write_index[0].item()
        )
        if not enable_write_verify_inputs:
            assert (
                delta == 0
            ), f"enable_write_verify_inputs=OFF must produce no violations, got {delta}"
            return
        if expected_input_tokens is None or expected_input_positions is None:
            return
        n_active = int(plan.write_num_valid_reqs[0].item())
        if n_active == 0:
            assert delta == 0, f"empty plan produced {delta} violations"
            return
        total = int(plan.write_offsets[n_active].item())
        tok = input_ids[:total].detach().cpu().tolist()
        pos = positions[:total].detach().cpu().tolist()
        exp_tok = expected_input_tokens[:total].detach().cpu().tolist()
        exp_pos = expected_input_positions[:total].detach().cpu().tolist()
        slots_cpu = out_cache_loc[:total].detach().cpu().tolist()
        mismatch_entries = sum(
            1
            for i in range(total)
            if slots_cpu[i] >= 0 and (tok[i] != exp_tok[i] or pos[i] != exp_pos[i])
        )
        no_mismatch = mismatch_entries == 0
        if no_mismatch:
            assert (
                delta == 0
            ), f"enable_write_verify_inputs=ON with no mismatch produced {delta} violations"
        else:
            assert (
                delta == mismatch_entries
            ), f"write input mismatch count {mismatch_entries} produced {delta} violations"

    @staticmethod
    def _assert_write_slot_run_counter_incremented(
        *,
        log_before: FakeViolationLog,
        log_after: FakeViolationLog,
        plan: WritePlan,
        out_cache_loc: torch.Tensor,
    ) -> None:
        n_active = int(plan.write_num_valid_reqs[0].item())
        if n_active == 0:
            delta = int(log_after.slot_run_counter[0].item()) - int(
                log_before.slot_run_counter[0].item()
            )
            assert delta == 0, f"empty plan incremented slot_run_counter by {delta}"
            return
        total = int(plan.write_offsets[n_active].item())
        # The write kernel skips entries where out_cache_loc < 0 (the documented "mark
        # skip" path used by SWA-translated callers), so the slot_run_counter delta
        # tracks the count of writeable entries, not the planned total.
        writeable = int((out_cache_loc[:total] >= 0).sum().item())
        delta = int(log_after.slot_run_counter[0].item()) - int(
            log_before.slot_run_counter[0].item()
        )
        assert delta == writeable, (
            f"slot_run_counter delta {delta} != writeable entries {writeable} "
            f"(total={total}, skipped={total - writeable})"
        )

    @staticmethod
    def _assert_write_kernel_run_counter_incremented_by_one(
        *,
        log_before: FakeViolationLog,
        log_after: FakeViolationLog,
    ) -> None:
        delta = int(log_after.kernel_run_counter[0].item()) - int(
            log_before.kernel_run_counter[0].item()
        )
        assert delta == 1, f"kernel_run_counter delta {delta} != 1"
