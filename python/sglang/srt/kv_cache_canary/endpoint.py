"""Canary launch slot — a configuration bundle that fires one (head | tail | sweep) launch.

A :class:`CanaryEndpoint` packages everything the runner needs to invoke ``canary_verify_step`` (and
optionally ``canary_write_step``) for one (canary_buf, kernel_kind, real_kv_sources) triple. The endpoint
owns no mutable state — the runner threads the ``ViolationLog`` and health counters in per call. Frozen
dataclass so the runner cannot accidentally swap buffer handles between forwards.

Two launch methods:

- :meth:`CanaryEndpoint.launch_per_forward` — calls both ``canary_verify_step`` and ``canary_write_step``.
- :meth:`CanaryEndpoint.launch_sweep` — calls only ``canary_verify_step`` (sweep doesn't write).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.jit_kernel.kv_cache_canary_verify import (
    CanaryLaunchTag,
    RealKvHashMode,
    RealKvSource,
    VerifyPlan,
    canary_verify_step,
)
from sglang.jit_kernel.kv_cache_canary_write import (
    CanaryPseudoMode,
    WritePlan,
    canary_write_step,
)
from sglang.srt.kv_cache_canary.violation_state import ViolationLog


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryEndpoint:
    """One launch slot.

    Identifies the canary buffer the launch operates on, the :class:`CanaryLaunchTag` it stamps into
    violation rows, the :class:`RealKvSource` tuple it folds into the fingerprint, and the
    :class:`RealKvHashMode` toggle.

    Fields:
        canary_buf: Canary buffer this endpoint writes/verifies, shape ``[num_slots, CANARY_SLOT_BYTES]``,
            uint8.
        kernel_kind: :class:`CanaryLaunchTag` stamped into every violation row this endpoint produces.
        real_kv_sources: Real KV pieces folded into the per-slot ``real_kv_hash``. Empty tuple disables the
            mixin for this endpoint (kernel still runs in OFF mode).
        real_kv_hash_mode: :class:`RealKvHashMode` (OFF / BIT / ALL).
        full_to_swa_index_mapping: Optional SWA LUT. Required (non-None) iff this endpoint lives on the SWA
            group. Threaded straight into :func:`canary_write_step` for inline write-time slot translation.
    """

    canary_buf: torch.Tensor
    kernel_kind: CanaryLaunchTag
    real_kv_sources: tuple[RealKvSource, ...]
    real_kv_hash_mode: RealKvHashMode
    full_to_swa_index_mapping: Optional[torch.Tensor]

    def launch_per_forward(
        self,
        *,
        verify_plan: VerifyPlan,
        write_plan: WritePlan,
        fb_input_ids: torch.Tensor,
        fb_positions: torch.Tensor,
        fb_out_cache_loc: torch.Tensor,
        pseudo_mode: CanaryPseudoMode,
        pseudo_expected_tokens: torch.Tensor,
        pseudo_expected_positions: torch.Tensor,
        violation_log: ViolationLog,
        slot_run_counter: torch.Tensor,
        kernel_run_counter: torch.Tensor,
    ) -> None:
        """Verify then write.

        Per kernels.md §5: ``canary_verify_step`` checks the slot state predecessors against the chain
        invariant *before* the write overwrites them, then ``canary_write_step`` stamps in the new chain.
        Both share the same :class:`ViolationLog` and per-(head|tail|sweep) health counters so head and
        tail flavors of the same kernel kind feed into a single pair of counters.
        """
        canary_verify_step(
            canary_buf=self.canary_buf,
            plan=verify_plan,
            kernel_kind=self.kernel_kind,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=self.real_kv_sources,
            real_kv_hash_mode=self.real_kv_hash_mode,
        )
        canary_write_step(
            canary_buf=self.canary_buf,
            plan=write_plan,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            full_to_swa_index_mapping=self.full_to_swa_index_mapping,
            kernel_kind=self.kernel_kind,
            pseudo_mode=pseudo_mode,
            pseudo_expected_tokens=pseudo_expected_tokens,
            pseudo_expected_positions=pseudo_expected_positions,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=self.real_kv_sources,
            real_kv_hash_mode=self.real_kv_hash_mode,
        )

    def launch_sweep(
        self,
        *,
        verify_plan: VerifyPlan,
        violation_log: ViolationLog,
        slot_run_counter: torch.Tensor,
        kernel_run_counter: torch.Tensor,
    ) -> None:
        """Verify only — sweep callers never invoke ``canary_write_step`` (kernels.md §1.3)."""
        canary_verify_step(
            canary_buf=self.canary_buf,
            plan=verify_plan,
            kernel_kind=self.kernel_kind,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=self.real_kv_sources,
            real_kv_hash_mode=self.real_kv_hash_mode,
        )
