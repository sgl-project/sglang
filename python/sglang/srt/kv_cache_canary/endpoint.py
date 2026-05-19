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
    CanaryPseudoMode as CanaryInputCheckMode,
)
from sglang.jit_kernel.kv_cache_canary_write import (
    WritePlan,
    canary_write_step,
)
from sglang.srt.kv_cache_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_cache_canary.violation_state import (
    CanaryDeviceState,
    ViolationLog,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryEndpoint:
    """One canary launch unit. Bundles everything a single (slot × K-half | V-half) launch needs.

    A canary attached to one pool produces up to 12 endpoints: 3 slots (head / tail / sweep) × 2 halves
    (K / V) × 2 groups (FULL / SWA). MLA-style pools have no V half (6 endpoints max). The SWA group
    only exists on pools whose model has SWA layers.

    Endpoints are constructed once at install time, frozen, and looked up by the runner per launch.

    Fields:
        kernel_kind: CanaryLaunchTag identifying this endpoint's slot × half × group. Stamped into every
            violation row this endpoint produces.
        canary_buf: This endpoint's canary buffer, shape [num_slots, CANARY_SLOT_BYTES], uint8.
            head and tail endpoints on the same (half, group) hold DISTINCT buffers (so they can be
            staged at different forward-pass points without overwriting); sweep endpoint shares its
            buffer with one of head/tail (typically tail — sweep verifies the most recent canary state).
        full_to_swa_index_mapping: SWA LUT, shape [full_pool_size + 1], int32, or None. None iff this
            endpoint is on the FULL group.
        real_kv_sources: RealKvSource tuple folded into real_kv_hash. Length 0..4 (kernels.md §2.4.1).
            Empty tuple disables the mixin for this endpoint.
        slot_run_counter_view: One-element int64 view into CanaryDeviceState.slot_run_counters at
            kernel_kind's slot index.
        kernel_run_counter_view: One-element int64 view into CanaryDeviceState.kernel_run_counters at
            kernel_kind's slot index.
    """

    kernel_kind: CanaryLaunchTag
    canary_buf: torch.Tensor
    full_to_swa_index_mapping: Optional[torch.Tensor]
    real_kv_sources: tuple[RealKvSource, ...]
    slot_run_counter_view: torch.Tensor
    kernel_run_counter_view: torch.Tensor

    def launch_per_forward(
        self,
        *,
        verify_plan: VerifyPlan,
        write_plan: WritePlan,
        fb_input_ids: torch.Tensor,
        fb_positions: torch.Tensor,
        fb_out_cache_loc: torch.Tensor,
        input_check_mode: CanaryInputCheckMode,
        expected_input_tokens: torch.Tensor,
        expected_input_positions: torch.Tensor,
        violation_log: ViolationLog,
        real_kv_hash_mode: RealKvHashMode,
    ) -> None:
        """Call canary_verify_step then canary_write_step against this endpoint's canary_buf. Both use
        the shared violation_log. Used by head and tail endpoints; sweep endpoints raise NotImplementedError.
        """
        if _is_sweep_tag(self.kernel_kind):
            raise NotImplementedError(
                f"kv-canary: launch_per_forward not supported on sweep endpoint {self.kernel_kind.name}"
            )

        canary_verify_step(
            canary_buf=self.canary_buf,
            plan=verify_plan,
            kernel_kind=self.kernel_kind,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=self.slot_run_counter_view,
            kernel_run_counter=self.kernel_run_counter_view,
            real_kv_sources=self.real_kv_sources,
            real_kv_hash_mode=real_kv_hash_mode,
        )
        canary_write_step(
            canary_buf=self.canary_buf,
            plan=write_plan,
            fb_input_ids=fb_input_ids,
            fb_positions=fb_positions,
            fb_out_cache_loc=fb_out_cache_loc,
            full_to_swa_index_mapping=self.full_to_swa_index_mapping,
            kernel_kind=self.kernel_kind,
            pseudo_mode=input_check_mode,
            pseudo_expected_tokens=expected_input_tokens,
            pseudo_expected_positions=expected_input_positions,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=self.slot_run_counter_view,
            kernel_run_counter=self.kernel_run_counter_view,
            real_kv_sources=self.real_kv_sources,
            real_kv_hash_mode=real_kv_hash_mode,
        )

    def launch_sweep(
        self,
        *,
        verify_plan: VerifyPlan,
        violation_log: ViolationLog,
        real_kv_hash_mode: RealKvHashMode,
    ) -> None:
        """Call only canary_verify_step against this endpoint's canary_buf. Used by sweep endpoints;
        head/tail endpoints raise NotImplementedError.
        """
        if not _is_sweep_tag(self.kernel_kind):
            raise NotImplementedError(
                f"kv-canary: launch_sweep not supported on non-sweep endpoint {self.kernel_kind.name}"
            )

        canary_verify_step(
            canary_buf=self.canary_buf,
            plan=verify_plan,
            kernel_kind=self.kernel_kind,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=self.slot_run_counter_view,
            kernel_run_counter=self.kernel_run_counter_view,
            real_kv_sources=self.real_kv_sources,
            real_kv_hash_mode=real_kv_hash_mode,
        )


def _is_sweep_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.SWEEP_K_FULL,
        CanaryLaunchTag.SWEEP_V_FULL,
        CanaryLaunchTag.SWEEP_K_SWA,
        CanaryLaunchTag.SWEEP_V_SWA,
    )


def _resolve_canary_buf(
    *,
    slot: str,
    half: str,
    group: CanaryBufferGroup,
) -> torch.Tensor:
    if half == "K":
        if slot == "HEAD":
            return group.k_head
        return group.k_tail
    if slot == "HEAD":
        return group.v_head
    return group.v_tail


def _resolve_real_kv_sources(
    *,
    half: str,
    group: CanaryBufferGroup,
) -> tuple[RealKvSource, ...]:
    if half == "K":
        return group.real_kv_sources_k
    return group.real_kv_sources_v


def build_endpoints_from_group(
    *,
    group: CanaryBufferGroup,
    device_state: CanaryDeviceState,
) -> tuple[CanaryEndpoint, ...]:
    """Enumerate (slot × half) endpoints for one CanaryBufferGroup.

    Produces up to 6 endpoints per group: 3 slots (HEAD / TAIL / SWEEP) × 2 halves (K / V). MLA-style
    pools (group.has_v_half == False) skip the V half → 3 endpoints. Head and tail get distinct canary
    buffers (group.k_head vs group.k_tail; group.v_head vs group.v_tail); the sweep endpoint shares its
    canary buffer with the tail endpoint of the same half.

    Counter views: slot_run_counter_view / kernel_run_counter_view are 1-element views into
    device_state.slot_run_counters / device_state.kernel_run_counters at kernel_kind.value, so kernel
    atomicAdd writes through to the original tensor.

    The SWA LUT (group.swa_index_lut) is threaded into the endpoint's full_to_swa_index_mapping for SWA
    groups and left None for FULL groups.
    """
    pool_kind = group.kind
    expected_suffix = pool_kind.name

    endpoints: list[CanaryEndpoint] = []
    for tag in CanaryLaunchTag:
        slot, half, suffix = tag.name.split("_")

        if suffix != expected_suffix:
            continue
        if half == "V" and not group.has_v_half:
            continue

        buf_slot = "TAIL" if slot == "SWEEP" else slot
        canary_buf = _resolve_canary_buf(slot=buf_slot, half=half, group=group)
        real_kv_sources = _resolve_real_kv_sources(half=half, group=group)
        lut = group.swa_index_lut if pool_kind is PoolKind.SWA else None
        slot_view = device_state.slot_run_counters[tag.value : tag.value + 1]
        kernel_view = device_state.kernel_run_counters[tag.value : tag.value + 1]
        endpoints.append(
            CanaryEndpoint(
                kernel_kind=tag,
                canary_buf=canary_buf,
                full_to_swa_index_mapping=lut,
                real_kv_sources=real_kv_sources,
                slot_run_counter_view=slot_view,
                kernel_run_counter_view=kernel_view,
            )
        )

    return tuple(endpoints)
