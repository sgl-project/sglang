from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    VerifyOrWriteContext,
    VerifyPlan,
    launch_canary_verify_kernel,
)
from sglang.jit_kernel.kv_canary.write import (
    WritePlan,
    launch_canary_write_kernel,
)
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.state import (
    CanaryDeviceState,
    ViolationLog,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryEndpoint:
    kernel_kind: CanaryLaunchTag
    canary_buf: torch.Tensor
    full_to_swa_index_mapping: Optional[torch.Tensor]
    slot_run_counter_view: torch.Tensor
    kernel_run_counter_view: torch.Tensor
    enable_chain_position_assert: torch.Tensor

    def launch_per_forward(
        self,
        *,
        verify_plan: VerifyPlan,
        write_plan: WritePlan,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        enable_write_input_assert: bool,
        enable_verify_token_assert: bool,
        expected_inputs: ExpectedInputs,
        violation_log: ViolationLog,
    ) -> None:
        context = self._make_verify_or_write_context(
            violation_log=violation_log,
        )
        launch_canary_verify_kernel(
            context=context,
            plan=verify_plan,
            check_verify_expected_token=enable_verify_token_assert,
        )

        # SWA endpoints translate the per-token slot indices via a device tensor index op before invoking the write kernel.
        if self.full_to_swa_index_mapping is not None:
            out_cache_loc_for_canary = self.full_to_swa_index_mapping[out_cache_loc]
        else:
            out_cache_loc_for_canary = out_cache_loc
        if enable_write_input_assert:
            expected_input_tokens = expected_inputs.tokens
            expected_input_positions = expected_inputs.positions
        else:
            expected_input_tokens = None
            expected_input_positions = None

        launch_canary_write_kernel(
            context=context,
            plan=write_plan,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc_for_canary,
            enable_write_input_assert=enable_write_input_assert,
            expected_input_tokens=expected_input_tokens,
            expected_input_positions=expected_input_positions,
        )

    def _make_verify_or_write_context(
        self,
        *,
        violation_log: ViolationLog,
    ) -> VerifyOrWriteContext:
        return VerifyOrWriteContext(
            canary_buf=self.canary_buf,
            kernel_kind=self.kernel_kind,
            violation_ring=violation_log.violation_ring,
            violation_write_index=violation_log.violation_write_index,
            slot_run_counter=self.slot_run_counter_view,
            kernel_run_counter=self.kernel_run_counter_view,
            enable_chain_position_assert=self.enable_chain_position_assert,
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


_FULL_LAYOUT: tuple[tuple[CanaryLaunchTag, str, str], ...] = (
    (CanaryLaunchTag.HEAD_K_FULL, "HEAD", "K"),
    (CanaryLaunchTag.HEAD_V_FULL, "HEAD", "V"),
    (CanaryLaunchTag.TAIL_K_FULL, "TAIL", "K"),
    (CanaryLaunchTag.TAIL_V_FULL, "TAIL", "V"),
)


_SWA_LAYOUT: tuple[tuple[CanaryLaunchTag, str, str], ...] = (
    (CanaryLaunchTag.HEAD_K_SWA, "HEAD", "K"),
    (CanaryLaunchTag.HEAD_V_SWA, "HEAD", "V"),
    (CanaryLaunchTag.TAIL_K_SWA, "TAIL", "K"),
    (CanaryLaunchTag.TAIL_V_SWA, "TAIL", "V"),
)


def build_endpoints_from_group(
    *,
    group: CanaryBufferGroup,
    device_state: CanaryDeviceState,
) -> tuple[CanaryEndpoint, ...]:
    """Enumerate (slot × half) endpoints for one CanaryBufferGroup."""
    pool_kind = group.kind
    layout = _FULL_LAYOUT if pool_kind is PoolKind.FULL else _SWA_LAYOUT

    endpoints: list[CanaryEndpoint] = []
    for tag, slot, half in layout:
        if half == "V" and not group.has_v_half:
            continue

        canary_buf = _resolve_canary_buf(slot=slot, half=half, group=group)
        lut = group.swa_index_lut if pool_kind is PoolKind.SWA else None
        slot_view = device_state.slot_run_counters[tag.value : tag.value + 1]
        kernel_view = device_state.kernel_run_counters[tag.value : tag.value + 1]
        endpoints.append(
            CanaryEndpoint(
                kernel_kind=tag,
                canary_buf=canary_buf,
                full_to_swa_index_mapping=lut,
                slot_run_counter_view=slot_view,
                kernel_run_counter_view=kernel_view,
                enable_chain_position_assert=device_state.enable_chain_position_assert,
            )
        )

    return tuple(endpoints)
