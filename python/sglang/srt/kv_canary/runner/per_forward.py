from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode as CanaryInputCheckMode
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.mock_model.sampler import fill_expected_inputs
from sglang.srt.kv_canary.plan_input import build_plan_input_per_forward

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PerForwardOrchestrator:
    def __init__(
        self,
        *,
        owner: "CanaryRunner",
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        per_forward_write_entry_capacity: int,
    ) -> None:
        self._owner = owner
        device = owner._device
        self._verify_plan_per_forward = VerifyPlan.allocate(
            verify_capacity=max(1, per_forward_verify_capacity), device=device
        )
        self._write_plan_per_forward = WritePlan.allocate(
            write_req_capacity=max(1, per_forward_write_req_capacity), device=device
        )
        write_entry_capacity = max(1, per_forward_write_entry_capacity)
        self._expected_input_tokens = torch.zeros(
            write_entry_capacity, dtype=torch.int32, device=device
        )
        self._expected_input_positions = torch.zeros(
            write_entry_capacity, dtype=torch.int32, device=device
        )
        self._last_forward_batch: Optional["ForwardBatch"] = None

    def forward_step_before_model(self, forward_batch: "ForwardBatch") -> None:
        owner = self._owner
        if owner.config.mode == "off":
            return

        owner._perturb.perturb_hook(forward_batch)
        owner._perturb.perturb_real_kv_hook(forward_batch)
        self._last_forward_batch = forward_batch

        if owner.config.input_check_mode == CanaryInputCheckMode.ON:
            fill_expected_inputs(
                forward_batch=forward_batch,
                expected_input_tokens_out=self._expected_input_tokens,
                expected_input_positions_out=self._expected_input_positions,
            )

        for pool_idx, groups in enumerate(owner._groups_per_pool):
            for group in groups:
                plan_input = build_plan_input_per_forward(
                    forward_batch=forward_batch,
                    swa_window_size=(
                        owner._swa_window_size if group.kind is PoolKind.SWA else 0
                    ),
                    full_to_swa_index_mapping=group.swa_index_lut,
                )
                owner._invoke_plan(
                    plan_input=plan_input,
                    verify_plan=self._verify_plan_per_forward,
                    write_plan=self._write_plan_per_forward,
                    group=group,
                )
                owner._launch_endpoints(
                    pool_idx=pool_idx,
                    group=group,
                    tag_filter=_is_head_tag,
                    verify_plan=self._verify_plan_per_forward,
                    forward_batch=forward_batch,
                )

    def forward_step_after_model(self) -> None:
        owner = self._owner
        if owner.config.mode == "off":
            return
        forward_batch = self._last_forward_batch
        if forward_batch is None:
            return

        for pool_idx, groups in enumerate(owner._groups_per_pool):
            for group in groups:
                owner._launch_endpoints(
                    pool_idx=pool_idx,
                    group=group,
                    tag_filter=_is_tail_tag,
                    verify_plan=self._verify_plan_per_forward,
                    forward_batch=forward_batch,
                )


def _is_head_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.HEAD_K_FULL,
        CanaryLaunchTag.HEAD_V_FULL,
        CanaryLaunchTag.HEAD_K_SWA,
        CanaryLaunchTag.HEAD_V_SWA,
    )


def _is_tail_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.TAIL_K_FULL,
        CanaryLaunchTag.TAIL_V_FULL,
        CanaryLaunchTag.TAIL_K_SWA,
        CanaryLaunchTag.TAIL_V_SWA,
    )


def _is_sweep_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.SWEEP_K_FULL,
        CanaryLaunchTag.SWEEP_V_FULL,
        CanaryLaunchTag.SWEEP_K_SWA,
        CanaryLaunchTag.SWEEP_V_SWA,
    )


def _endpoint_belongs_to_group(
    endpoint: CanaryEndpoint, group: CanaryBufferGroup
) -> bool:
    suffix = endpoint.kernel_kind.name.rsplit("_", 1)[1]
    return suffix == group.kind.name
