from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import CanaryPseudoMode as CanaryInputCheckMode
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.mock_model.sampler import OracleSamplerHook
from sglang.srt.kv_canary.plan_input import PlanInput, fill_plan_input_per_forward
from sglang.srt.kv_canary.runner.launch import (
    invoke_plan,
    launch_endpoints_per_forward,
)
from sglang.srt.kv_canary.runner.perturb import PerturbHook
from sglang.srt.kv_canary.state import CanaryDeviceState

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PerForwardOrchestrator:
    """Per-forward orchestrator. Split into three phases tightly aligned with the cuda-graph
    capture boundary:

    - ``before_forward(forward_batch)`` runs HOST-SIDE (called by ModelRunner.forward outside the
      captured region): perturb hooks, fill the static expected_input buffers, fill the static
      per-forward PlanInput buffers.
    - ``launch_head_kernels(forward_batch)`` runs INSIDE the captured region (called by the
      monkey-patched model.forward, before the original forward): canary_plan_step kernel +
      HEAD endpoint launches.
    - ``launch_tail_kernels(forward_batch)`` runs INSIDE the captured region (called by the
      monkey-patched model.forward, after the original forward): TAIL endpoint launches reusing
      the plan staged in launch_head_kernels.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        device: torch.device,
        device_state: CanaryDeviceState,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        endpoints: tuple[CanaryEndpoint, ...],
        req_to_token_pool: "ReqToTokenPool",
        swa_window_size: int,
        perturb_hook: PerturbHook,
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        per_forward_write_entry_capacity: int,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._buffer_groups = buffer_groups
        self._endpoints = endpoints
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._perturb_hook = perturb_hook
        self._oracle_sampler_hook: Optional[OracleSamplerHook] = None

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

        bs_capacity = max(1, per_forward_write_req_capacity)
        self._plan_input_per_forward = PlanInput(
            fb_req_pool_indices=torch.zeros(
                bs_capacity, dtype=torch.int64, device=device
            ),
            fb_prefix_lens=torch.zeros(bs_capacity, dtype=torch.int32, device=device),
            fb_extend_seq_lens=torch.zeros(
                bs_capacity, dtype=torch.int32, device=device
            ),
            extra_verify_slot_indices=torch.zeros(0, dtype=torch.int32, device=device),
            extra_verify_positions=torch.zeros(0, dtype=torch.int32, device=device),
            extra_verify_prev_slot_indices=torch.zeros(
                0, dtype=torch.int32, device=device
            ),
            extra_verify_num_valid=torch.zeros(1, dtype=torch.int32, device=device),
        )

        self._last_forward_batch: Optional["ForwardBatch"] = None

    def attach_oracle_sampler_hook(self, hook: OracleSamplerHook) -> None:
        self._oracle_sampler_hook = hook

    def before_forward(self, forward_batch: "ForwardBatch") -> None:
        if self._config.mode == "off":
            return

        self._perturb_hook.perturb_hook(forward_batch)
        self._perturb_hook.perturb_real_kv_hook(forward_batch)
        self._last_forward_batch = forward_batch

        if self._config.input_check_mode == CanaryInputCheckMode.ON:
            hook = self._oracle_sampler_hook
            if hook is None:
                raise RuntimeError(
                    "kv-canary: input_check_mode=ON requires an OracleSamplerHook; call "
                    "CanaryRunner.attach_oracle_sampler_hook(hook) where hook is the return "
                    "value of install_oracle_sampler(oracle=...)"
                )
            hook.fill_expected_inputs(
                forward_batch=forward_batch,
                expected_input_tokens_out=self._expected_input_tokens,
                expected_input_positions_out=self._expected_input_positions,
            )

        fill_plan_input_per_forward(
            forward_batch=forward_batch,
            plan_input_out=self._plan_input_per_forward,
        )

    def launch_head_kernels(self, forward_batch: "ForwardBatch") -> None:
        if self._config.mode == "off":
            return

        violation_log = self._device_state.violation_log
        for group in self._buffer_groups:
            invoke_plan(
                plan_input=self._plan_input_per_forward,
                verify_plan=self._verify_plan_per_forward,
                write_plan=self._write_plan_per_forward,
                group=group,
                req_to_token=self._req_to_token_pool.req_to_token,
                swa_window_size=self._swa_window_size,
            )
            launch_endpoints_per_forward(
                endpoints=self._endpoints,
                group=group,
                tag_filter=_is_head_tag,
                verify_plan=self._verify_plan_per_forward,
                write_plan=self._write_plan_per_forward,
                forward_batch=forward_batch,
                expected_input_tokens=self._expected_input_tokens,
                expected_input_positions=self._expected_input_positions,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=self._config.input_check_mode,
            )

    def launch_tail_kernels(self, forward_batch: "ForwardBatch") -> None:
        if self._config.mode == "off":
            return

        violation_log = self._device_state.violation_log
        for group in self._buffer_groups:
            launch_endpoints_per_forward(
                endpoints=self._endpoints,
                group=group,
                tag_filter=_is_tail_tag,
                verify_plan=self._verify_plan_per_forward,
                write_plan=self._write_plan_per_forward,
                forward_batch=forward_batch,
                expected_input_tokens=self._expected_input_tokens,
                expected_input_positions=self._expected_input_positions,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=self._config.input_check_mode,
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
