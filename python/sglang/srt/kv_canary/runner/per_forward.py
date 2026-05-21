from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.perturb.manager import PerturbManager
from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.srt.kv_canary.runner.enable_warner import _CanaryEnableWarner
from sglang.srt.kv_canary.runner.kernel_launch import (
    invoke_plan,
    launch_endpoints_per_forward,
)
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager

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
      monkey-patched model.forward, before the original forward): plan sub-kernels + HEAD endpoint launches.
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
        perturb_manager: PerturbManager,
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        per_forward_write_entry_capacity: int,
        d2h_stream: Optional[torch.cuda.Stream] = None,
        token_oracle_manager: Optional[TokenOracleManager] = None,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._buffer_groups = buffer_groups
        self._endpoints = endpoints
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._perturb_manager = perturb_manager
        self._d2h_stream = d2h_stream
        self._token_oracle_manager: Optional[TokenOracleManager] = token_oracle_manager

        self._verify_plan_per_forward = VerifyPlan.allocate(
            verify_capacity=per_forward_verify_capacity, device=device
        )
        write_req_capacity = per_forward_write_req_capacity
        self._write_plan_per_forward = WritePlan.allocate(
            write_req_capacity=write_req_capacity, device=device
        )

        write_entry_capacity = per_forward_write_entry_capacity
        self._expected_inputs = ExpectedInputs.allocate(
            capacity=write_entry_capacity, device=device
        )

        self._plan_input_per_forward = PlanInput.allocate(
            bs_capacity=write_req_capacity,
            device=device,
        )

        self._write_req_capacity = write_req_capacity
        self._write_entry_capacity = write_entry_capacity
        self._verify_capacity = per_forward_verify_capacity

        self._enable_warner = _CanaryEnableWarner(
            verify_capacity=self._verify_capacity,
            d2h_stream=d2h_stream,
        )

        self._last_forward_batch: Optional["ForwardBatch"] = None

    def before_forward(self, forward_batch: "ForwardBatch") -> None:
        if self._config.mode == "off":
            return

        self._last_forward_batch = forward_batch

        bs = int(forward_batch.batch_size)
        num_tokens = int(forward_batch.positions.shape[0])
        if bs > self._write_req_capacity:
            raise RuntimeError(
                f"kv-canary: forward_batch.batch_size={bs} exceeds pre-allocated "
                f"write_req_capacity={self._write_req_capacity}; raise --cuda-graph-max-bs "
                f"or check CanaryLaunchCapacities.from_args"
            )
        if num_tokens > self._write_entry_capacity:
            raise RuntimeError(
                f"kv-canary: forward_batch token count={num_tokens} exceeds pre-allocated "
                f"write_entry_capacity={self._write_entry_capacity}; raise "
                f"--chunked-prefill-size / --max-prefill-tokens or check "
                f"CanaryLaunchCapacities.from_args"
            )

        self._perturb_manager.perturb(forward_batch)

        if self._config.input_check_mode:
            manager = self._token_oracle_manager
            if manager is None:
                raise RuntimeError(
                    "kv-canary: input_check_mode=True requires a TokenOracleManager; pass "
                    "token_oracle_manager=install_oracle_sampler(oracle=...) into "
                    "install_canary(...)"
                )
            manager.fill_expected_inputs(
                forward_batch=forward_batch,
                expected_inputs_out=self._expected_inputs,
            )

        self._plan_input_per_forward.fill_from_forward_batch(
            forward_batch=forward_batch,
        )

    def launch_head_kernels(self, forward_batch: "ForwardBatch") -> None:
        if self._config.mode == "off":
            return

        violation_log = self._device_state.violation_log
        num_tokens = int(forward_batch.positions.shape[0])
        expected_inputs_slice = self._expected_inputs.slice(num_tokens)
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
                expected_inputs=expected_inputs_slice,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=self._config.input_check_mode,
            )

    def launch_tail_kernels(self, forward_batch: "ForwardBatch") -> None:
        if self._config.mode == "off":
            return

        violation_log = self._device_state.violation_log
        num_tokens = int(forward_batch.positions.shape[0])
        expected_inputs_slice = self._expected_inputs.slice(num_tokens)
        for group in self._buffer_groups:
            launch_endpoints_per_forward(
                endpoints=self._endpoints,
                group=group,
                tag_filter=_is_tail_tag,
                verify_plan=self._verify_plan_per_forward,
                write_plan=self._write_plan_per_forward,
                forward_batch=forward_batch,
                expected_inputs=expected_inputs_slice,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
                input_check_mode=self._config.input_check_mode,
            )

    def end_of_step(self) -> None:
        if self._config.mode == "off":
            return
        if self._last_forward_batch is not None:
            self._perturb_manager.end_of_forward(self._last_forward_batch)
        self._enable_warner.tick(self._verify_plan_per_forward.enable)


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
