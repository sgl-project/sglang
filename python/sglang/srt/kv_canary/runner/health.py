from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor, stage_d2h_future
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.state import CanaryDeviceState

logger = logging.getLogger("sglang.srt.kv_canary.runner.canary_runner")

_HEALTH_CHECK_EVERY_N_STEPS: int = 1000
_HEALTH_CHECK_WARMUP_STEPS: int = 100


class HealthAndStats:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device: torch.device,
        device_state: CanaryDeviceState,
        active_tags: tuple[CanaryLaunchTag, ...],
        pump_and_allreduce: PumpAndAllreduce,
        sweep_orchestrator: SweepOrchestrator,
        d2h_stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._active_tags = active_tags
        self._pump_and_allreduce = pump_and_allreduce
        self._sweep_orchestrator = sweep_orchestrator
        self._d2h_stream = d2h_stream
        self._previous_health_future: Optional[FutureTensor] = None
        self._previous_slot_sum_future: Optional[FutureTensor] = None
        self._previous_write_index_future: Optional[FutureTensor] = None

    def health_check_step(self) -> None:
        step_counter = self._pump_and_allreduce.step_counter
        if step_counter < _HEALTH_CHECK_WARMUP_STEPS:
            return
        if step_counter % _HEALTH_CHECK_EVERY_N_STEPS != 0:
            return
        if not self._active_tags:
            return

        device_state = self._device_state
        if self._previous_health_future is not None:
            counters = self._previous_health_future.wait().tolist()
            zero_tags = [
                tag for tag in self._active_tags if int(counters[tag.value]) == 0
            ]
            if zero_tags:
                names = ", ".join(tag.name for tag in zero_tags)
                raise RuntimeError(
                    f"kv-canary: kernel_run_counter is zero after warmup for tags=[{names}] "
                    f"at step={step_counter}; canary path is not executing"
                )

        self._previous_health_future = stage_d2h_future(
            src_device=device_state.kernel_run_counters, stream=self._d2h_stream
        )

    def print_periodic_stats(self) -> None:
        period = self._config.stats_print_every_n_steps
        if period <= 0:
            return
        step_counter = self._pump_and_allreduce.step_counter
        if step_counter == 0 or step_counter % period != 0:
            return

        device_state = self._device_state
        prev_slot_sum = self._previous_slot_sum_future
        prev_write_index = self._previous_write_index_future
        if prev_slot_sum is not None and prev_write_index is not None:
            protected = int(prev_slot_sum.wait().item())
            violations = int(prev_write_index.wait().item())
            active = len(self._active_tags)
            logger.info(
                "[canary] step=%d protected_tokens=%d sweep_passes=%d violations=%d "
                "launch_tags_active=%d/%d",
                step_counter,
                protected,
                self._sweep_orchestrator.sweep_passes,
                violations,
                active,
                len(CanaryLaunchTag),
            )

        slot_sum_device = device_state.slot_run_counters.sum().view(1)
        self._previous_slot_sum_future = stage_d2h_future(
            src_device=slot_sum_device, stream=self._d2h_stream
        )
        self._previous_write_index_future = stage_d2h_future(
            src_device=device_state.violation_log.violation_write_index,
            stream=self._d2h_stream,
        )
