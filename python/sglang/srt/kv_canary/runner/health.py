from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.d2h_slot import DelayedD2HReadSlot
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.state import CanaryDeviceState, CanaryHostState

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
        host_state: CanaryHostState,
        active_tags: tuple[CanaryLaunchTag, ...],
        pump_and_allreduce: PumpAndAllreduce,
        sweep_orchestrator: SweepOrchestrator,
        d2h_stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._host_state = host_state
        self._active_tags = active_tags
        self._pump_and_allreduce = pump_and_allreduce
        self._sweep_orchestrator = sweep_orchestrator
        self._health_slot: DelayedD2HReadSlot = DelayedD2HReadSlot(
            host=host_state.kernel_run_counters_host,
            stream=d2h_stream,
        )
        self._stats_slot_sum_slot: DelayedD2HReadSlot = DelayedD2HReadSlot(
            host=host_state.slot_run_counters_sum_host,
            stream=d2h_stream,
        )
        self._stats_write_index_slot: DelayedD2HReadSlot = DelayedD2HReadSlot(
            host=host_state.violation_write_index_host,
            stream=d2h_stream,
        )

    def health_check_step(self) -> None:
        step_counter = self._pump_and_allreduce.step_counter
        if step_counter < _HEALTH_CHECK_WARMUP_STEPS:
            return
        if step_counter % _HEALTH_CHECK_EVERY_N_STEPS != 0:
            return
        if not self._active_tags:
            return

        device_state = self._device_state
        prev_counters = self._health_slot.pop()
        if prev_counters is not None:
            counters = prev_counters.tolist()
            zero_tags = [
                tag for tag in self._active_tags if int(counters[tag.value]) == 0
            ]
            if zero_tags:
                names = ", ".join(tag.name for tag in zero_tags)
                raise RuntimeError(
                    f"kv-canary: kernel_run_counter is zero after warmup for tags=[{names}] "
                    f"at step={step_counter}; canary path is not executing"
                )

        self._health_slot.stage(src_device=device_state.kernel_run_counters)

    def print_periodic_stats(self) -> None:
        period = self._config.stats_print_every_n_steps
        if period <= 0:
            return
        step_counter = self._pump_and_allreduce.step_counter
        if step_counter == 0 or step_counter % period != 0:
            return

        device_state = self._device_state
        prev_slot_sum = self._stats_slot_sum_slot.pop()
        prev_write_index = self._stats_write_index_slot.pop()
        if prev_slot_sum is not None and prev_write_index is not None:
            protected = int(prev_slot_sum.item())
            violations = int(prev_write_index.item())
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
        self._stats_slot_sum_slot.stage(src_device=slot_sum_device)
        self._stats_write_index_slot.stage(
            src_device=device_state.violation_log.violation_write_index,
        )
