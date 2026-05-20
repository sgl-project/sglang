from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.d2h_pipeline import CanaryD2HPipeline
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.violation_state import CanaryDeviceState

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
        pump: PumpAndAllreduce,
        sweep: SweepOrchestrator,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._active_tags = active_tags
        self._pump = pump
        self._sweep = sweep
        self._d2h: CanaryD2HPipeline = CanaryD2HPipeline(device=device)
        self._previous_health_event: Optional[torch.cuda.Event] = None
        self._previous_stats_write_index_event: Optional[torch.cuda.Event] = None
        self._previous_stats_slot_sum_event: Optional[torch.cuda.Event] = None

    def health_check_step(self) -> None:
        step_counter = self._pump.step_counter
        if step_counter < _HEALTH_CHECK_WARMUP_STEPS:
            return
        if step_counter % _HEALTH_CHECK_EVERY_N_STEPS != 0:
            return
        if not self._active_tags:
            return

        device_state = self._device_state
        if self._previous_health_event is not None:
            CanaryD2HPipeline.wait(self._previous_health_event)
            counters = device_state.kernel_run_counters_host.tolist()
            zero_tags = [
                tag for tag in self._active_tags if int(counters[tag.value]) == 0
            ]
            if zero_tags:
                names = ", ".join(tag.name for tag in zero_tags)
                raise RuntimeError(
                    f"kv-canary: kernel_run_counter is zero after warmup for tags=[{names}] "
                    f"at step={step_counter}; canary path is not executing"
                )

        self._previous_health_event = self._d2h.stage(
            dst_host=device_state.kernel_run_counters_host,
            src_device=device_state.kernel_run_counters,
        )

    def print_periodic_stats(self) -> None:
        period = self._config.stats_print_every_n_steps
        if period <= 0:
            return
        step_counter = self._pump.step_counter
        if step_counter == 0 or step_counter % period != 0:
            return

        device_state = self._device_state
        if (
            self._previous_stats_slot_sum_event is not None
            and self._previous_stats_write_index_event is not None
        ):
            CanaryD2HPipeline.wait(self._previous_stats_slot_sum_event)
            CanaryD2HPipeline.wait(self._previous_stats_write_index_event)
            protected = int(device_state.slot_run_counters_sum_host.item())
            violations = int(device_state.violation_write_index_host.item())
            active = len(self._active_tags)
            logger.info(
                "[canary] step=%d protected_tokens=%d sweep_passes=%d violations=%d "
                "launch_tags_active=%d/%d",
                step_counter,
                protected,
                self._sweep.sweep_passes,
                violations,
                active,
                len(CanaryLaunchTag),
            )

        slot_sum_device = device_state.slot_run_counters.sum().view(1)
        self._previous_stats_slot_sum_event = self._d2h.stage(
            dst_host=device_state.slot_run_counters_sum_host,
            src_device=slot_sum_device,
        )
        self._previous_stats_write_index_event = self._d2h.stage(
            dst_host=device_state.violation_write_index_host,
            src_device=device_state.violation_log.violation_write_index,
        )
