from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.state import CanaryDeviceState

logger = logging.getLogger(__name__)

_HEALTH_CHECK_EVERY_N_STEPS: int = 100
_HEALTH_CHECK_WARMUP_STEPS: int = 100
_SWEEP_TAGS: frozenset[CanaryLaunchTag] = frozenset(
    (
        CanaryLaunchTag.SWEEP_K_FULL,
        CanaryLaunchTag.SWEEP_V_FULL,
        CanaryLaunchTag.SWEEP_K_SWA,
        CanaryLaunchTag.SWEEP_V_SWA,
    )
)


class KernelRunCounterHealthChecker:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
        active_tags: tuple[CanaryLaunchTag, ...],
        outer_step_counter_getter: Callable[[], int],
        d2h_stream: torch.cuda.Stream,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._active_tags = active_tags
        self._outer_step_counter_getter = outer_step_counter_getter
        self._handler = DelayedDeviceHostHandler(d2h_stream=d2h_stream)

    def step(self) -> None:
        self._handler.step(
            compute_on_device=self._compute_on_device,
            postprocess_on_host=self._postprocess_on_host,
        )

    def _compute_on_device(self) -> Optional[torch.Tensor]:
        outer_step_counter = self._outer_step_counter_getter()
        if outer_step_counter < _HEALTH_CHECK_WARMUP_STEPS:
            return None
        if outer_step_counter % _HEALTH_CHECK_EVERY_N_STEPS != 0:
            return None
        if not self._active_tags:
            return None
        return self._device_state.kernel_run_counters

    def _postprocess_on_host(self, host_tensor: torch.Tensor) -> None:
        counters = host_tensor.tolist()
        expected_tags = self._expected_active_tags_for_health_check()
        zero_tags = [tag for tag in expected_tags if int(counters[tag.value]) == 0]
        if zero_tags:
            names = ", ".join(tag.name for tag in zero_tags)
            raise RuntimeError(
                f"kv-canary: kernel_run_counter is zero after warmup for tags=[{names}] "
                f"at step={self._outer_step_counter_getter()}; canary path is not executing"
            )

    def _expected_active_tags_for_health_check(self) -> tuple[CanaryLaunchTag, ...]:
        if self._config.sweep_interval > 0:
            return self._active_tags
        return tuple(tag for tag in self._active_tags if tag not in _SWEEP_TAGS)


class PeriodicCanaryStatsLogger:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
        active_tags: tuple[CanaryLaunchTag, ...],
        outer_step_counter_getter: Callable[[], int],
        sweep_orchestrator: SweepOrchestrator,
        d2h_stream: torch.cuda.Stream,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._active_tags = active_tags
        self._outer_step_counter_getter = outer_step_counter_getter
        self._sweep_orchestrator = sweep_orchestrator
        self._handler = DelayedDeviceHostHandler(d2h_stream=d2h_stream)

    def step(self) -> None:
        self._handler.step(
            compute_on_device=self._compute_on_device,
            postprocess_on_host=self._postprocess_on_host,
        )

    def _compute_on_device(self) -> Optional[dict[str, Any]]:
        period = self._config.stats_print_every_n_steps
        if period <= 0:
            return None
        outer_step_counter = self._outer_step_counter_getter()
        if outer_step_counter == 0 or outer_step_counter % period != 0:
            return None
        device_state = self._device_state
        return {
            "step": outer_step_counter,
            "slot_sum": device_state.slot_run_counters.sum().view(1),
            "write_index": device_state.violation_log.violation_write_index,
        }

    def _postprocess_on_host(self, host_data: dict[str, Any]) -> None:
        logger.info(
            "[canary] step=%d protected_tokens=%d sweep_passes=%d violations=%d "
            "launch_tags_active=%d/%d",
            int(host_data["step"]),
            int(host_data["slot_sum"].item()),
            self._sweep_orchestrator.sweep_passes,
            int(host_data["write_index"].item()),
            len(self._active_tags),
            len(CanaryLaunchTag),
        )
