from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_canary.runner.kernel_launcher import launch_endpoints_sweep
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.sweep_plan_builder import build_verify_plan_radix_sweep

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


class SweepOrchestrator:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        endpoints: tuple[CanaryEndpoint, ...],
        swa_window_size: int,
        outer_step_counter_getter: Callable[[], int],
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._buffer_groups = buffer_groups
        self._endpoints = endpoints
        self._swa_window_size = swa_window_size
        self._outer_step_counter_getter = outer_step_counter_getter
        self._radix_cache: Optional[BasePrefixCache] = None

        self._last_sweep_step: int = -1
        self._sweep_passes: int = 0

    @property
    def sweep_passes(self) -> int:
        return self._sweep_passes

    def attach_radix_cache(self, radix_cache: BasePrefixCache) -> None:
        self._radix_cache = radix_cache

    def maybe_run_sweep(self) -> None:
        if self._config.sweep_interval == 0:
            return
        outer_step_counter = self._outer_step_counter_getter()
        if (
            self._last_sweep_step >= 0
            and outer_step_counter - self._last_sweep_step < self._config.sweep_interval
        ):
            return
        self._last_sweep_step = outer_step_counter

        if self._radix_cache is None:
            return

        violation_log = self._device_state.violation_log
        for group in self._buffer_groups:
            window = self._swa_window_size if group.kind is PoolKind.SWA else 0
            verify_plan = build_verify_plan_radix_sweep(
                radix_cache=self._radix_cache,
                swa_window_size=window,
                full_to_swa_index_mapping=group.swa_index_lut,
            )
            launch_endpoints_sweep(
                endpoints=self._endpoints,
                group=group,
                verify_plan=verify_plan,
                violation_log=violation_log,
                real_kv_hash_mode=self._config.real_kv_hash_mode,
            )

        self._sweep_passes += 1
        logger.info(
            "[canary] sweep succeeded %d times (last_step=%d)",
            self._sweep_passes,
            outer_step_counter,
        )
