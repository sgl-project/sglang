from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.utils import WarmupGate

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PerturbManager:
    def __init__(
        self,
        *,
        config: PerturbConfig,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        outer_step_counter_getter: Callable[[], int],
        swa_window_size: int = 0,
        sweep_interval: int = 0,
    ) -> None:
        self._config = config
        self._buffer_groups = buffer_groups
        self._outer_step_counter_getter = outer_step_counter_getter
        self._swa_window_size = swa_window_size
        self._sweep_interval = sweep_interval
        self._radix_cache: Optional["BasePrefixCache"] = None
        self._warmup_gate = WarmupGate(
            config=config, outer_step_counter_getter=outer_step_counter_getter
        )

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._radix_cache = radix_cache

    def perturb(
        self,
        *,
        maybe_inaccurate_forward_batch: Optional["ForwardBatch"],
    ) -> None:
        pass
