from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb import (
    real_kv_post_forward,
    real_kv_unused_cache,
    real_kv_used,
    req_to_token,
)
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.utils import WarmupGate

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PerturbManager:
    def __init__(
        self,
        *,
        config: PerturbConfig,
        req_to_token_pool: "ReqToTokenPool",
        buffer_groups: tuple[CanaryBufferGroup, ...],
        outer_step_counter_getter: Callable[[], int],
        swa_window_size: int = 0,
        sweep_interval: int = 0,
    ) -> None:
        self._config = config
        self._req_to_token_pool = req_to_token_pool
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
        self.perturb_req_to_token(maybe_inaccurate_forward_batch)
        self.perturb_real_kv_used(maybe_inaccurate_forward_batch)
        self.perturb_real_kv_unused_cache(maybe_inaccurate_forward_batch)

    def perturb_post_forward(
        self,
        *,
        maybe_inaccurate_forward_batch: Optional["ForwardBatch"],
    ) -> None:
        self.perturb_real_kv_post_forward(maybe_inaccurate_forward_batch)

    def perturb_req_to_token(
        self, maybe_inaccurate_forward_batch: Optional["ForwardBatch"]
    ) -> None:
        req_to_token.run(
            maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            config=self._config,
            req_to_token_pool=self._req_to_token_pool,
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_used(
        self, maybe_inaccurate_forward_batch: Optional["ForwardBatch"]
    ) -> None:
        real_kv_used.run(
            maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            config=self._config,
            req_to_token_pool=self._req_to_token_pool,
            buffer_groups=self._buffer_groups,
            swa_window_size=self._swa_window_size,
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_unused_cache(
        self, maybe_inaccurate_forward_batch: Optional["ForwardBatch"]
    ) -> None:
        real_kv_unused_cache.run(
            maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            config=self._config,
            buffer_groups=self._buffer_groups,
            radix_cache=self._radix_cache,
            swa_window_size=self._swa_window_size,
            sweep_interval=self._sweep_interval,
            outer_step_counter=self._outer_step_counter_getter(),
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_post_forward(
        self, maybe_inaccurate_forward_batch: Optional["ForwardBatch"]
    ) -> None:
        real_kv_post_forward.run(
            maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            config=self._config,
            buffer_groups=self._buffer_groups,
            warmup_gate=self._warmup_gate,
        )
