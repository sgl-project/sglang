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
    """Thin orchestrator that owns the per-canary-runner state (config, buffer groups,
    req-to-token pool, radix cache handle, warmup gate) and dispatches each per-forward
    invocation to the four perturb-point modules.

    Per-perturb logic lives in :mod:`sglang.srt.kv_canary.perturb.req_to_token`,
    :mod:`...perturb.real_kv_used`, :mod:`...perturb.real_kv_unused_cache`,
    :mod:`...perturb.real_kv_post_forward`.
    """

    def __init__(
        self,
        *,
        config: PerturbConfig,
        req_to_token_pool: "ReqToTokenPool",
        buffer_groups: tuple[CanaryBufferGroup, ...],
        step_counter_getter: Callable[[], int],
        swa_window_size: int = 0,
        sweep_interval: int = 0,
    ) -> None:
        self._config = config
        self._req_to_token_pool = req_to_token_pool
        self._buffer_groups = buffer_groups
        self._step_counter_getter = step_counter_getter
        self._swa_window_size = swa_window_size
        self._sweep_interval = sweep_interval
        self._radix_cache: Optional["BasePrefixCache"] = None
        self._warmup_gate = WarmupGate(
            config=config, step_counter_getter=step_counter_getter
        )

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._radix_cache = radix_cache

    def perturb(
        self,
        *,
        maybe_non_mature_forward_batch: Optional["ForwardBatch"],
    ) -> None:
        """Phase 1 entrypoint. The argument name flags that only batch-level
        fields (active req set, base out_cache_loc) are safe to consume —
        step-specific ForwardBatch fields may not yet be filled when an
        EAGLE outer cycle calls this before any inner draft step has run."""
        self.perturb_req_to_token(maybe_non_mature_forward_batch)
        self.perturb_real_kv_used(maybe_non_mature_forward_batch)
        self.perturb_real_kv_unused_cache(maybe_non_mature_forward_batch)

    def perturb_post_forward(
        self,
        *,
        maybe_non_mature_forward_batch: Optional["ForwardBatch"],
    ) -> None:
        """Phase 4 entrypoint. Fires the real_kv_post_forward perturb against
        the same ForwardBatch the outer cycle handed in at phase 1 (possibly
        mutated by inner forwards in between). Tail-after ordering means
        this MUST fire after the captured forward writes have completed."""
        self.perturb_real_kv_post_forward(maybe_non_mature_forward_batch)

    def perturb_req_to_token(self, forward_batch: Optional["ForwardBatch"]) -> None:
        req_to_token.run(
            forward_batch=forward_batch,
            config=self._config,
            req_to_token_pool=self._req_to_token_pool,
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_used(self, forward_batch: Optional["ForwardBatch"]) -> None:
        real_kv_used.run(
            forward_batch=forward_batch,
            config=self._config,
            req_to_token_pool=self._req_to_token_pool,
            buffer_groups=self._buffer_groups,
            swa_window_size=self._swa_window_size,
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_unused_cache(
        self, forward_batch: Optional["ForwardBatch"]
    ) -> None:
        real_kv_unused_cache.run(
            forward_batch=forward_batch,
            config=self._config,
            buffer_groups=self._buffer_groups,
            radix_cache=self._radix_cache,
            swa_window_size=self._swa_window_size,
            sweep_interval=self._sweep_interval,
            step_counter=self._step_counter_getter(),
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_post_forward(
        self, forward_batch: Optional["ForwardBatch"]
    ) -> None:
        real_kv_post_forward.run(
            forward_batch=forward_batch,
            config=self._config,
            buffer_groups=self._buffer_groups,
            warmup_gate=self._warmup_gate,
        )
