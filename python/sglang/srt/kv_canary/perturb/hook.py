from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb import (
    real_kv_unused_cache,
    real_kv_used,
    req_to_token,
)
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.utils import WarmupGate
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PerturbHook:
    """Thin orchestrator that owns the per-canary-runner state (config, buffer groups,
    req-to-token pool, radix cache handle, warmup gate) and dispatches each per-forward
    invocation to the three perturb-point modules.

    Per-perturb logic lives in :mod:`sglang.srt.kv_canary.perturb.req_to_token`,
    :mod:`...perturb.real_kv_used`, :mod:`...perturb.real_kv_unused_cache`.
    """

    def __init__(
        self,
        *,
        config: PerturbConfig,
        req_to_token_pool: "ReqToTokenPool",
        buffer_groups: tuple[CanaryBufferGroup, ...],
        pump_and_allreduce: PumpAndAllreduce,
    ) -> None:
        self._config = config
        self._req_to_token_pool = req_to_token_pool
        self._buffer_groups = buffer_groups
        self._radix_cache: Optional["BasePrefixCache"] = None
        self._warmup_gate = WarmupGate(
            config=config, pump_and_allreduce=pump_and_allreduce
        )

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._radix_cache = radix_cache

    def perturb_req_to_token_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        req_to_token.run(
            forward_batch=forward_batch,
            config=self._config,
            req_to_token_pool=self._req_to_token_pool,
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_used_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        real_kv_used.run(
            forward_batch=forward_batch,
            config=self._config,
            req_to_token_pool=self._req_to_token_pool,
            buffer_groups=self._buffer_groups,
            warmup_gate=self._warmup_gate,
        )

    def perturb_real_kv_unused_cache_hook(
        self, forward_batch: Optional["ForwardBatch"]
    ) -> None:
        real_kv_unused_cache.run(
            forward_batch=forward_batch,
            config=self._config,
            buffer_groups=self._buffer_groups,
            radix_cache=self._radix_cache,
            warmup_gate=self._warmup_gate,
        )
