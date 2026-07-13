from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.mem_cache.kv_cache_configurator import (
    _get_dsv4_compress_state_dtypes,
    _InitializedPools,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_memory_pool import UnifiedPoolBundle
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

logger = logging.getLogger(__name__)


_is_npu = is_npu()
_is_hip = is_hip()


class ModelRunnerKVCacheMixin:
    def _profile_available_bytes(self: ModelRunner, pre_model_load_memory: int) -> int:
        return self.kv_cache_configurator._profile_available_bytes(
            pre_model_load_memory
        )

    def _handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
        return self.kv_cache_configurator._handle_max_mamba_cache(total_rest_memory)

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        return self.kv_cache_configurator._calculate_mamba_ratio()

    def _validate_prefill_only_disable_kv_cache_pool_family(
        self: ModelRunner,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        current_platform,
    ):
        return self.kv_cache_configurator._validate_prefill_only_disable_kv_cache_pool_family(
            is_dsa_model, is_dsv4_model, current_platform
        )

    def _init_unified_mamba_pools(
        self: ModelRunner, *, max_num_reqs: int, max_total_num_tokens: int
    ) -> UnifiedPoolBundle:
        return self.kv_cache_configurator._init_unified_mamba_pools(
            max_num_reqs=max_num_reqs, max_total_num_tokens=max_total_num_tokens
        )

    def _init_unified_swa_pools(
        self: ModelRunner,
        *,
        max_num_reqs: int,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ) -> UnifiedPoolBundle:
        return self.kv_cache_configurator._init_unified_swa_pools(
            max_num_reqs=max_num_reqs,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
        )

    def _init_pools(
        self: ModelRunner,
        *,
        max_total_num_tokens: int,
        max_running_requests: int,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
        c4_max_total_num_tokens: int,
        c128_max_total_num_tokens: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        c4_state_dtype: Optional[torch.dtype],
        c128_state_dtype: Optional[torch.dtype],
        req_to_token_pool: Optional[ReqToTokenPool],
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator],
    ) -> _InitializedPools:
        return self.kv_cache_configurator._init_pools(
            max_total_num_tokens=max_total_num_tokens,
            max_running_requests=max_running_requests,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
            c4_max_total_num_tokens=c4_max_total_num_tokens,
            c128_max_total_num_tokens=c128_max_total_num_tokens,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            c4_state_dtype=c4_state_dtype,
            c128_state_dtype=c128_state_dtype,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def _apply_token_constraints(self: ModelRunner, token_capacity: int) -> int:
        return self.kv_cache_configurator._apply_token_constraints(token_capacity)

    def resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        return self.kv_cache_configurator.resolve_max_num_reqs(token_capacity)

    def _apply_memory_pool_config(self: ModelRunner, config: MemoryPoolConfig):
        """Apply a resolved MemoryPoolConfig and initialize pools."""
        self.max_total_num_tokens = config.max_total_num_tokens
        self.max_running_requests = config.max_running_requests
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens

        # DSV4 compressed-attention pool sizes. Draft worker reuses target's
        # full/swa sizes but does NOT own c4/c128/state pools (those live on
        # the target rank only); zero them out regardless of what config holds.
        if self.is_draft_worker:
            c4_max_total_num_tokens = 0
            c128_max_total_num_tokens = 0
            c4_state_pool_size = 0
            c128_state_pool_size = 0
        else:
            c4_max_total_num_tokens = config.c4_max_total_num_tokens
            c128_max_total_num_tokens = config.c128_max_total_num_tokens
            c4_state_pool_size = config.c4_state_pool_size
            c128_state_pool_size = config.c128_state_pool_size

        # Draft worker does not own the compression-state pools, but keep the
        # dtype attributes initialized so _init_pools can share one code path.
        c4_state_dtype: Optional[torch.dtype] = None
        c128_state_dtype: Optional[torch.dtype] = None
        if is_deepseek_v4(self.model_config.hf_config):
            c4_state_dtype, c128_state_dtype = _get_dsv4_compress_state_dtypes()

        pools = self._init_pools(
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            full_max_total_num_tokens=self.full_max_total_num_tokens,
            swa_max_total_num_tokens=self.swa_max_total_num_tokens,
            c4_max_total_num_tokens=c4_max_total_num_tokens,
            c128_max_total_num_tokens=c128_max_total_num_tokens,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            c4_state_dtype=c4_state_dtype,
            c128_state_dtype=c128_state_dtype,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )
        self.req_to_token_pool = pools.req_to_token_pool
        self.token_to_kv_pool = pools.token_to_kv_pool
        self.token_to_kv_pool_allocator = pools.token_to_kv_pool_allocator
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = pools.unified_memory_pool

    def config_from_budget(
        self: ModelRunner, budget_bytes: int, *, cap_tokens: Optional[int] = None
    ) -> MemoryPoolConfig:
        return self.kv_cache_configurator.config_from_budget(
            budget_bytes, cap_tokens=cap_tokens
        )

    def _resolve_memory_pool_config(
        self: ModelRunner, pre_model_load_memory: int
    ) -> MemoryPoolConfig:
        return self.kv_cache_configurator._resolve_memory_pool_config(
            pre_model_load_memory
        )

    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
        else:
            self.memory_pool_config = self._resolve_memory_pool_config(
                pre_model_load_memory
            )

        self._apply_memory_pool_config(self.memory_pool_config)

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
