from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig, is_deepseek_nsa
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, NSATokenToKVPool, ReqToTokenPool
from sglang.srt.utils.common import is_hip

_is_hip = is_hip()
import logging

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_v4
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hisparse_memory_pool import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_npu,
)

logger = logging.getLogger(__name__)

_is_npu = is_npu()

# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner_components.pool_configurator import (
        MemoryPoolConfig,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheConfigResult:
    """Configurator output — caller writes back to ModelRunner fields."""

    max_total_num_tokens: int
    max_running_requests: int
    full_max_total_num_tokens: Optional[int]
    swa_max_total_num_tokens: Optional[int]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    memory_pool_config: "MemoryPoolConfig"


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheConfigurator:
    """KV cache pipeline (profile -> resolve -> constrain -> init pools).

    Replaces ``ModelRunnerKVCacheMixin`` via composition. ``frozen=True``
    blocks any stale ``self.X = Y`` writes left over from the mixin
    migration; ``slots=True`` blocks attribute typos at runtime;
    ``kw_only=True`` forces named-kwargs construction at the caller.

    Pipeline intermediate state (profiled bytes / resolved configs / pool
    objects) flows through local variables + return values, not via
    attribute writes on ``self``.
    """

    # deployment env
    device: str
    gpu_id: int
    mem_fraction_static: float
    page_size: int
    # parallel rank / size
    tp_rank: int
    tp_size: int
    pp_size: int
    dp_size: int
    attention_tp_size: int
    # model / dtype
    model_config: ModelConfig
    server_args: ServerArgs
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    # speculative decoding
    spec_algorithm: SpeculativeAlgorithm
    is_draft_worker: bool
    # DFLASH-only: target's `cell_size` is scaled to include draft KV cache.
    # ``pool_configurator.DefaultPoolConfigurator`` reads this off the
    # configurator (was ``getattr(mr, "dflash_draft_num_layers", None)`` in
    # the mixin era — silent ``None`` if missing). Must be plumbed through;
    # otherwise the target KV pool oversizes by 1+ GB on 32GB GPUs and
    # OOMs at cuda graph capture (see debug_journal 2026-05-11-kvc-...).
    dflash_draft_num_layers: Optional[int]
    # arch flags
    is_hybrid_swa: bool
    is_hybrid_swa_compress: bool
    use_mla_backend: bool
    enable_hisparse: bool
    mambaish_config: Optional[Any]
    hybrid_gdn_config: Optional[Any]
    # PP slice
    start_layer: int
    end_layer: int
    num_effective_layers: int
    # optional pre-injection (draft worker reuses target's pool)
    req_to_token_pool: Optional[ReqToTokenPool]
    token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator]
    # draft worker budget
    memory_pool_config: Optional["MemoryPoolConfig"]

    def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:
        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
            config = self.memory_pool_config
        else:
            config = self._resolve_memory_pool_config(pre_model_load_memory)

        max_total_num_tokens = config.max_total_num_tokens
        max_running_requests = config.max_running_requests
        full_max_total_num_tokens = None
        swa_max_total_num_tokens = None
        if self.is_hybrid_swa:
            full_max_total_num_tokens = config.full_max_total_num_tokens
            swa_max_total_num_tokens = config.swa_max_total_num_tokens

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

        # state_dtype is a DSV4 architectural constant (fp32 for c4/c128
        # state buffers); set unconditionally so draft workers have it before
        # _init_pools reads it (target path also overwrites this in the
        # configurator's resolve() for parity, harmless here).
        state_dtype: Optional[torch.dtype] = None
        if is_deepseek_v4(self.model_config.hf_config):
            state_dtype = torch.float32

        req_to_token_pool, token_to_kv_pool, token_to_kv_pool_allocator = (
            self._init_pools(
                max_total_num_tokens=max_total_num_tokens,
                max_running_requests=max_running_requests,
                full_max_total_num_tokens=full_max_total_num_tokens,
                swa_max_total_num_tokens=swa_max_total_num_tokens,
                c4_max_total_num_tokens=c4_max_total_num_tokens,
                c128_max_total_num_tokens=c128_max_total_num_tokens,
                c4_state_pool_size=c4_state_pool_size,
                c128_state_pool_size=c128_state_pool_size,
                state_dtype=state_dtype,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )
        )

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return KVCacheConfigResult(
            max_total_num_tokens=max_total_num_tokens,
            max_running_requests=max_running_requests,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            memory_pool_config=config,
        )

    def _init_pools(
        self,
        *,
        max_total_num_tokens: int,
        max_running_requests: int,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
        c4_max_total_num_tokens: int,
        c128_max_total_num_tokens: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        state_dtype: Optional[torch.dtype],
        req_to_token_pool: Optional[ReqToTokenPool],
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator],
    ) -> tuple[ReqToTokenPool, KVCache, BaseTokenToKVPoolAllocator]:
        """Initialize the memory pools."""
        token_to_kv_pool = None
        max_num_reqs = max_running_requests

        # Initialize req_to_token_pool
        if req_to_token_pool is None:
            # FIXME(lsyin): this is the temporary fix for the context length issue when using speculative decoding
            extra_max_context_len = 4
            if self.server_args.speculative_num_draft_tokens is not None:
                extra_max_context_len += self.server_args.speculative_num_draft_tokens

            if self.server_args.disaggregation_mode == "decode":
                # subscribe memory for pre-allocated requests
                # if max_num_reqs <= 32, we pre-allocate 2x requests

                pre_alloc_size = envs.SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS.get()
                pre_alloc_size = (
                    max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size
                )
                if self.mambaish_config:
                    req_to_token_pool = self._hybrid_mamba_decode_req_pool(
                        max_num_reqs=max_num_reqs,
                        extra_max_context_len=extra_max_context_len,
                        pre_alloc_size=pre_alloc_size,
                    )
                else:
                    req_to_token_pool = self._decode_req_pool(
                        max_num_reqs=max_num_reqs,
                        extra_max_context_len=extra_max_context_len,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif self.mambaish_config:
                req_to_token_pool = self._hybrid_req_pool(
                    max_num_reqs=max_num_reqs,
                    extra_max_context_len=extra_max_context_len,
                )
            else:
                req_to_token_pool = self._default_req_pool(
                    max_num_reqs=max_num_reqs,
                    extra_max_context_len=extra_max_context_len,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # Initialize token_to_kv_pool
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        token_to_kv_pool = self._build_token_to_kv_pool(
            max_total_num_tokens=max_total_num_tokens,
            max_running_requests=max_running_requests,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
            c4_max_total_num_tokens=c4_max_total_num_tokens,
            c128_max_total_num_tokens=c128_max_total_num_tokens,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            state_dtype=state_dtype,
            is_nsa_model=is_nsa_model,
            is_dsv4_model=is_dsv4_model,
            req_to_token_pool=req_to_token_pool,
        )

        # Initialize token_to_kv_pool_allocator
        from sglang.srt.platforms import current_platform

        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if token_to_kv_pool_allocator is None:
            if current_platform.is_out_of_tree():
                AllocatorCls = current_platform.get_paged_allocator_cls()
                token_to_kv_pool_allocator = AllocatorCls(
                    max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=token_to_kv_pool,
                    need_sort=need_sort,
                )
            elif _is_npu and (
                self.server_args.attention_backend == "ascend"
                or self.hybrid_gdn_config is not None
            ):
                if self.is_hybrid_swa:
                    token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        full_max_total_num_tokens,
                        swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    from sglang.srt.hardware_backend.npu.allocator_npu import (
                        NPUPagedTokenToKVPoolAllocator,
                    )

                    token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                        max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if self.is_hybrid_swa:
                    token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        full_max_total_num_tokens,
                        swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        token_to_kv_pool_allocator = HiSparseTokenToKVPoolAllocator(
                            max_total_num_tokens,
                            page_size=self.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                            host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                        )
                    elif self.page_size == 1:
                        token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            max_total_num_tokens,
                            page_size=self.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                        )

            if self.enable_hisparse and is_dsv4_model:
                assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
                token_to_kv_pool_allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(
                    token_to_kv_pool_allocator
                )

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                swa_allocator = getattr(
                    token_to_kv_pool_allocator,
                    "logical_attn_allocator",
                    token_to_kv_pool_allocator,
                )
                assert swa_allocator.__class__ == SWATokenToKVPoolAllocator
                token_to_kv_pool.full_to_swa_index_mapping = (
                    swa_allocator.full_to_swa_index_mapping
                )
        return req_to_token_pool, token_to_kv_pool, token_to_kv_pool_allocator

    def _hybrid_mamba_decode_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
        pre_alloc_size: int,
    ) -> ReqToTokenPool:
        from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool

        return HybridMambaDecodeReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            cache_params=self.mambaish_config.mamba2_cache_params,
            mamba_layer_ids=(
                [
                    i
                    for i in self.mambaish_config.mamba2_cache_params.layers
                    if self.start_layer <= i < self.end_layer
                ]
            ),
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
            pre_alloc_size=pre_alloc_size,
            enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
            mamba_size=self.server_args.max_mamba_cache_size,
            start_layer=self.start_layer,
        )

    def _decode_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
        pre_alloc_size: int,
    ) -> ReqToTokenPool:
        from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

        return DecodeReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

    def _hybrid_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
    ) -> ReqToTokenPool:
        return HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=self.server_args.max_mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            cache_params=self.mambaish_config.mamba2_cache_params,
            mamba_layer_ids=(
                [
                    i
                    for i in self.mambaish_config.mamba2_cache_params.layers
                    if self.start_layer <= i < self.end_layer
                ]
            ),
            enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
            start_layer=self.start_layer,
        )

    def _default_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
    ) -> ReqToTokenPool:
        return ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
        )

    def _build_token_to_kv_pool(
        self,
        *,
        max_total_num_tokens: int,
        max_running_requests: int,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
        c4_max_total_num_tokens: int,
        c128_max_total_num_tokens: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        state_dtype: Optional[torch.dtype],
        is_nsa_model: bool,
        is_dsv4_model: bool,
        req_to_token_pool: ReqToTokenPool,
    ) -> KVCache:
        # Out-of-tree platform plugin system — used by elif below
        from sglang.srt.platforms import current_platform

        if is_dsv4_model:
            token_to_kv_pool = self._dsv4_kv_pool(
                max_running_requests=max_running_requests,
                swa_max_total_num_tokens=swa_max_total_num_tokens,
                c4_max_total_num_tokens=c4_max_total_num_tokens,
                c128_max_total_num_tokens=c128_max_total_num_tokens,
                c4_state_pool_size=c4_state_pool_size,
                c128_state_pool_size=c128_state_pool_size,
                state_dtype=state_dtype,
            )
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            if self.use_mla_backend and is_nsa_model:
                token_to_kv_pool = self._oot_nsa_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                )
            elif self.use_mla_backend:
                token_to_kv_pool = self._oot_mla_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                    is_nsa_model=is_nsa_model,
                )
            else:
                token_to_kv_pool = self._oot_mha_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                )
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            if self.is_hybrid_swa:
                token_to_kv_pool = self._ascend_swa_kv_pool(
                    full_max_total_num_tokens=full_max_total_num_tokens,
                    swa_max_total_num_tokens=swa_max_total_num_tokens,
                )
            elif self.use_mla_backend:
                token_to_kv_pool = self._ascend_mla_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                    is_nsa_model=is_nsa_model,
                )
            else:
                token_to_kv_pool = self._ascend_mha_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                )
        elif self.use_mla_backend and is_nsa_model:
            token_to_kv_pool = self._nsa_kv_pool(
                max_total_num_tokens=max_total_num_tokens,
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_nsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                token_to_kv_pool = self._mla_fp4_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                )
            else:
                token_to_kv_pool = self._mla_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                )
        else:
            if self.is_hybrid_swa:
                token_to_kv_pool = self._hybrid_swa_kv_pool(
                    full_max_total_num_tokens=full_max_total_num_tokens,
                    swa_max_total_num_tokens=swa_max_total_num_tokens,
                )
            elif self.mambaish_config:
                token_to_kv_pool = self._hybrid_linear_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                    req_to_token_pool=req_to_token_pool,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    token_to_kv_pool = self._mha_fp4_kv_pool(
                        max_total_num_tokens=max_total_num_tokens,
                    )
                else:
                    token_to_kv_pool = self._mha_kv_pool(
                        max_total_num_tokens=max_total_num_tokens,
                    )
        return token_to_kv_pool

    def _dsv4_kv_pool(
        self,
        *,
        max_running_requests: int,
        swa_max_total_num_tokens: Optional[int],
        c4_max_total_num_tokens: int,
        c128_max_total_num_tokens: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        state_dtype: Optional[torch.dtype],
    ) -> KVCache:
        swa_page_size = self.page_size
        assert swa_page_size == 256, "In paged swa mode, page_size must be 256."

        if self.is_draft_worker:
            from sglang.srt.models.deepseek_v4_nextn import (
                COMPRESS_RATIO_NEXTN_LAYER,
            )

            compression_ratios = [
                COMPRESS_RATIO_NEXTN_LAYER
            ] * self.num_effective_layers
        else:
            compression_ratios = self.model_config.compress_ratios
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=max_running_requests,
            swa_size=swa_max_total_num_tokens,
            c4_size=c4_max_total_num_tokens,
            c128_size=c128_max_total_num_tokens,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            page_size=self.page_size,
            swa_page_size=swa_page_size,
            dtype=self.kv_cache_dtype,
            state_dtype=state_dtype,
            qk_nope_head_dim=self.model_config.qk_nope_head_dim,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            indexer_head_dim=self.model_config.index_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            compression_ratios=compression_ratios,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            enable_hisparse=self.enable_hisparse,
        )

    def _oot_nsa_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        from sglang.srt.platforms import current_platform

        PoolCls = current_platform.get_nsa_kv_pool_cls()
        return PoolCls(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config,
                kv_cache_dtype=self.kv_cache_dtype,
                server_args=self.server_args,
            ),
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            index_head_dim=get_nsa_index_head_dim(self.model_config.hf_config),
        )

    def _oot_mla_kv_pool(
        self, *, max_total_num_tokens: int, is_nsa_model: bool
    ) -> KVCache:
        from sglang.srt.platforms import current_platform

        PoolCls = current_platform.get_mla_kv_pool_cls()
        return PoolCls(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            index_head_dim=(self.model_config.index_head_dim if is_nsa_model else None),
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

    def _oot_mha_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        from sglang.srt.platforms import current_platform

        PoolCls = current_platform.get_mha_kv_pool_cls()
        return PoolCls(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

    def _ascend_swa_kv_pool(
        self,
        *,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ) -> KVCache:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMHATokenToKVPool,
        )

        kwargs = {}
        if self.is_hybrid_swa_compress:
            kwargs = {
                "swa_head_num": max(
                    1,
                    self.model_config.hf_text_config.swa_num_key_value_heads
                    // get_attention_tp_size(),
                ),
                "swa_head_dim": self.model_config.hf_text_config.swa_head_dim,
                "swa_v_head_dim": self.model_config.hf_text_config.swa_v_head_dim,
                "v_head_dim": self.model_config.hf_text_config.v_head_dim,
            }
        return SWAKVPool(
            size=full_max_total_num_tokens,
            size_swa=swa_max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
            full_attention_layer_ids=self.model_config.full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=self.device,
            token_to_kv_pool_class=NPUMHATokenToKVPool,
            **kwargs,
        )

    def _ascend_mla_kv_pool(
        self, *, max_total_num_tokens: int, is_nsa_model: bool
    ) -> KVCache:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMLATokenToKVPool,
        )

        return NPUMLATokenToKVPool(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            index_head_dim=(self.model_config.index_head_dim if is_nsa_model else None),
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

    def _ascend_mha_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMHATokenToKVPool,
        )

        return NPUMHATokenToKVPool(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

    def _nsa_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        PoolCls = HiSparseNSATokenToKVPool if self.enable_hisparse else NSATokenToKVPool
        pool_kwargs = {}
        if self.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                self.server_args
            ).host_to_device_ratio
        return PoolCls(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config,
                kv_cache_dtype=self.kv_cache_dtype,
                server_args=self.server_args,
            ),
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            index_head_dim=get_nsa_index_head_dim(self.model_config.hf_config),
            **pool_kwargs,
        )

    def _mla_fp4_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        return MLATokenToKVPoolFP4(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

    def _mla_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        return MLATokenToKVPool(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

    def _hybrid_swa_kv_pool(
        self,
        *,
        full_max_total_num_tokens: Optional[int],
        swa_max_total_num_tokens: Optional[int],
    ) -> KVCache:
        kwargs = {}
        if self.is_hybrid_swa_compress:
            kwargs = {
                "swa_head_num": max(
                    1,
                    self.model_config.hf_text_config.swa_num_key_value_heads
                    // get_attention_tp_size(),
                ),
                "swa_head_dim": self.model_config.hf_text_config.swa_head_dim,
                "swa_v_head_dim": self.model_config.hf_text_config.swa_v_head_dim,
                "v_head_dim": self.model_config.hf_text_config.v_head_dim,
            }
        return SWAKVPool(
            size=full_max_total_num_tokens,
            size_swa=swa_max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
            full_attention_layer_ids=self.model_config.full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=self.device,
            **kwargs,
        )

    def _hybrid_linear_kv_pool(
        self, *, max_total_num_tokens: int, req_to_token_pool: ReqToTokenPool
    ) -> KVCache:
        extra_args = {}
        if self.use_mla_backend:
            extra_args = {
                "kv_lora_rank": self.model_config.kv_lora_rank,
                "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
            }
        return HybridLinearKVPool(
            page_size=self.page_size,
            size=max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            # if draft worker, we only need 1 attention layer's kv pool
            full_attention_layer_ids=(
                [0]
                if self.is_draft_worker
                else [
                    i
                    for i in self.mambaish_config.full_attention_layer_ids
                    if self.start_layer <= i < self.end_layer
                ]
            ),
            enable_kvcache_transpose=False,
            device=self.device,
            mamba_pool=req_to_token_pool.mamba_pool,
            enable_memory_saver=self.server_args.enable_memory_saver,
            use_mla=self.use_mla_backend,
            start_layer=self.start_layer,
            **extra_args,
        )

    def _mha_fp4_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        return MHATokenToKVPoolFP4(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            v_head_dim=self.model_config.v_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            enable_alt_stream=not self.server_args.enable_pdmux,
            enable_kv_cache_copy=(self.server_args.speculative_algorithm is not None),
        )

    def _mha_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        return MHATokenToKVPool(
            max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            v_head_dim=self.model_config.v_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            enable_alt_stream=not self.server_args.enable_pdmux,
            enable_kv_cache_copy=(self.server_args.speculative_algorithm is not None),
        )

    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self._handle_max_mamba_cache(rest_memory)

        return int(rest_memory * (1 << 30))  # return in bytes

    def _calculate_mamba_ratio(self) -> int:
        if self.server_args.disable_radix_cache:
            return 1

        additional_ratio = 0
        if self.server_args.enable_mamba_extra_buffer():
            # ping-pong buffer size is 2 when overlap schedule is on, 1 otherwise.
            if not self.server_args.disable_overlap_schedule:
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP

        return MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO + additional_ratio

    def _apply_token_constraints(self, token_capacity: int) -> int:
        """Apply external constraints to token capacity: user cap, PP sync.

        Page alignment is handled by the configurator, not here.
        If constraints change the value, the configurator re-runs and re-aligns.
        """
        user_limit = self.server_args.max_total_tokens

        # Apply user-specified upper bound
        if user_limit is not None:
            if user_limit > token_capacity:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{token_capacity}. Use the profiled value instead."
                )
            token_capacity = min(token_capacity, user_limit)

        # Sync across PP ranks (each may have different layer counts)
        if self.pp_size > 1:
            tensor = torch.tensor(token_capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            token_capacity = tensor.item()

        return token_capacity

    def _resolve_max_num_reqs(self, token_capacity: int) -> int:
        """Compute max concurrent requests (per dp worker) from the finalized
        token capacity."""
        # Estimate pool size (used as upper bound when user specifies max_running_requests)
        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = self.server_args.max_running_requests
        if max_num_reqs is not None:
            max_num_reqs = min(max_num_reqs // self.dp_size, estimated)
        else:
            max_num_reqs = min(estimated, token_capacity // 2)

        if self.mambaish_config is not None:
            ratio = self._calculate_mamba_ratio()
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )

        return max_num_reqs

    def _resolve_memory_pool_config(
        self, pre_model_load_memory: int
    ) -> MemoryPoolConfig:
        """Profile GPU memory and resolve all pool parameters into a config."""
        from sglang.srt.model_executor.model_runner_components.pool_configurator import (
            create_memory_pool_configurator,
        )

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        page_size = self.server_args.page_size

        configurator = create_memory_pool_configurator(self)
        config = configurator.calculate_pool_sizes(available_bytes, page_size)

        # Apply external constraints (user cap, page alignment, PP sync)
        constrained = self._apply_token_constraints(config.max_total_num_tokens)
        if constrained != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(
                constrained, page_size
            )

        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        config.mem_fraction_static = self.server_args.mem_fraction_static
        return config

    def _handle_max_mamba_cache(self, total_rest_memory):
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        # reserve the memory for the intermediate mamba states used for spec dec
        if not self.spec_algorithm.is_none():
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

            max_running_requests = server_args.max_running_requests // (
                self.dp_size if server_args.enable_dp_attention else 1
            )
            mamba_state_intermediate_size = (
                config.mamba2_cache_params.mamba_cache_per_req
                * max_running_requests
                * server_args.speculative_num_draft_tokens
            )
            total_rest_memory = total_rest_memory - (
                mamba_state_intermediate_size / (1 << 30)
            )

        if server_args.max_mamba_cache_size is not None:
            # Use explicitly set max_mamba_cache_size
            server_args.max_mamba_cache_size = server_args.max_mamba_cache_size // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        elif (
            server_args.disable_radix_cache
            and server_args.max_running_requests is not None
        ):
            # Use explicitly set max_running_requests when radix cache is disabled
            server_args.max_mamba_cache_size = server_args.max_running_requests // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        else:
            # Use ratio-based calculation to auto-fit available memory
            assert config.mamba2_cache_params.mamba_cache_per_req > 0

            # allocate the memory based on the ratio between mamba state memory vs. full kv cache memory
            # solve the equations:
            # 1. mamba_state_memory + full_kv_cache_memory == total_rest_memory
            # 2. mamba_state_memory / full_kv_cache_memory == server_args.mamba_full_memory_ratio
            mamba_state_memory_raw = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            # calculate the max_mamba_cache_size based on the given total mamba memory
            server_args.max_mamba_cache_size = int(
                (mamba_state_memory_raw * (1 << 30))
                // config.mamba2_cache_params.mamba_cache_per_req
            )

        mamba_state_memory = (
            server_args.max_mamba_cache_size
            * config.mamba2_cache_params.mamba_cache_per_req
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory


def calculate_mla_kv_cache_dim(
    *,
    model_config: ModelConfig,
    kv_cache_dtype: torch.dtype,
    server_args: ServerArgs,
) -> int:
    is_nsa_model = is_deepseek_nsa(model_config.hf_config)
    kv_cache_dtype = kv_cache_dtype
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

    # For non-NSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
    if not is_nsa_model:
        return kv_cache_dim

    # TRTLLM backend does not override kv_cache_dim for MLA kv cache
    # Assuming nsa prefill and decode backends are the same when using trtllm MLA backend,
    # since it is not compatible for trtllm and other mla attn backend due to the different
    # kv cache layout.
    if (
        server_args.nsa_prefill_backend == "trtllm"
        or server_args.nsa_decode_backend == "trtllm"
    ):
        return kv_cache_dim

    # On HIP with TileLang backend, keep the default MLA KV cache dimension.
    # FP8 attention uses the nope(512 fp8) + rope(64 fp8) layout, without extra per-block scales.
    if _is_hip and (
        server_args.nsa_prefill_backend == "tilelang"
        or server_args.nsa_decode_backend == "tilelang"
    ):
        return kv_cache_dim

    quant_block_size = NSATokenToKVPool.quant_block_size
    rope_storage_dtype = NSATokenToKVPool.rope_storage_dtype
    # Calculate override_kv_cache_dim for FP8 storage in backends that use scaled KV layout (excluding TRTLLM and HIP+TileLang).
    # kv_lora_rank + scale storage (kv_lora_rank // quant_block_size * 4 bytes) + rope dimension storage
    # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
    if kv_cache_dtype == torch.float8_e4m3fn:
        assert (
            kv_lora_rank % quant_block_size == 0
        ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

        return (
            kv_lora_rank
            + kv_lora_rank // quant_block_size * 4
            + qk_rope_head_dim * rope_storage_dtype.itemsize
        )

    return kv_cache_dim
