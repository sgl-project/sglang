from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.utils.common import is_hip

_is_hip = is_hip()
import logging
import math

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    is_deepseek_dsa,
    is_deepseek_v4,
)
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hisparse_memory_pool import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseDSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.platforms import current_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_npu,
)

logger = logging.getLogger(__name__)


def _get_dsv4_compress_state_dtypes() -> tuple[torch.dtype, torch.dtype]:
    dtype_name = envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.get().strip().lower()
    if dtype_name in ("float32", "fp32"):
        return torch.float32, torch.float32
    if dtype_name in ("bfloat16", "bf16"):
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            raise ValueError(
                "SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 is not supported when "
                "SGLANG_OPT_USE_ONLINE_COMPRESS=1; online c128 state must stay float32."
            )
        return torch.bfloat16, torch.bfloat16
    raise ValueError(
        "Unsupported SGLANG_DSV4_COMPRESS_STATE_DTYPE="
        f"{dtype_name!r}. Expected one of: float32, fp32, bfloat16, bf16."
    )


_is_npu = is_npu()

# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
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
    memory_pool_config: MemoryPoolConfig


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

    # deployment env (runtime, not in server_args)
    device: str
    gpu_id: int
    # model / dtype (resolved objects, not in server_args)
    model_config: ModelConfig
    server_args: ServerArgs
    kv_cache_dtype: torch.dtype
    # speculative decoding (runtime / derived, not in server_args)
    spec_algorithm: SpeculativeAlgorithm
    is_draft_worker: bool
    # DFLASH-only: target's `cell_size` is scaled to include draft KV cache.
    # ``pool_configurator.DefaultPoolConfigurator`` reads this off the
    # configurator (was ``getattr(mr, "dflash_draft_num_layers", None)`` in
    # the mixin era — silent ``None`` if missing). Must be plumbed through;
    # otherwise the target KV pool oversizes by 1+ GB on 32GB GPUs and
    # OOMs at cuda graph capture (see debug_journal 2026-05-11-kvc-...).
    dflash_draft_num_layers: Optional[int]
    # arch flags (derived, not direct server_args fields)
    is_hybrid_swa: bool
    is_hybrid_swa_compress: bool
    use_mla_backend: bool
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
    memory_pool_config: Optional[MemoryPoolConfig]

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

        # Draft worker does not own the compression-state pools, but keep the
        # dtype attributes initialized so _init_pools can share one code path.
        c4_state_dtype: Optional[torch.dtype] = None
        c128_state_dtype: Optional[torch.dtype] = None
        if is_deepseek_v4(self.model_config.hf_config):
            c4_state_dtype, c128_state_dtype = _get_dsv4_compress_state_dtypes()

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
                c4_state_dtype=c4_state_dtype,
                c128_state_dtype=c128_state_dtype,
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
        c4_state_dtype: Optional[torch.dtype],
        c128_state_dtype: Optional[torch.dtype],
        req_to_token_pool: Optional[ReqToTokenPool],
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator],
    ) -> tuple[ReqToTokenPool, KVCache, BaseTokenToKVPoolAllocator]:
        """Initialize the memory pools."""
        token_to_kv_pool = None
        max_num_reqs = max_running_requests

        # Initialize req_to_token_pool
        if req_to_token_pool is None:
            extra_max_context_len = get_req_to_token_extra_context_len(self.server_args)

            if self.server_args.disaggregation_mode == "decode":
                # Extra slots for pre-allocated requests
                pre_alloc_size = self.server_args.disaggregation_decode_extra_slots
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
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        self._validate_prefill_only_disable_kv_cache_pool_family(
            is_dsa_model, is_dsv4_model, current_platform
        )

        if is_dsv4_model:
            token_to_kv_pool = self._dsv4_kv_pool(
                max_running_requests=max_running_requests,
                swa_max_total_num_tokens=swa_max_total_num_tokens,
                c4_max_total_num_tokens=c4_max_total_num_tokens,
                c128_max_total_num_tokens=c128_max_total_num_tokens,
                c4_state_pool_size=c4_state_pool_size,
                c128_state_pool_size=c128_state_pool_size,
                c4_state_dtype=c4_state_dtype,
                c128_state_dtype=c128_state_dtype,
                req_to_token_pool=req_to_token_pool,
            )
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            if self.use_mla_backend and is_dsa_model:
                token_to_kv_pool = self._oot_dsa_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                )
            elif self.use_mla_backend:
                token_to_kv_pool = self._oot_mla_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                    is_dsa_model=is_dsa_model,
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
                    is_dsa_model=is_dsa_model,
                )
            else:
                token_to_kv_pool = self._ascend_mha_kv_pool(
                    max_total_num_tokens=max_total_num_tokens,
                )
        elif self.use_mla_backend and is_dsa_model:
            token_to_kv_pool = self._dsa_kv_pool(
                max_total_num_tokens=max_total_num_tokens,
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_dsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                token_to_kv_pool = MLATokenToKVPoolFP4(
                    max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                token_to_kv_pool = MLATokenToKVPool(
                    max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        else:
            if self.is_hybrid_swa:
                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.v_head_dim,
                    }
                token_to_kv_pool = SWAKVPool(
                    size=full_max_total_num_tokens,
                    size_swa=swa_max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    **kwargs,
                )
            elif config := self.mambaish_config:
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.server_args.page_size,
                    size=max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool
                    full_attention_layer_ids=(
                        [0]
                        if self.is_draft_worker
                        else [
                            i
                            for i in config.full_attention_layer_ids
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_kvcache_transpose=False,
                    device=self.device,
                    mamba_pool=req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    use_mla=self.use_mla_backend,
                    start_layer=self.start_layer,
                    **extra_args,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    token_to_kv_pool = MHATokenToKVPoolFP4(
                        max_total_num_tokens,
                        page_size=self.server_args.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )
                else:
                    pool_cls = (
                        NoOpMHATokenToKVPool
                        if self.server_args.prefill_only_disable_kv_cache
                        else MHATokenToKVPool
                    )
                    token_to_kv_pool = pool_cls(
                        max_total_num_tokens,
                        page_size=self.server_args.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        v_head_dim=self.model_config.v_head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )

        # Initialize token_to_kv_pool_allocator
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if token_to_kv_pool_allocator is None:
            if current_platform.is_out_of_tree():
                AllocatorCls = current_platform.get_paged_allocator_cls()
                token_to_kv_pool_allocator = AllocatorCls(
                    max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=token_to_kv_pool,
                    need_sort=need_sort,
                )
            elif _is_npu and (
                self.server_args.attention_backend == "ascend"
                or is_dsv4_model
                or self.hybrid_gdn_config is not None
            ):
                if self.is_hybrid_swa:
                    # DSV4 on NPU: SWA allocator subclass that also drives the
                    # c4/c128 allocators, producing a DSV4OutCacheLoc per alloc.
                    if is_dsv4_model:
                        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
                            DSV4NPUTokenToKVPoolAllocator,
                        )

                        swa_allocator_cls = DSV4NPUTokenToKVPoolAllocator
                    else:
                        swa_allocator_cls = SWATokenToKVPoolAllocator
                    token_to_kv_pool_allocator = swa_allocator_cls(
                        full_max_total_num_tokens,
                        swa_max_total_num_tokens,
                        page_size=self.server_args.page_size,
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
                        page_size=self.server_args.page_size,
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
                        page_size=self.server_args.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.server_args.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        token_to_kv_pool_allocator = HiSparseTokenToKVPoolAllocator(
                            max_total_num_tokens,
                            page_size=self.server_args.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                            host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                        )
                    elif (
                        self.server_args.page_size == 1
                        and self.server_args.dcp_size == 1
                    ):
                        token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            max_total_num_tokens * self.server_args.dcp_size,
                            page_size=self.server_args.page_size
                            * self.server_args.dcp_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=token_to_kv_pool,
                            need_sort=need_sort,
                        )

            if self.server_args.enable_hisparse and is_dsv4_model:
                assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
                token_to_kv_pool_allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(
                    token_to_kv_pool_allocator
                )

            # DSV4-NPU: wire allocator back-ref into req_to_token_pool so its
            # free(req) can release c4/c128 pool pages alongside the slot.
            if hasattr(req_to_token_pool, "register_dsv4_allocator"):
                req_to_token_pool.register_dsv4_allocator(token_to_kv_pool_allocator)

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

        # Defensive check: the explicit validation above should reject known
        # unsupported pool families before allocation. Keep this guard here so
        # future pool-selection refactors fail at boot instead of on first use.
        if (
            self.server_args.prefill_only_disable_kv_cache
            and not self.is_draft_worker
            and not isinstance(token_to_kv_pool, NoOpMHATokenToKVPool)
        ):
            raise RuntimeError(
                "--prefill-only-disable-kv-cache expected NoOpMHATokenToKVPool but the "
                f"runtime pool is {type(token_to_kv_pool).__name__}. This pool "
                "family is not yet supported by --prefill-only-disable-kv-cache. "
                "Supported configurations today: plain MHA models on CUDA with the FA "
                "(fa3/fa4) prefill backend, --is-embedding, --chunked-prefill-size=-1, "
                "--disable-radix-cache, no context-parallel attention, no HiSparse, "
                "and --kv-cache-dtype != fp4_e2m1."
            )
        return req_to_token_pool, token_to_kv_pool, token_to_kv_pool_allocator

    def _validate_prefill_only_disable_kv_cache_pool_family(
        self,
        is_dsa_model: bool,
        is_dsv4_model: bool,
        current_platform,
    ):
        if not self.server_args.prefill_only_disable_kv_cache or self.is_draft_worker:
            return

        unsupported_pool_family = None
        if is_dsv4_model:
            unsupported_pool_family = "DeepSeekV4TokenToKVPool"
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            unsupported_pool_family = "out-of-tree platform KV pool"
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
        ):
            unsupported_pool_family = "NPU/Ascend KV pool"
        elif self.use_mla_backend and is_dsa_model:
            unsupported_pool_family = "DSA/MLA KV pool"
        elif self.use_mla_backend and not self.mambaish_config:
            unsupported_pool_family = "MLA KV pool"
        elif self.is_hybrid_swa:
            unsupported_pool_family = "SWA KV pool"
        elif self.mambaish_config:
            unsupported_pool_family = "hybrid linear/Mamba KV pool"
        elif is_float4_e2m1fn_x2(self.kv_cache_dtype):
            unsupported_pool_family = "FP4 MHA KV pool"

        if unsupported_pool_family is not None:
            raise RuntimeError(
                "--prefill-only-disable-kv-cache is not supported for "
                f"{unsupported_pool_family}. Supported configurations today: plain MHA "
                "models on CUDA with the FA (fa3/fa4) prefill backend, --is-embedding, "
                "--chunked-prefill-size=-1, --disable-radix-cache, no context-parallel "
                "attention, no HiSparse, and --kv-cache-dtype != fp4_e2m1."
            )

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
            speculative_num_draft_tokens=self.server_args.max_speculative_num_draft_tokens,
            speculative_eagle_topk=self.server_args.speculative_eagle_topk,
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
            enable_mamba_extra_buffer_lazy=self.server_args.enable_mamba_extra_buffer_lazy(),
            speculative_num_draft_tokens=self.server_args.max_speculative_num_draft_tokens,
            speculative_eagle_topk=self.server_args.speculative_eagle_topk,
            enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
            start_layer=self.start_layer,
        )

    def _default_req_pool(
        self,
        *,
        max_num_reqs: int,
        extra_max_context_len: int,
    ) -> ReqToTokenPool:
        # DSV4 on NPU needs an extended ReqToTokenPool holding per-req
        # swa/c4/c128/c{4,128}_state tables; others stay on the stock one.
        req_to_token_pool_cls = ReqToTokenPool
        if _is_npu and is_deepseek_v4(self.model_config.hf_config):
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                DSV4NPUReqToTokenPool,
            )

            req_to_token_pool_cls = DSV4NPUReqToTokenPool

        return req_to_token_pool_cls(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
        )

    def _dsv4_kv_pool(
        self,
        *,
        max_running_requests: int,
        swa_max_total_num_tokens: Optional[int],
        c4_max_total_num_tokens: int,
        c128_max_total_num_tokens: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        c4_state_dtype: Optional[torch.dtype],
        c128_state_dtype: Optional[torch.dtype],
        req_to_token_pool: ReqToTokenPool,
    ) -> KVCache:
        swa_page_size = self.server_args.page_size
        if not _is_npu:
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

        # NPU + DSV4 → paged-state subclass: the fused compressor kernel
        # needs cache_mode=1 (paged); Atlas A3 rejects cache_mode=2 (ring),
        # so the CUDA ring-buffer state path can't be shared. CUDA keeps
        # DeepSeekV4TokenToKVPool unchanged; NPU recomputes state sizes below.
        if _is_npu:
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
                DSV4NPUTokenToKVPool,
                npu_state_pool_size,
            )

            pool_cls = DSV4NPUTokenToKVPool
            # Recompute state pool sizes for the NPU paged formula (CUDA's
            # ring sizes are dropped here). Tail-only allocation keeps the
            # per-req-budget formula sufficient at any prefill length: long
            # prompts allocate only ``tail+128`` (c4) / ``tail`` (c128)
            # slots (tail = seq_len % 128), and decode is drained by
            # sliding eviction in ``ScheduleBatch._evict_swa``.
            c4_state_pool_size = npu_state_pool_size(
                ratio=4,
                page_size=self.server_args.page_size,
                max_num_reqs=max_running_requests,
            )
            c128_state_pool_size = npu_state_pool_size(
                ratio=128,
                page_size=self.server_args.page_size,
                max_num_reqs=max_running_requests,
            )
        else:
            pool_cls = DeepSeekV4TokenToKVPool
            c4_state_pool_size = c4_state_pool_size
            c128_state_pool_size = c128_state_pool_size

        return pool_cls(
            max_num_reqs=max_running_requests,
            # SWA ring is indexed by req_pool_idx; PD decode inflates req_to_token
            # past max_running_requests (pre-alloc), so size to the real capacity.
            num_req_slots=req_to_token_pool.req_to_token.shape[0],
            swa_size=swa_max_total_num_tokens,
            c4_size=c4_max_total_num_tokens,
            c128_size=c128_max_total_num_tokens,
            c4_state_pool_size=c4_state_pool_size,
            c128_state_pool_size=c128_state_pool_size,
            page_size=self.server_args.page_size,
            swa_page_size=swa_page_size,
            sliding_window=self.model_config.window_size,
            dtype=self.kv_cache_dtype,
            c4_state_dtype=c4_state_dtype,
            c128_state_dtype=c128_state_dtype,
            qk_nope_head_dim=self.model_config.qk_nope_head_dim,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            indexer_head_dim=self.model_config.index_head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            compression_ratios=compression_ratios,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            enable_hisparse=self.server_args.enable_hisparse,
            online_mtp_max_draft_tokens=(
                self.server_args.max_speculative_num_draft_tokens or 0
            ),
        )

    def _oot_dsa_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        from sglang.srt.platforms import current_platform

        PoolCls = current_platform.get_dsa_kv_pool_cls()
        return PoolCls(
            max_total_num_tokens,
            page_size=self.server_args.page_size,
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
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
        )

    def _oot_mla_kv_pool(
        self, *, max_total_num_tokens: int, is_dsa_model: bool
    ) -> KVCache:
        from sglang.srt.platforms import current_platform

        PoolCls = current_platform.get_mla_kv_pool_cls()
        return PoolCls(
            max_total_num_tokens,
            page_size=self.server_args.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            index_head_dim=(self.model_config.index_head_dim if is_dsa_model else None),
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
            page_size=self.server_args.page_size,
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
                "swa_head_dim": self.model_config.swa_head_dim,
                "swa_v_head_dim": self.model_config.swa_v_head_dim,
                "v_head_dim": self.model_config.v_head_dim,
            }
        return SWAKVPool(
            size=full_max_total_num_tokens,
            size_swa=swa_max_total_num_tokens,
            page_size=self.server_args.page_size,
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
        self, *, max_total_num_tokens: int, is_dsa_model: bool
    ) -> KVCache:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMLATokenToKVPool,
        )

        return NPUMLATokenToKVPool(
            max_total_num_tokens,
            page_size=self.server_args.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            index_head_dim=(self.model_config.index_head_dim if is_dsa_model else None),
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
            page_size=self.server_args.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )

    def _dsa_kv_pool(self, *, max_total_num_tokens: int) -> KVCache:
        PoolCls = (
            HiSparseDSATokenToKVPool
            if self.server_args.enable_hisparse
            else DSATokenToKVPool
        )
        pool_kwargs = {}
        if self.server_args.enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                self.server_args
            ).host_to_device_ratio
        return PoolCls(
            max_total_num_tokens,
            page_size=self.server_args.page_size,
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
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
            **pool_kwargs,
        )

    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        # KV pool budget = currently-free GPU memory minus the non-static runtime
        # slack (pre_model_load_memory * (1 - mem_fraction_static)). Whatever is
        # already resident (model weights, etc.) is thus charged against it.
        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        rest_memory = available_gpu_memory - pre_model_load_memory * (
            1 - self.server_args.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self._handle_max_mamba_cache(rest_memory)

        # Loaded weights (target + draft) can exceed the static budget
        if rest_memory <= 0:
            minimum_mem_fraction_static = (
                1 - available_gpu_memory / pre_model_load_memory
            )
            suggested_mem_fraction_static = (
                math.ceil(minimum_mem_fraction_static * 1000) / 1000
            )
            raise ValueError(
                f"Loaded weights leave no GPU memory for the KV cache under "
                f"--mem-fraction-static={self.server_args.mem_fraction_static}. "
                f"Raise --mem-fraction-static above "
                f"{suggested_mem_fraction_static:.3f} "
                f"(minimum viable = 1 - available/pre = "
                f"{minimum_mem_fraction_static:.4f}). If using speculative "
                f"decoding, draft weights are now counted."
            )

        return int(rest_memory * (1 << 30))  # return in bytes

    def _calculate_mamba_ratio(self) -> int:
        if self.server_args.disable_radix_cache:
            return 1

        additional_ratio = 0
        if self.server_args.enable_mamba_extra_buffer():
            # ping-pong buffer size is 2 when overlap schedule is on, 1 otherwise.
            # Lazy mode saves 1 slot (2 → 1) for overlap; non-overlap already uses 1.
            if not self.server_args.disable_overlap_schedule:
                if self.server_args.enable_mamba_extra_buffer_lazy():
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY
                else:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                assert (
                    not self.server_args.enable_mamba_extra_buffer_lazy()
                ), "Lazy extra buffer requires overlap schedule (--disable-overlap-schedule is incompatible)"
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
        if self.server_args.pp_size > 1:
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
            requested_per_worker = max_num_reqs // self.server_args.dp_size
            max_num_reqs = min(requested_per_worker, token_capacity // 2)
        else:
            requested_per_worker = None
            max_num_reqs = min(estimated, token_capacity // 2)

        if self.mambaish_config is not None:
            ratio = self._calculate_mamba_ratio()
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )

            if max_num_reqs <= 0:
                raise RuntimeError(
                    f"Hybrid (mamba/linear-attention) state cache is too small to serve "
                    f"any requests. max_mamba_cache_size={self.server_args.max_mamba_cache_size}, "
                    f"mamba_ratio={ratio}, resulting max_num_reqs={max_num_reqs}. "
                    f"Try: (1) reduce --max-running-requests, "
                    f"(2) increase --mem-fraction-static, or "
                    f"(3) use GPUs with more memory."
                )
        if requested_per_worker is not None and max_num_reqs < requested_per_worker:
            logger.warning(
                "max_running_requests was reduced from the requested %d to %d "
                "(per dp worker) due to the available KV cache capacity.",
                requested_per_worker,
                max_num_reqs,
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

        has_spec_dec = not self.spec_algorithm.is_none()
        if has_spec_dec:
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

        if server_args.max_mamba_cache_size is not None:
            # Use explicitly set max_mamba_cache_size
            server_args.max_mamba_cache_size = (
                server_args.max_mamba_cache_size // self.ps.attn_dp_size
            )
            # Reserve intermediate memory based on capped max_num_reqs
            if has_spec_dec:
                ratio = self._calculate_mamba_ratio()
                capped_reqs = min(
                    server_args.max_running_requests // self.ps.attn_dp_size,
                    server_args.max_mamba_cache_size // ratio,
                )
                intermediate_size = (
                    config.mamba2_cache_params.mamba_cache_per_req
                    * capped_reqs
                    * server_args.speculative_num_draft_tokens
                )
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        elif (
            server_args.disable_radix_cache
            and server_args.max_running_requests is not None
        ):
            # Use explicitly set max_running_requests when radix cache is disabled
            server_args.max_mamba_cache_size = (
                server_args.max_running_requests // self.ps.attn_dp_size
            )
            # Reserve intermediate memory based on capped max_num_reqs
            if has_spec_dec:
                intermediate_size = (
                    config.mamba2_cache_params.mamba_cache_per_req
                    * server_args.max_mamba_cache_size
                    * server_args.speculative_num_draft_tokens
                )
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
        else:
            # Use ratio-based calculation to auto-fit available memory
            assert config.mamba2_cache_params.mamba_cache_per_req > 0
            per_req = config.mamba2_cache_params.mamba_cache_per_req

            # Solve jointly for max_mamba_cache_size accounting for intermediate memory.
            # The mamba budget (from the ratio split) must cover both:
            #   1. main mamba state: max_mamba_cache_size * per_req
            #   2. intermediate states: (max_mamba_cache_size / ratio) * D * per_req
            # So: max_mamba_cache_size * per_req * (1 + D/ratio) = mamba_budget_bytes
            mamba_budget = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            mamba_budget_bytes = mamba_budget * (1 << 30)

            if has_spec_dec:
                ratio = self._calculate_mamba_ratio()
                D = server_args.speculative_num_draft_tokens
                # Joint solve: main_state + intermediate = mamba_budget
                server_args.max_mamba_cache_size = int(
                    mamba_budget_bytes // (per_req * (1 + D / ratio))
                )
                # Intermediate memory is included in mamba_budget, subtract it
                # so the return value only has main_state subtracted from total
                capped_reqs = min(
                    server_args.max_running_requests // self.ps.attn_dp_size,
                    server_args.max_mamba_cache_size // ratio,
                )
                intermediate_size = per_req * capped_reqs * D
                total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
            else:
                server_args.max_mamba_cache_size = int(mamba_budget_bytes // per_req)

        # Validate: max_mamba_cache_size must be positive after memory allocation.
        # A non-positive value means GPU memory is insufficient for the requested
        # configuration. Fail fast with actionable advice instead of silently
        # producing garbled output at runtime.
        if server_args.max_mamba_cache_size <= 0:
            raise RuntimeError(
                f"Not enough GPU memory for hybrid (mamba/linear-attention) state cache. "
                f"Computed max_mamba_cache_size={server_args.max_mamba_cache_size} "
                f"(total_rest_memory={total_rest_memory:.2f} GB, "
                f"mamba_cache_per_req={config.mamba2_cache_params.mamba_cache_per_req / (1 << 20):.2f} MB). "
                f"Try: (1) reduce --max-running-requests, "
                f"(2) increase --mem-fraction-static, "
                f"(3) reduce --speculative-num-draft-tokens, or "
                f"(4) use GPUs with more memory."
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
    is_dsa_model = is_deepseek_dsa(model_config.hf_config)
    kv_cache_dtype = kv_cache_dtype
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

    # For non-DSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
    if not is_dsa_model:
        return kv_cache_dim

    # TRTLLM backend does not override kv_cache_dim for MLA kv cache
    # Assuming dsa prefill and decode backends are the same when using trtllm MLA backend,
    # since it is not compatible for trtllm and other mla attn backend due to the different
    # kv cache layout.
    if (
        server_args.dsa_prefill_backend == "trtllm"
        or server_args.dsa_decode_backend == "trtllm"
    ):
        return kv_cache_dim

    # On HIP, TileLang and AITER DSA kernels consume the raw MLA KV layout:
    # nope(512 fp8) + rope(64 fp8), without extra per-block scales.
    if _is_hip and (
        server_args.dsa_prefill_backend in ("tilelang", "aiter")
        or server_args.dsa_decode_backend in ("tilelang", "aiter")
    ):
        return kv_cache_dim

    quant_block_size = DSATokenToKVPool.quant_block_size
    rope_storage_dtype = DSATokenToKVPool.rope_storage_dtype
    # Calculate override_kv_cache_dim for FP8 storage in backends that use scaled KV layout
    # (excluding TRTLLM and HIP raw-layout kernels).
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
