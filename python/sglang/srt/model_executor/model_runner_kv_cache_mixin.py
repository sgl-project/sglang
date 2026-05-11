from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    mambaish_config,
)
from sglang.srt.configs.model_config import (
    get_nsa_index_head_dim,
    is_deepseek_nsa,
    is_deepseek_v4,
)
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
from sglang.srt.mem_cache.kv_cache_configurator import (
    calculate_mla_kv_cache_dim,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_components.pool_configurator import (
        MemoryPoolConfig,
    )


# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

logger = logging.getLogger(__name__)

_is_npu = is_npu()
_is_hip = is_hip()


class ModelRunnerKVCacheMixin:
    def _profile_available_bytes(self: ModelRunner, pre_model_load_memory: int) -> int:
        return self.kv_cache_configurator._profile_available_bytes(
            pre_model_load_memory
        )

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        return self.kv_cache_configurator._calculate_mamba_ratio()

    def _init_pools(self: ModelRunner):
        """Initialize the memory pools."""
        max_num_reqs = self.max_running_requests

        # Initialize req_to_token_pool
        if self.req_to_token_pool is None:
            # FIXME(lsyin): this is the temporary fix for the context length issue when using speculative decoding
            extra_max_context_len = 4
            if self.server_args.speculative_num_draft_tokens is not None:
                extra_max_context_len += self.server_args.speculative_num_draft_tokens

            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # subscribe memory for pre-allocated requests
                # if max_num_reqs <= 32, we pre-allocate 2x requests

                pre_alloc_size = envs.SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS.get()
                pre_alloc_size = (
                    max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size
                )
                if config := mambaish_config(self.model_config):
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        mamba_layer_ids=(
                            [
                                i
                                for i in config.mamba2_cache_params.layers
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
                else:
                    self.req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := mambaish_config(self.model_config):
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    mamba_spec_state_size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    mamba_layer_ids=(
                        [
                            i
                            for i in config.mamba2_cache_params.layers
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                    speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                    enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                    start_layer=self.start_layer,
                )
            else:
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # Initialize token_to_kv_pool
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        # Out-of-tree platform plugin system — used by elif below
        from sglang.srt.platforms import current_platform

        if is_dsv4_model:
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
            self.token_to_kv_pool = DeepSeekV4TokenToKVPool(
                max_num_reqs=self.max_running_requests,
                swa_size=self.swa_max_total_num_tokens,
                c4_size=self.c4_max_total_num_tokens,
                c128_size=self.c128_max_total_num_tokens,
                c4_state_pool_size=self.c4_state_pool_size,
                c128_state_pool_size=self.c128_state_pool_size,
                page_size=self.page_size,
                swa_page_size=swa_page_size,
                dtype=self.kv_cache_dtype,
                state_dtype=self.state_dtype,
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
        elif current_platform.is_out_of_tree() and not mambaish_config(
            self.model_config
        ):
            if self.use_mla_backend and is_nsa_model:
                PoolCls = current_platform.get_nsa_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
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
            elif self.use_mla_backend:
                PoolCls = current_platform.get_mla_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_nsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                PoolCls = current_platform.get_mha_kv_pool_cls()
                self.token_to_kv_pool = PoolCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.server_args.attention_backend == "ascend" and not mambaish_config(
            self.model_config
        ):
            if self.is_hybrid_swa:
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
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    token_to_kv_pool_class=NPUMHATokenToKVPool,
                    **kwargs,
                )
            elif self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_nsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_nsa_model:
            PoolCls = (
                HiSparseNSATokenToKVPool if self.enable_hisparse else NSATokenToKVPool
            )
            pool_kwargs = {}
            if self.enable_hisparse:
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                    self.server_args
                ).host_to_device_ratio
            self.token_to_kv_pool = PoolCls(
                self.max_total_num_tokens,
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
        elif self.use_mla_backend and not mambaish_config(self.model_config):
            assert not is_nsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                self.token_to_kv_pool = MLATokenToKVPoolFP4(
                    self.max_total_num_tokens,
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
            else:
                self.token_to_kv_pool = MLATokenToKVPool(
                    self.max_total_num_tokens,
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
                        "swa_head_dim": self.model_config.hf_text_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.hf_text_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.hf_text_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    **kwargs,
                )
            elif config := mambaish_config(self.model_config):
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=self.max_total_num_tokens,
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
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    use_mla=self.use_mla_backend,
                    start_layer=self.start_layer,
                    **extra_args,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
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
                    self.token_to_kv_pool = MHATokenToKVPool(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
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
        if self.token_to_kv_pool_allocator is None:
            if current_platform.is_out_of_tree():
                AllocatorCls = current_platform.get_paged_allocator_cls()
                self.token_to_kv_pool_allocator = AllocatorCls(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
            elif _is_npu and (
                self.server_args.attention_backend == "ascend"
                or hybrid_gdn_config(self.model_config) is not None
            ):
                if self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    from sglang.srt.hardware_backend.npu.allocator_npu import (
                        NPUPagedTokenToKVPoolAllocator,
                    )

                    self.token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        self.token_to_kv_pool_allocator = (
                            HiSparseTokenToKVPoolAllocator(
                                self.max_total_num_tokens,
                                page_size=self.page_size,
                                dtype=self.kv_cache_dtype,
                                device=self.device,
                                kvcache=self.token_to_kv_pool,
                                need_sort=need_sort,
                                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                            )
                        )
                    elif self.page_size == 1:
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            page_size=self.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )

            if self.enable_hisparse and is_dsv4_model:
                assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
                self.token_to_kv_pool_allocator = (
                    DeepSeekV4HiSparseTokenToKVPoolAllocator(
                        self.token_to_kv_pool_allocator
                    )
                )

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                swa_allocator = getattr(
                    self.token_to_kv_pool_allocator,
                    "logical_attn_allocator",
                    self.token_to_kv_pool_allocator,
                )
                assert swa_allocator.__class__ == SWATokenToKVPoolAllocator
                self.token_to_kv_pool.full_to_swa_index_mapping = (
                    swa_allocator.full_to_swa_index_mapping
                )

    def _apply_token_constraints(self: ModelRunner, token_capacity: int) -> int:
        return self.kv_cache_configurator._apply_token_constraints(token_capacity)

    def _resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        return self.kv_cache_configurator._resolve_max_num_reqs(token_capacity)

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
            self.c4_max_total_num_tokens = 0
            self.c128_max_total_num_tokens = 0
            self.c4_state_pool_size = 0
            self.c128_state_pool_size = 0
        else:
            self.c4_max_total_num_tokens = config.c4_max_total_num_tokens
            self.c128_max_total_num_tokens = config.c128_max_total_num_tokens
            self.c4_state_pool_size = config.c4_state_pool_size
            self.c128_state_pool_size = config.c128_state_pool_size

        # state_dtype is a DSV4 architectural constant (fp32 for c4/c128
        # state buffers); set unconditionally so draft workers have it before
        # _init_pools reads it (target path also overwrites this in the
        # configurator's resolve() for parity, harmless here).
        if is_deepseek_v4(self.model_config.hf_config):
            self.state_dtype = torch.float32

        self._init_pools()

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
