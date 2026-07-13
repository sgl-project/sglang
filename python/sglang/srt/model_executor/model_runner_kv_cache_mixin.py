from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    is_deepseek_dsa,
    is_deepseek_v4,
    is_minimax_sparse,
)
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.kv_cache_configurator import (
    _get_dsv4_compress_state_dtypes,
    _InitializedPools,
    calculate_mla_kv_cache_dim,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
    PageMajorMHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
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
        """Initialize the memory pools."""
        token_to_kv_pool = None

        # Unified-pool fast path: build req_to_token + token_to_kv pool + allocator
        # from one byte buffer, then return. Gated to the target worker
        # (req_to_token_pool is None); supports hybrid Mamba and hybrid SWA (not DSV4).
        if (
            self.server_args.enable_unified_memory
            and self.server_args.disaggregation_mode == "null"
            and req_to_token_pool is None
        ):
            if self.mambaish_config is not None:
                bundle = self._init_unified_mamba_pools(
                    max_num_reqs=max_running_requests,
                    max_total_num_tokens=max_total_num_tokens,
                )
            elif self.is_hybrid_swa and not is_deepseek_v4(self.model_config.hf_config):
                bundle = self._init_unified_swa_pools(
                    max_num_reqs=max_running_requests,
                    full_max_total_num_tokens=full_max_total_num_tokens,
                    swa_max_total_num_tokens=swa_max_total_num_tokens,
                )
            else:
                # Fail loud, not silently fall through to the normal pools (which would
                # leave the flag a no-op). The feature replaces the HYBRID pools only.
                raise ValueError(
                    "--enable-unified-memory only supports hybrid Mamba and "
                    "hybrid sliding-window-attention models (DeepSeek-V4 excluded); "
                    f"the current model ({self.model_config.hf_config.architectures}) "
                    "is neither, so the unified memory pool cannot be built. Drop "
                    "--enable-unified-memory for this model."
                )
            return _InitializedPools(
                req_to_token_pool=bundle.req_to_token_pool,
                token_to_kv_pool=bundle.token_to_kv_pool,
                token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
                unified_memory_pool=bundle.unified_memory_pool,
            )
        max_num_reqs = max_running_requests

        # Initialize req_to_token_pool
        if req_to_token_pool is None:
            max_spec_draft_tokens = self.server_args.max_speculative_num_draft_tokens
            extra_max_context_len = get_req_to_token_extra_context_len(self.server_args)

            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # Extra slots for pre-allocated requests
                pre_alloc_size = self.server_args.disaggregation_decode_extra_slots
                if config := self.mambaish_config:
                    req_to_token_pool = HybridMambaDecodeReqToTokenPool(
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
                        speculative_num_draft_tokens=max_spec_draft_tokens,
                        speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                        pre_alloc_size=pre_alloc_size,
                        enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                        mamba_size=self.server_args.max_mamba_cache_size,
                        start_layer=self.start_layer,
                    )
                else:
                    req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := self.mambaish_config:
                req_to_token_pool = HybridReqToTokenPool(
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
                    enable_mamba_extra_buffer_lazy=self.server_args.enable_mamba_extra_buffer_lazy(),
                    speculative_num_draft_tokens=max_spec_draft_tokens,
                    speculative_eagle_topk=self.server_args.speculative_eagle_topk,
                    enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                    start_layer=self.start_layer,
                    enable_linear_replayssm=self.server_args.enable_linear_replayssm,
                    linear_replayssm_cache_len=self.server_args.linear_replayssm_cache_len,
                    mamba_envelope_layout=self.server_args.enable_page_major_kv_layout,
                )
            else:
                # DSV4 on NPU needs an extended ReqToTokenPool holding per-req
                # swa/c4/c128/c{4,128}_state tables; others stay on the stock one.
                req_to_token_pool_cls = ReqToTokenPool
                if _is_npu and is_deepseek_v4(self.model_config.hf_config):
                    from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import (
                        DSV4NPUReqToTokenPool,
                    )

                    req_to_token_pool_cls = DSV4NPUReqToTokenPool

                req_to_token_pool = req_to_token_pool_cls(
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
        is_dsa_model = is_deepseek_dsa(self.model_config.hf_config)
        is_dsv4_model = is_deepseek_v4(self.model_config.hf_config)

        self._validate_prefill_only_disable_kv_cache_pool_family(
            is_dsa_model, is_dsv4_model, current_platform
        )

        # Page-granularity envelope layout for the MHA-shaped (full / SWA) pools,
        # selected by swapping in the PageMajorMHATokenToKVPool subclass. The
        # default keeps upstream's per-layer layout. The Mamba state pool is routed
        # separately via `mamba_envelope_layout` on the req-to-token pool above.
        enable_page_major = self.server_args.enable_page_major_kv_layout
        mha_pool_class = (
            PageMajorMHATokenToKVPool if enable_page_major else MHATokenToKVPool
        )

        if is_dsv4_model:
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

            token_to_kv_pool = pool_cls(
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
        elif current_platform.is_out_of_tree() and not self.mambaish_config:
            if self.use_mla_backend and is_dsa_model:
                PoolCls = current_platform.get_dsa_kv_pool_cls()
                token_to_kv_pool = PoolCls(
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
            elif self.use_mla_backend:
                PoolCls = current_platform.get_mla_kv_pool_cls()
                token_to_kv_pool = PoolCls(
                    max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                PoolCls = current_platform.get_mha_kv_pool_cls()
                token_to_kv_pool = PoolCls(
                    max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif (
            self.server_args.attention_backend == "ascend" and not self.mambaish_config
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
                            // get_parallel().attn_tp_size,
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
                    post_capture_active=self.post_capture_kv_active,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    token_to_kv_pool_class=NPUMHATokenToKVPool,
                    **kwargs,
                )
            elif self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                token_to_kv_pool = NPUMLATokenToKVPool(
                    max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_dsa_model else None
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

                token_to_kv_pool = NPUMHATokenToKVPool(
                    max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_dsa_model:
            from sglang.srt.layers.cp.utils import get_glm_dsa_cp_layer_shard_info

            (
                dsa_cp_layer_shard_rank,
                dsa_cp_layer_shard_size,
            ) = get_glm_dsa_cp_layer_shard_info(self)
            pool_kwargs = {}
            if self.server_args.enable_hisparse:
                PoolCls = HiSparseDSATokenToKVPool
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                pool_kwargs["host_to_device_ratio"] = parse_hisparse_config(
                    self.server_args
                ).host_to_device_ratio
            elif dsa_cp_layer_shard_rank is not None:
                # DSA cache layer split: shard KV/indexer layers across CP ranks.
                from sglang.srt.mem_cache.dsa_cache_layer_split import (
                    LayerSplitDSATokenToKVPool,
                )

                PoolCls = LayerSplitDSATokenToKVPool
                pool_kwargs["layer_shard_rank"] = dsa_cp_layer_shard_rank
                pool_kwargs["layer_shard_size"] = dsa_cp_layer_shard_size
            else:
                PoolCls = DSATokenToKVPool
            token_to_kv_pool = PoolCls(
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
                            // get_parallel().attn_tp_size,
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
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    device=self.device,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    token_to_kv_pool_class=mha_pool_class,
                    **kwargs,
                )
            elif is_minimax_sparse(self.model_config.hf_config):
                _hf_config = self.model_config.hf_config
                sparse_cfg = get_minimax_sparse_attention_config(_hf_config)
                dense_layer_ids, sparse_layer_ids = get_minimax_sparse_layer_ids(
                    sparse_cfg
                )
                disable_value_sparse_layer_ids = (
                    get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
                )
                token_to_kv_pool = MiniMaxSparseKVPool(
                    size=max_total_num_tokens,
                    page_size=self.server_args.page_size,
                    dtype=self.kv_cache_dtype,
                    index_dtype=self.model_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_parallel().attn_tp_size
                    ),
                    head_dim=self.model_config.head_dim,
                    idx_head_dim=sparse_cfg["sparse_index_dim"],
                    dense_layer_ids=dense_layer_ids,
                    sparse_layer_ids=sparse_layer_ids,
                    disable_value_sparse_layer_ids=disable_value_sparse_layer_ids,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
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
                        get_parallel().attn_tp_size
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
                    device=self.device,
                    mamba_pool=req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    enable_kv_cache_copy=(
                        self.server_args.speculative_algorithm is not None
                    ),
                    use_mla=self.use_mla_backend,
                    start_layer=self.start_layer,
                    full_kv_pool_class=mha_pool_class,
                    post_capture_active=self.post_capture_kv_active,
                    **extra_args,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    assert (
                        not enable_page_major
                    ), "page-major KV layout is not supported with fp4 KV cache"
                    token_to_kv_pool = MHATokenToKVPoolFP4(
                        max_total_num_tokens,
                        page_size=self.server_args.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_parallel().attn_tp_size
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
                        else mha_pool_class
                    )
                    token_to_kv_pool = pool_cls(
                        max_total_num_tokens,
                        page_size=self.server_args.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_parallel().attn_tp_size
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
                        post_capture_active=self.post_capture_kv_active,
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
                if self.is_hybrid_swa and full_max_total_num_tokens == 0:
                    token_to_kv_pool_allocator = PureSWATokenToKVPoolAllocator(
                        swa_max_total_num_tokens,
                        page_size=self.server_args.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=token_to_kv_pool,
                        need_sort=need_sort,
                    )
                elif self.is_hybrid_swa:
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
                assert isinstance(swa_allocator, SWATokenToKVPoolAllocator)
                token_to_kv_pool.register_mapping(
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
        return _InitializedPools(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
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
