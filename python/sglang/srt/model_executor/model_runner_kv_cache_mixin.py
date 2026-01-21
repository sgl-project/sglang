from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MHATokenToKVPoolMixedPrecision,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    MLATokenToKVPoolMixedPrecision,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

logger = logging.getLogger(__name__)

_is_npu = is_npu()


class ModelRunnerKVCacheMixin:
    def _calculate_single_layer_cell_size(
        self: ModelRunner, layer_id: int, layer_dtype: torch.dtype
    ) -> int:
        """Calculate cell size for a single layer with given dtype."""
        kv_size = torch._utils._element_size(layer_dtype)

        if self.use_mla_backend:
            cell_size = (
                self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim
            ) * kv_size
            if is_float4_e2m1fn_x2(layer_dtype):
                # kv_scale_buffer for FP4
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (
                            self.model_config.kv_lora_rank
                            + self.model_config.qk_rope_head_dim
                        )
                        // scale_block_size
                    )
                    * kv_size
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(self.model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(self.model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * element_size
        else:
            # MHA backend
            cell_size = (
                self.model_config.get_num_kv_heads(get_attention_tp_size())
                * (self.model_config.head_dim + self.model_config.v_head_dim)
                * kv_size
            )

            if is_float4_e2m1fn_x2(layer_dtype):
                # kv_scale_buffer for FP4
                scale_block_size = 16
                n = self.model_config.get_num_kv_heads(get_attention_tp_size())
                k = self.model_config.head_dim
                v_k = self.model_config.v_head_dim
                # Base KV cache (packed): (n * (k + v_k)) * kv_size // 2
                # Scale buffers: (n * k) // scale_block_size * kv_size + (n * v_k) // scale_block_size * kv_size
                cell_size = (cell_size // 2) + (
                    ((n * k) // scale_block_size + (n * v_k) // scale_block_size)
                    * kv_size
                )

            # SWA layers (only for specific architectures)
            if (
                "MiMoV2FlashForCausalLM" in self.model_config.hf_config.architectures
                and layer_id in self.model_config.swa_attention_layer_ids
            ):
                cell_size += (
                    self.model_config.get_swa_num_kv_heads(get_attention_tp_size())
                    * (
                        self.model_config.hf_text_config.swa_head_dim
                        + self.model_config.hf_text_config.swa_v_head_dim
                    )
                    * kv_size
                )

        return cell_size

    def get_cell_size_per_token(
        self: ModelRunner,
        num_layers: int,
        per_layer_dtypes: Optional[Dict[int, torch.dtype]] = None,
    ) -> int:
        """Calculate cell size per token.

        If per_layer_dtypes is provided, calculates layer-by-layer since each layer
        may have different precision. Otherwise, uses homogeneous calculation.
        """
        if per_layer_dtypes is not None:
            # Calculate layer-by-layer for heterogeneous precision
            cell_size = 0
            for layer_id in range(self.start_layer, self.start_layer + num_layers):
                layer_dtype = per_layer_dtypes.get(layer_id, self.kv_cache_dtype)
                layer_cell_size = self._calculate_single_layer_cell_size(
                    layer_id, layer_dtype
                )
                cell_size += layer_cell_size
            return cell_size

        # Original homogeneous calculation
        kv_size = torch._utils._element_size(self.kv_cache_dtype)
        if self.use_mla_backend:
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (
                            self.model_config.kv_lora_rank
                            + self.model_config.qk_rope_head_dim
                        )
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(self.model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(self.model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(get_attention_tp_size())
                * (self.model_config.head_dim + self.model_config.v_head_dim)
                * num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16

                n = self.model_config.get_num_kv_heads(get_attention_tp_size())
                k = self.model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )

            if "MiMoV2FlashForCausalLM" in self.model_config.hf_config.architectures:
                cell_size += (
                    self.model_config.get_swa_num_kv_heads(get_attention_tp_size())
                    * (
                        self.model_config.hf_text_config.swa_head_dim
                        + self.model_config.hf_text_config.swa_v_head_dim
                    )
                    * len(self.model_config.swa_attention_layer_ids)
                    * kv_size
                )
        return cell_size

    def profile_max_num_token(self: ModelRunner, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        # Get the number of layers used for KV cache calculation
        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif mambaish := self.mambaish_config:
            num_layers = len(mambaish.full_attention_layer_ids)
        else:
            num_layers = self.num_effective_layers

        per_layer_dtypes = getattr(self, "per_layer_kv_cache_dtypes", None)
        cell_size = self.get_cell_size_per_token(
            num_layers, per_layer_dtypes=per_layer_dtypes
        )

        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)

        return int(rest_memory * (1 << 30)) // cell_size

    def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
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

        if (
            server_args.disable_radix_cache
            or server_args.max_mamba_cache_size is not None
        ):
            # with disable radix cache, sets the max_mamba_cache_size based on the max_running_requests
            if server_args.max_mamba_cache_size is None:
                if server_args.max_running_requests is not None:
                    server_args.max_mamba_cache_size = server_args.max_running_requests
                else:
                    server_args.max_mamba_cache_size = 512
            server_args.max_mamba_cache_size = server_args.max_mamba_cache_size // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        else:
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

    def set_num_tokens_hybrid_swa(self: ModelRunner):
        page_size = self.server_args.page_size
        if "MiMoV2MTP" in self.model_config.hf_config.architectures:
            assert self.is_draft_worker
            # MiMoV2MTP uses SWA, so set full KV cache to 0
            self.full_max_total_num_tokens = 0
            self.swa_max_total_num_tokens = (
                self.max_total_num_tokens // page_size * page_size
            )
            self.max_total_num_tokens = self.swa_max_total_num_tokens
        else:
            assert self.sliding_window_size is not None and self.sliding_window_size > 0
            full_layers_num = len(self.model_config.full_attention_layer_ids)
            swa_layers_num = len(self.model_config.swa_attention_layer_ids)

            # Algorithm:
            # Existing max_total_num_tokens is per layer and assume all layers have the same number of tokens.
            # - Find total # of tokens available across layers.
            # - Calculate full_max_total_num_tokens and swa_max_total_num_tokens based on the given swa_full_tokens_ratio.
            total_tokens = (
                self.max_total_num_tokens * self.model_config.num_hidden_layers
            )
            swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

            # Solve the equations:
            # 1. swa_max_total_num_tokens * swa_layers_num + full_max_total_num_tokens * full_layers_num == total_tokens
            # 2. full_max_total_num_tokens * swa_full_tokens_ratio == swa_max_total_num_tokens
            denominator = swa_full_tokens_ratio * swa_layers_num + full_layers_num
            self.full_max_total_num_tokens = int(total_tokens / denominator)
            self.swa_max_total_num_tokens = int(
                self.full_max_total_num_tokens * swa_full_tokens_ratio
            )

            self.full_max_total_num_tokens = (
                self.full_max_total_num_tokens // page_size * page_size
            )
            self.swa_max_total_num_tokens = (
                self.swa_max_total_num_tokens // page_size * page_size
            )

            self.max_total_num_tokens = self.full_max_total_num_tokens

        logger.info(
            f"Use sliding window memory pool. full_layer_tokens={self.full_max_total_num_tokens}, swa_layer_tokens={self.swa_max_total_num_tokens}"
        )

    def init_memory_pool(self: ModelRunner, total_gpu_memory: int):
        max_num_reqs = self.server_args.max_running_requests
        max_total_tokens = self.server_args.max_total_tokens
        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        if self.mambaish_config is not None:
            additional_ratio = 0
            if self.server_args.enable_mamba_extra_buffer():
                if not self.spec_algorithm.is_none():
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP
                else:
                    additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            if self.server_args.disable_radix_cache:
                ratio = 1
            else:
                ratio = MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO + additional_ratio
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )
            # for dp attention, we need control the max_num_reqs for speculative decoding mamba space
            if (
                not self.spec_algorithm.is_none()
                and self.server_args.enable_dp_attention
            ):
                max_num_reqs = min(
                    max_num_reqs, self.server_args.max_running_requests // self.dp_size
                )

        if not self.spec_algorithm.is_none():
            if self.is_draft_worker:
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                max_num_reqs = self.server_args.max_num_reqs
            else:
                self.server_args.draft_runner_cache_size = self.max_total_num_tokens
                self.server_args.max_num_reqs = max_num_reqs

        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        self.max_total_num_tokens = (
            self.max_total_num_tokens
            // self.server_args.page_size
            * self.server_args.page_size
        )
        # different pp rank may have different num of layers, so we need to reduce the max_total_num_tokens
        if self.pp_size > 1:
            tensor = torch.tensor(self.max_total_num_tokens, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            self.max_total_num_tokens = tensor.item()

        # create token size for hybrid cache
        if self.is_hybrid_swa:
            self.set_num_tokens_hybrid_swa()

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                f"Not enough memory. Please try to increase --mem-fraction-static. "
                f"Current value: {self.server_args.mem_fraction_static=}"
            )

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
                pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0
                if config := self.mambaish_config:
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                        pre_alloc_size=pre_alloc_size,
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
            elif config := self.mambaish_config:
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    mamba_spec_state_size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                    speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
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
        if self.server_args.attention_backend == "ascend":
            if self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=self.model_config.index_head_dim,
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
            self.token_to_kv_pool = NSATokenToKVPool(
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
                index_head_dim=get_nsa_index_head_dim(self.model_config.hf_config),
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_nsa_model
            use_mixed_precision = False
            if self.per_layer_kv_cache_dtypes is not None:
                unique_dtypes = set(self.per_layer_kv_cache_dtypes.values())
                use_mixed_precision = len(unique_dtypes) > 1

            if use_mixed_precision:
                self.token_to_kv_pool = MLATokenToKVPoolMixedPrecision(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    per_layer_dtypes=self.per_layer_kv_cache_dtypes,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            elif is_float4_e2m1fn_x2(self.kv_cache_dtype):
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
        elif self.server_args.enable_double_sparsity:
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
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
            elif config := self.mambaish_config:
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
                        [0] if self.is_draft_worker else config.full_attention_layer_ids
                    ),
                    enable_kvcache_transpose=False,
                    device=self.device,
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    use_mla=self.use_mla_backend,
                    **extra_args,
                )
            else:
                # Check if we need heterogeneous pool (multiple dtypes)
                per_layer_dtypes = getattr(self, "per_layer_kv_cache_dtypes", None)
                use_heterogeneous = (
                    per_layer_dtypes is not None
                    and len(set(per_layer_dtypes.values())) > 1
                )

                if use_heterogeneous:
                    # Use mixed precision pool
                    self.token_to_kv_pool = MHATokenToKVPoolMixedPrecision(
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
                        per_layer_dtypes=per_layer_dtypes,
                        v_head_dim=self.model_config.v_head_dim,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )
                elif is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
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
            if _is_npu and (
                self.server_args.attention_backend == "ascend"
                or self.hybrid_gdn_config is not None
            ):
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
                    if self.page_size == 1:
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

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                assert (
                    self.token_to_kv_pool_allocator.__class__
                    == SWATokenToKVPoolAllocator
                )
                self.token_to_kv_pool.full_to_swa_index_mapping = (
                    self.token_to_kv_pool_allocator.full_to_swa_index_mapping
                )

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
