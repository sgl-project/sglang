from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NSATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

logger = logging.getLogger(__name__)

_is_npu = is_npu()


class ModelRunnerMemoryMixin:
    def init_memory_pool(
        self: ModelRunner, total_gpu_memory: int, server_args: ServerArgs
    ):
        max_num_reqs = server_args.max_running_requests
        max_total_tokens = server_args.max_total_tokens
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
            if (
                self.server_args.enable_mamba_extra_buffer()
                and not self.spec_algorithm.is_none()
            ):
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

        if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
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
            self.set_num_token_hybrid()

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
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
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
                if self.page_size == 1:
                    if self.is_hybrid_swa:
                        self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                            self.full_max_total_num_tokens,
                            self.swa_max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                else:
                    assert not self.is_hybrid_swa
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
