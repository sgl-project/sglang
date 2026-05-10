from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import (
    is_deepseek_nsa,
    is_deepseek_v4,
)
from sglang.srt.mem_cache.memory_pool import (
    NSATokenToKVPool,
)
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


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

    def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        kv_cache_dtype = self.kv_cache_dtype
        kv_lora_rank = self.model_config.kv_lora_rank
        qk_rope_head_dim = self.model_config.qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

        # For non-NSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
        if not is_nsa_model:
            return kv_cache_dim

        # TRTLLM backend does not override kv_cache_dim for MLA kv cache
        # Assuming nsa prefill and decode backends are the same when using trtllm MLA backend,
        # since it is not compatible for trtllm and other mla attn backend due to the different
        # kv cache layout.
        if (
            self.server_args.nsa_prefill_backend == "trtllm"
            or self.server_args.nsa_decode_backend == "trtllm"
        ):
            return kv_cache_dim

        # On HIP with TileLang backend, keep the default MLA KV cache dimension.
        # FP8 attention uses the nope(512 fp8) + rope(64 fp8) layout, without extra per-block scales.
        if _is_hip and (
            self.server_args.nsa_prefill_backend == "tilelang"
            or self.server_args.nsa_decode_backend == "tilelang"
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

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        return self.kv_cache_configurator._calculate_mamba_ratio()

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

        (
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.token_to_kv_pool_allocator,
        ) = self.kv_cache_configurator._init_pools(
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            full_max_total_num_tokens=getattr(self, "full_max_total_num_tokens", None),
            swa_max_total_num_tokens=getattr(self, "swa_max_total_num_tokens", None),
            c4_max_total_num_tokens=self.c4_max_total_num_tokens,
            c128_max_total_num_tokens=self.c128_max_total_num_tokens,
            c4_state_pool_size=self.c4_state_pool_size,
            c128_state_pool_size=self.c128_state_pool_size,
            state_dtype=getattr(self, "state_dtype", None),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
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
