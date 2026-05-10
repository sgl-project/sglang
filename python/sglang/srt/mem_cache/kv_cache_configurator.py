from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig, is_deepseek_nsa
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, NSATokenToKVPool, ReqToTokenPool
from sglang.srt.utils.common import is_hip

_is_hip = is_hip()
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


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
        raise NotImplementedError("populated in kvc-migrate-method-bodies")


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
