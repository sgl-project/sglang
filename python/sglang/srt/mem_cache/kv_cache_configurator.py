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
    memory_pool_config: Optional["MemoryPoolConfig"]

    def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:
        raise NotImplementedError("populated in kvc-migrate-method-bodies")

    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.server_args.mem_fraction_static
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
            max_num_reqs = min(max_num_reqs // self.server_args.dp_size, estimated)
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
                self.server_args.dp_size if server_args.enable_dp_attention else 1
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
