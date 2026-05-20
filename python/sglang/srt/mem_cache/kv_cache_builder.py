from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheBuildResult:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    sliding_window_size: Optional[int]
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    req_to_token_pool: object
    token_to_kv_pool_allocator: object
    disable_radix_cache: bool
    tree_cache: object


from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.session.streaming_session import StreamingSession

if TYPE_CHECKING:

    from torch.distributed import ProcessGroup

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def get_draft_kv_pool(
    *,
    draft_worker: "BaseTpWorker",
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    enable_overlap: bool,
):
    """Return (draft_token_to_kv_pool, draft_model_config) for the current
    draft worker, or (None, None) when no draft KV pool is available."""
    if draft_worker is None or spec_algorithm.is_ngram():
        return None, None

    if spec_algorithm.supports_spec_v2() and enable_overlap:
        if server_args.enable_multi_layer_eagle:
            draft_runner = draft_worker.draft_worker.draft_runner_list[0]
        else:
            draft_runner = draft_worker.draft_worker.draft_runner
        return draft_runner.token_to_kv_pool, draft_runner.model_config

    return (
        draft_worker.model_runner.token_to_kv_pool,
        draft_worker.model_config,
    )


def maybe_register_hicache_draft(
    *,
    tree_cache: "BasePrefixCache",
    draft_worker: "BaseTpWorker",
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    enable_hierarchical_cache: bool,
    enable_overlap: bool,
    page_size: int,
) -> None:
    """Register draft KV pool with HiCacheController for piggyback L2/L3 ops."""
    if not enable_hierarchical_cache:
        return

    draft_kv_pool, _ = get_draft_kv_pool(
        draft_worker=draft_worker,
        spec_algorithm=spec_algorithm,
        server_args=server_args,
        enable_overlap=enable_overlap,
    )
    if draft_kv_pool is None:
        return

    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MHATokenToKVPool,
        MLATokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool_host import (
        MHATokenToKVPoolHost,
        MLATokenToKVPoolHost,
    )

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        pool = pool.full_kv_pool

    # Create host pool for draft with the same slot count as the target host pool,
    # so that host indices stay 1-to-1 between target and draft KV caches.
    primary = tree_cache.cache_controller.mem_pool_host
    kw = dict(
        host_to_device_ratio=primary.size / pool.size,
        host_size=0,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
    )
    if isinstance(pool, MHATokenToKVPool):
        draft_host_pool = MHATokenToKVPoolHost(pool, **kw)
    elif isinstance(pool, MLATokenToKVPool):
        draft_host_pool = MLATokenToKVPoolHost(pool, **kw)
    else:
        logger.warning(
            "Draft pool type %s not supported for HiCache, skipping.",
            type(pool).__name__,
        )
        return

    tree_cache.cache_controller.set_draft_kv_pool(pool, draft_host_pool)


def build_kv_cache(
    *,
    server_args: "ServerArgs",
    model_config: "ModelConfig",
    tp_worker: "BaseTpWorker",
    page_size: int,
    spec_algorithm: "SpeculativeAlgorithm",
    attn_tp_cpu_group: "ProcessGroup",
    tp_cpu_group: "ProcessGroup",
    attn_cp_cpu_group: "ProcessGroup",
    enable_metrics: bool,
    enable_kv_cache_events: bool,
    ps: "ParallelState",
    tp_group: "GroupCoordinator",
    enable_hierarchical_cache: bool,
) -> "KVCacheBuildResult":
    sliding_window_size: Optional[int] = None
    full_tokens_per_layer: Optional[int] = None
    swa_tokens_per_layer: Optional[int] = None
    uses_transformers_backend = (
        get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
    )

    # Hybrid memory pool
    is_hybrid_swa = tp_worker.is_hybrid_swa
    _spec = tp_worker.model_runner.linear_attn_model_spec
    _registry_needs_mamba = _spec.uses_mamba_radix_cache if _spec is not None else False
    is_hybrid_ssm = (
        tp_worker.model_runner.hybrid_gdn_config is not None
        or tp_worker.model_runner.mamba2_config is not None
        or _registry_needs_mamba
    )

    sliding_window_size = None
    if is_hybrid_swa:
        sliding_window_size = tp_worker.sliding_window_size
        full_tokens_per_layer, swa_tokens_per_layer = (
            tp_worker.get_tokens_per_layer_info()
        )

    req_to_token_pool, token_to_kv_pool_allocator = tp_worker.get_memory_pool()

    disable_radix_cache = server_args.disable_radix_cache or (
        model_config.is_multimodal and uses_transformers_backend
    )
    if disable_radix_cache and not server_args.disable_radix_cache:
        logger.warning(
            "Radix cache is disabled for multimodal models with the "
            "Transformers backend to avoid multimodal prefix-cache mismatches."
        )

    # Decode radix cache is unsupported with hybrid SWA/SSM models —
    # these use specialized memory pools incompatible with the
    # prefix-match-and-lock allocation path.
    if (
        server_args.disaggregation_decode_enable_radix_cache
        and server_args.disaggregation_mode == "decode"
    ):
        if is_hybrid_swa:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is incompatible "
                "with sliding window attention (SWA) models"
            )
        if is_hybrid_ssm:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is incompatible "
                "with Mamba/SSM models"
            )

    effective_chunked_prefill_size = server_args.chunked_prefill_size
    if model_config.is_multimodal and uses_transformers_backend:
        effective_chunked_prefill_size = None

    params = CacheInitParams(
        disable=disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        is_eagle=spec_algorithm.is_eagle(),
        tp_cache_group=(
            attn_tp_cpu_group if server_args.enable_dp_attention else tp_cpu_group
        ),
        attn_cp_cache_group=attn_cp_cpu_group,
        attn_tp_cache_group=attn_tp_cpu_group,
        eviction_policy=server_args.radix_eviction_policy,
        enable_metrics=enable_metrics,
        enable_kv_cache_events=enable_kv_cache_events,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
        pp_rank=ps.pp_rank,
        pp_size=ps.pp_size,
        chunked_prefill_size=effective_chunked_prefill_size,
        sliding_window_size=sliding_window_size,
    )

    if effective_chunked_prefill_size is not None and disable_radix_cache:
        if not is_hybrid_swa:
            from sglang.srt.mem_cache.chunk_cache import ChunkCache

            tree_cache = ChunkCache(params)
        else:
            from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

            tree_cache = SWAChunkCache(params)
    else:
        if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
            # lazy import to avoid JIT overhead
            from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

            logger.info("Using experimental C++ radix tree implementation.")
            tree_cache = RadixCacheCpp(params=params, server_args=server_args)
        elif envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get():
            from sglang.srt.mem_cache.unified_cache_components import (
                ComponentType,
            )
            from sglang.srt.mem_cache.unified_radix_cache import (
                UnifiedRadixCache,
            )

            tree_components = [ComponentType.FULL]
            if is_hybrid_swa or is_hybrid_ssm:
                tree_components.append(
                    ComponentType.SWA if is_hybrid_swa else ComponentType.MAMBA
                )
            params.tree_components = tuple(tree_components)
            tree_cache = UnifiedRadixCache(params)
            if enable_hierarchical_cache:
                tree_cache.init_hicache(server_args, params)
                tp_worker.register_hicache_layer_transfer_counter(
                    tree_cache.cache_controller.layer_done_counter
                )
        elif enable_hierarchical_cache:
            if is_hybrid_ssm:
                from sglang.srt.mem_cache.hi_mamba_radix_cache import (
                    HiMambaRadixCache,
                )

                tree_cache = HiMambaRadixCache(params=params, server_args=server_args)
            else:
                from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

                tree_cache = HiRadixCache(params=params, server_args=server_args)
            tp_worker.register_hicache_layer_transfer_counter(
                tree_cache.cache_controller.layer_done_counter
            )
        elif is_hybrid_swa:
            from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

            tree_cache = SWARadixCache(params=params)
        elif is_hybrid_ssm:
            from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

            tree_cache = MambaRadixCache(params)
        elif server_args.enable_lmcache:
            from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                LMCRadixCache,
            )

            tree_cache = LMCRadixCache(
                params=params,
                model_config=model_config,
                tp_size=ps.tp_size,
                rank=ps.tp_rank,
                tp_group=tp_group,
            )
        else:
            tree_cache = RadixCache(params)

    if (
        server_args.enable_streaming_session
        and not tree_cache.supports_streaming_session()
    ):
        tree_cache = StreamingSession(tree_cache)

    embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
    init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    return KVCacheBuildResult(
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        sliding_window_size=sliding_window_size,
        full_tokens_per_layer=full_tokens_per_layer,
        swa_tokens_per_layer=swa_tokens_per_layer,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        disable_radix_cache=disable_radix_cache,
        tree_cache=tree_cache,
    )
