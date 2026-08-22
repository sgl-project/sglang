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

from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    linear_attn_model_spec,
    mamba2_config,
)
from sglang.srt.configs.model_config import ModelImpl, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.managers.mm_schedule import init_mm_embedding_cache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_memory,
    get_parallel,
    get_schedule,
)

if TYPE_CHECKING:

    from torch.distributed import ProcessGroup

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import HiCacheDraftPlan
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def maybe_register_hicache_draft(
    *,
    tree_cache,
    draft_plan: HiCacheDraftPlan,
    server_args: ServerArgs,
    page_size: int,
) -> None:
    from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode

    if draft_plan.mode != HiCacheDraftMode.SIDECAR:
        return

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if not isinstance(tree_cache, UnifiedRadixCache):
        _register_legacy_hicache_draft(
            tree_cache=tree_cache,
            draft_pool=draft_plan.device_pools[0],
            server_args=server_args,
            page_size=page_size,
        )
        return

    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        build_hicache_draft_sidecars,
    )

    specs, entries = build_hicache_draft_sidecars(
        draft_device_pools=draft_plan.device_pools,
        tree_cache=tree_cache,
        server_args=server_args,
    )
    tree_cache.register_hicache_draft_pools(specs, entries)


def _register_legacy_hicache_draft(
    *,
    tree_cache,
    draft_pool,
    server_args: ServerArgs,
    page_size: int,
) -> None:
    from sglang.srt.mem_cache.memory_pool import (
        MHATokenToKVPool,
        MLATokenToKVPool,
    )
    from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    pool = draft_pool
    if pool.layer_num == 0:
        return

    # Create host pool for draft with the same slot count as the target host pool,
    # so that host indices stay 1-to-1 between target and draft KV caches.
    primary_host_pool = tree_cache.cache_controller.mem_pool_host
    host_pool_kwargs = dict(
        host_to_device_ratio=primary_host_pool.logical_size / pool.size,
        host_size=0,
        page_size=page_size,
        layout=get_memory().hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        pool_label="draft",
    )
    if isinstance(pool, MHATokenToKVPool):
        draft_host_pool = get_mha_host_pool_cls(pool)(pool, **host_pool_kwargs)
    elif isinstance(pool, MLATokenToKVPool):
        draft_host_pool = MLATokenToKVPoolHost(pool, **host_pool_kwargs)
    else:
        logger.warning(
            "Draft pool type %s is not supported by the legacy HiCache path; "
            "skipping draft KV registration.",
            type(pool).__name__,
        )
        return

    tree_cache.cache_controller.set_draft_kv_pool(pool, draft_host_pool)


# Host slots a backup-only retraction pool gets, as a fraction of the device
# pool. Sized well under 1.0 because a retraction burst touches a fraction of
# the device tokens; overflow aborts the request rather than pre-reserving.
BACKUP_ONLY_HICACHE_RATIO = 0.2


def resolve_decode_retraction_backup(*, tp_worker: BaseTpWorker) -> str:
    """Resolve the retraction backend onto the config bags and return it.

    The backend needs the built KV pool, so it cannot resolve in
    ``ServerArgs.__post_init__``; it lands on the bags via ``override`` and
    every reader goes through ``get_disagg()`` / ``get_memory()``.
    """
    disagg = get_disagg()
    memory = get_memory()
    fields = {}

    backend = disagg.disaggregation_decode_retraction_backup
    if backend is None:
        kv_cache = tp_worker.get_memory_pool()[1].get_kvcache()
        full_tokens_per_layer = (
            tp_worker.get_tokens_per_layer_info()[0]
            if tp_worker.is_hybrid_swa
            else None
        )
        supports_host_pool = isinstance(kv_cache, MHATokenToKVPool) or (
            isinstance(kv_cache, SWAKVPool) and full_tokens_per_layer > 0
        )
        schedule = get_schedule()
        priority_preemption = (
            schedule.enable_priority_scheduling
            and not schedule.disable_priority_preemption
        )
        backend = (
            "host_pool"
            if disagg.disaggregation_mode == "decode"
            and not get_parallel().dcp_enabled
            and not disagg.disaggregation_decode_enable_radix_cache
            # KV offload already owns a host pool; a second one double-books host memory.
            and not disagg.disaggregation_decode_enable_offload_kvcache
            and not priority_preemption
            and supports_host_pool
            else "cpu_tensor"
        )
        fields["disaggregation_decode_retraction_backup"] = backend

    if memory.hicache_ratio is None:
        # Only a decode server reaches resolution with the ratio unset. A
        # backup-only pool can be small: retractions that overflow it abort their
        # request instead of crashing the scheduler. Sharing the pool with
        # HiCache keeps the standard default.
        if backend == "host_pool" and not memory.enable_hierarchical_cache:
            fields["hicache_ratio"] = BACKUP_ONLY_HICACHE_RATIO
        else:
            fields["hicache_ratio"] = 2.0

    source = "kv_cache_builder.decode_retraction"
    get_context().override(source, **fields)
    return backend


def build_kv_cache(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    tp_worker: BaseTpWorker,
    page_size: int,
    spec_algorithm: SpeculativeAlgorithm,
    attn_tp_cpu_group: ProcessGroup,
    tp_cpu_group: ProcessGroup,
    attn_cp_cpu_group: ProcessGroup,
    enable_metrics: bool,
    enable_kv_cache_events: bool,
    ps: ParallelState,
    tp_group: GroupCoordinator,
    pp_group: GroupCoordinator,
    enable_hierarchical_cache: bool,
    hicache_draft_plan: Optional[HiCacheDraftPlan] = None,
) -> KVCacheBuildResult:
    sliding_window_size: Optional[int] = None
    full_tokens_per_layer: Optional[int] = None
    swa_tokens_per_layer: Optional[int] = None
    uses_transformers_backend = (
        get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
    )

    # Hybrid memory pool
    is_hybrid_swa = tp_worker.is_hybrid_swa
    _spec = linear_attn_model_spec(tp_worker.model_runner.model_config)
    _registry_needs_mamba = _spec.uses_mamba_radix_cache if _spec is not None else False
    is_hybrid_ssm = (
        hybrid_gdn_config(tp_worker.model_runner.model_config) is not None
        or mamba2_config(tp_worker.model_runner.model_config) is not None
        or _registry_needs_mamba
        or kimi_linear_config(tp_worker.model_runner.model_config) is not None
        or hybrid_lightning_config(tp_worker.model_runner.model_config) is not None
    )
    is_dsa = is_deepseek_dsa(model_config.hf_config)

    sliding_window_size = None
    if is_hybrid_swa:
        sliding_window_size = tp_worker.sliding_window_size
        full_tokens_per_layer, swa_tokens_per_layer = (
            tp_worker.get_tokens_per_layer_info()
        )

    req_to_token_pool, token_to_kv_pool_allocator = tp_worker.get_memory_pool()
    mtp_draft_device_pools = tp_worker.model_runner.mtp_draft_device_pools

    retraction_backup = resolve_decode_retraction_backup(tp_worker=tp_worker)

    disable_radix_cache = get_memory().disable_radix_cache or (
        model_config.is_multimodal and uses_transformers_backend
    )
    if disable_radix_cache and not get_memory().disable_radix_cache:
        logger.warning(
            "Radix cache is disabled for multimodal models with the "
            "Transformers backend to avoid multimodal prefix-cache mismatches."
        )

    # Decode-side radix cache supports SWA only through the unified tree, whose
    # component pools preserve the full-attention prefix while transferring the
    # SWA window fresh. The legacy SWA cache and hybrid SSM pools remain
    # incompatible with the prefix-match-and-lock allocation path.
    if (
        get_disagg().disaggregation_decode_enable_radix_cache
        and get_disagg().disaggregation_mode == "decode"
    ):
        if is_hybrid_swa:
            if not (envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get() or use_mlx()):
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache with sliding "
                    "window attention (SWA) models requires the unified radix "
                    "tree (set SGLANG_ENABLE_UNIFIED_RADIX_TREE=1)."
                )
            if enable_hierarchical_cache:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache with sliding "
                    "window attention (SWA) models currently supports only "
                    "device-resident cache and is incompatible with "
                    "--enable-hierarchical-cache."
                )
            if getattr(model_config, "is_deepseek_v4_arch", False):
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache does not support "
                    "DeepSeek-V4 (DSA) compressed KV (c4/c128/indexer) yet."
                )
            if getattr(model_config, "is_hybrid_swa_compress", False):
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache does not support "
                    "SWA-compress models (e.g. Gemma4 / MiMo-V2) yet."
                )
        if is_hybrid_ssm:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is incompatible "
                "with Mamba/SSM models"
            )

    effective_chunked_prefill_size = get_schedule().chunked_prefill_size
    if model_config.is_multimodal and uses_transformers_backend:
        effective_chunked_prefill_size = None

    params = CacheInitParams(
        disable=disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        # When dcp enabled, kv_pool_allocator.page_size is page_size * dcp_size.
        # TreeCache.page_size should keep the same as allocator.page_size to
        # avoid kv page eviction conflicts.
        page_size=(
            page_size
            if not get_parallel().dcp_enabled
            else token_to_kv_pool_allocator.page_size
        ),
        is_eagle=spec_algorithm.is_eagle(),
        tp_cache_group=(
            attn_tp_cpu_group if get_parallel().enable_dp_attention else tp_cpu_group
        ),
        attn_cp_cache_group=attn_cp_cpu_group,
        attn_tp_cache_group=attn_tp_cpu_group,
        pp_cache_group=pp_group.cpu_group,
        eviction_policy=get_memory().radix_eviction_policy,
        enable_metrics=enable_metrics,
        enable_kv_cache_events=enable_kv_cache_events,
        enable_session_radix_cache=get_memory().enable_session_radix_cache,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
        enable_mamba_extra_buffer_lazy=server_args.enable_mamba_extra_buffer_lazy(),
        pp_rank=ps.pp_rank,
        pp_size=ps.pp_size,
        chunked_prefill_size=effective_chunked_prefill_size,
        sliding_window_size=sliding_window_size,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )

    tree_cache = create_tree_cache(
        TreeCacheBuildContext(
            server_args=server_args,
            params=params,
            is_hybrid_swa=is_hybrid_swa,
            full_tokens_per_layer=full_tokens_per_layer,
            is_hybrid_ssm=is_hybrid_ssm,
            is_dsa=is_dsa,
            enable_hierarchical_cache=enable_hierarchical_cache,
            disable_radix_cache=disable_radix_cache,
            effective_chunked_prefill_size=effective_chunked_prefill_size,
            tp_worker=tp_worker,
            model_config=model_config,
            tp_size=ps.tp_size,
            tp_rank=ps.tp_rank,
            tp_group=tp_group,
        )
    )

    if (
        enable_hierarchical_cache or retraction_backup == "host_pool"
    ) and hicache_draft_plan is not None:
        maybe_register_hicache_draft(
            tree_cache=tree_cache,
            draft_plan=hicache_draft_plan,
            server_args=server_args,
            page_size=page_size,
        )

    if retraction_backup == "host_pool":
        if not isinstance(tree_cache, UnifiedRadixCache):
            raise ValueError(
                "--disaggregation-decode-retraction-backup=host_pool requires "
                "UnifiedRadixCache with HiCache attached."
            )
        tree_cache.validate_retraction_host_capacity()

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
