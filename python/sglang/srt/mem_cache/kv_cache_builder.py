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
from sglang.srt.managers.mm_schedule import init_mm_embedding_cache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode


def _device_pool_size_per_token(pool: object) -> float:
    """Bytes per token for a device KV pool, derived from its geometry.

    The host pool stores the same per-token layout as the device pool, so this
    gives the host pool's ``size_per_token`` without instantiating it.

    We cannot use ``get_kv_size_bytes() / pool.size`` because the device pool
    may be a no-op placeholder (KVarN post-capture allocation) whose buffer is
    tiny.  Instead, compute from the pool's structural attributes.
    """
    # KVarN no-op pool: compute from the KVarN backend reference.
    kvarn_backend = getattr(pool, "_kvarn_backend", None)
    if kvarn_backend is not None:
        tile_bytes = kvarn_backend.cfg.tile_bytes_aligned
        hk = kvarn_backend.num_kv_heads
        num_layers = kvarn_backend.num_layers
        page_size = kvarn_backend.page_size
        if page_size > 0:
            return hk * tile_bytes * num_layers // page_size

    # MHA pools: head_dim * head_num * layer_num * dtype_itemsize * 2 (K+V)
    head_num = getattr(pool, "head_num", None)
    head_dim = getattr(pool, "head_dim", None)
    layer_num = getattr(pool, "layer_num", None)
    dtype = getattr(pool, "store_dtype", None) or getattr(pool, "dtype", None)
    if (
        head_num is not None
        and head_dim is not None
        and layer_num is not None
        and dtype is not None
    ):
        itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 2
        return head_dim * head_num * layer_num * itemsize * 2

    # MLA pools: kv_cache_dim * layer_num * dtype_itemsize
    kv_cache_dim = getattr(pool, "kv_cache_dim", None)
    if kv_cache_dim is not None and layer_num is not None and dtype is not None:
        itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 2
        return kv_cache_dim * layer_num * itemsize

    # Fallback: derive from total buffer size (works only if buffers are allocated)
    size_bytes = pool.get_kv_size_bytes()
    total_bytes = sum(size_bytes) if isinstance(size_bytes, tuple) else size_bytes
    if pool.size == 0:
        return 0.0
    return total_bytes / pool.size


def _adjust_hicache_size_for_draft(
    server_args: ServerArgs,
    target_device_pool: object,
    draft_device_pools: tuple[object, ...],
) -> None:
    """Reduce ``hicache_size`` so target + draft host pools fit the GB budget.

    The draft host pool must have the same token count as the target (for 1-to-1
    index sharing in L2 transfers).  When ``--hicache-size`` caps the target
    pool to *H* GB, the draft pool adds ``H * draft_spt / target_spt`` GB on top.
    To keep the *total* within *H* GB, shrink the budget by the factor
    ``target_spt / (target_spt + draft_spt)``.

    Only applies in fixed-size (``hicache_size > 0``) mode; ratio mode already
    scales both pools proportionally.
    """
    if server_args.hicache_size <= 0:
        return
    if not draft_device_pools:
        return

    target_spt = _device_pool_size_per_token(target_device_pool)
    if target_spt == 0:
        return

    # Sum the per-token cost of every draft device pool that will get a
    # matching host pool.  In SIDECAR mode only the first draft runner is
    # registered, so we use just its pool.
    draft_spt = _device_pool_size_per_token(draft_device_pools[0])
    if draft_spt == 0:
        return

    effective = server_args.hicache_size * target_spt / (target_spt + draft_spt)
    if effective < server_args.hicache_size:
        logger.info(
            "Adjusting --hicache-size from %d GB to %.2f GB to account for "
            "draft host pool memory (target_spt=%.0f, draft_spt=%.0f).",
            server_args.hicache_size,
            effective,
            target_spt,
            draft_spt,
        )
        # Use the resolution-pipeline helper to bypass the strict post-publish
        # mutation guard on ServerArgs.  This is a one-shot, pre-tree-cache
        # adjustment: after create_tree_cache() returns, hicache_size is never
        # read again, so the instance and the memory bag stay in agreement.
        from sglang.srt.arg_groups.overrides import _apply_fields

        # Reduce both hicache_size AND hicache_ratio.  When the device pool is
        # a no-op placeholder (KVarN post-capture allocation), get_kv_size_bytes
        # returns 0, so _split_hicache_size gives the KV pool 0 GB and the host
        # pool falls back to hicache_ratio.  Reducing the ratio too ensures the
        # fallback also respects the draft budget.
        ratio_factor = target_spt / (target_spt + draft_spt)
        effective_ratio = server_args.hicache_ratio * ratio_factor
        _apply_fields(
            server_args,
            {
                "hicache_size": int(effective),
                "hicache_ratio": effective_ratio,
            },
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
        host_to_device_ratio=primary_host_pool.size / pool.size,
        host_size=0,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
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
            attn_tp_cpu_group if server_args.enable_dp_attention else tp_cpu_group
        ),
        attn_cp_cache_group=attn_cp_cpu_group,
        attn_tp_cache_group=attn_tp_cpu_group,
        pp_cache_group=pp_group.cpu_group,
        eviction_policy=server_args.radix_eviction_policy,
        enable_metrics=enable_metrics,
        enable_kv_cache_events=enable_kv_cache_events,
        enable_session_radix_cache=server_args.enable_session_radix_cache,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
        enable_mamba_extra_buffer_lazy=server_args.enable_mamba_extra_buffer_lazy(),
        pp_rank=ps.pp_rank,
        pp_size=ps.pp_size,
        chunked_prefill_size=effective_chunked_prefill_size,
        sliding_window_size=sliding_window_size,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )

    # When --hicache-size is set and a SIDECAR draft will be registered,
    # shrink the budget so the *total* (target + draft) host memory stays
    # within the user-specified cap.  The draft host pool must match the
    # target's token count (1-to-1 index sharing), but its per-token byte
    # cost can be larger than the target's (e.g. quantized target + bf16
    # draft), causing the combined allocation to exceed --hicache-size.
    if (
        enable_hierarchical_cache
        and hicache_draft_plan is not None
        and hicache_draft_plan.mode == HiCacheDraftMode.SIDECAR
    ):
        target_device_pool = tp_worker.model_runner.token_to_kv_pool
        _adjust_hicache_size_for_draft(
            server_args,
            target_device_pool,
            hicache_draft_plan.device_pools,
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

    if enable_hierarchical_cache and hicache_draft_plan is not None:
        maybe_register_hicache_draft(
            tree_cache=tree_cache,
            draft_plan=hicache_draft_plan,
            server_args=server_args,
            page_size=page_size,
        )

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
