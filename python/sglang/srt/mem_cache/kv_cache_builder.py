from __future__ import annotations

import logging

import psutil
import torch

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional, Sequence


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
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import HICACHE_HOST_MEMORY_RESERVE_BYTES
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.runtime_context import (
    configured_pp_size,
    configured_tp_size,
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


def _local_scheduler_count() -> int:
    """Schedulers sharing this node, mirroring the launcher's own arithmetic.

    ``entrypoints/engine.py`` derives the per-node rank ranges as
    ``pp_size_per_node * tp_size_per_node`` with ``tp_size_per_node = tp_size //
    nnodes_per_pp_rank``. The divisibility invariant is ``(tp_size * pp_size) %
    nnodes == 0``, so ``tp_size // nnodes`` undercounts whenever ``pp_size >
    1``. Classic data parallelism (``dp_size > 1`` without attention DP) instead
    launches ``dp_size`` independent replicas, each owning its own KV pool and
    therefore its own host mirror.
    """
    parallel = get_parallel()
    nnodes = max(1, parallel.nnodes)
    pp_size = max(1, configured_pp_size())
    tp_size = max(1, configured_tp_size())

    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    tp_size_per_node = max(tp_size // nnodes_per_pp_rank, 1)
    replicas = 1 if parallel.enable_dp_attention else max(1, parallel.dp_size)
    return pp_size_per_node * tp_size_per_node * replicas


def _agree_across_ranks(fits: bool) -> bool:
    """Reduce a per-rank verdict to one the whole job shares.

    Divergence is the hazard this exists to remove: a rank that keeps
    ``host_pool`` builds a ``UnifiedRadixCache`` with a host pool while a rank
    that falls back builds a ``ChunkCache``, leaving one job with two cache
    topologies. Follows ``sync_fixed_hicache_size``'s shape -- all-reduce MIN
    over a CPU group, degrading to the local verdict when there is no process
    group to reduce over.

    Callers must reach this on every rank, never behind a per-rank predicate:
    a rank that skips the reduction leaves the others blocked in the
    all-reduce, converting a mismatch into a hang.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return fits
    try:
        from sglang.srt.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
    except (AssertionError, RuntimeError):
        return fits
    if cpu_group is None:
        return fits

    tensor = torch.tensor(int(fits), dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
    return bool(tensor.item())


def _kv_pool_bytes(pool: object) -> int:
    """Total bytes held by a KV pool, across both shapes of the accessor.

    ``get_kv_size_bytes`` is not a uniform contract: MHA and SWA pools return a
    ``(k, v)`` pair while MLA, DSA, and Mamba pools return a scalar. Callers that
    see pools of every class have to normalise, so keep that knowledge in one
    place rather than assuming the tuple shape.
    """
    size = pool.get_kv_size_bytes()
    return sum(size) if isinstance(size, tuple) else int(size)


def _host_pool_retraction_fits(
    kv_cache: object, mtp_draft_device_pools: Sequence[object]
) -> tuple[bool, int, int]:
    """Whether host RAM can back a host-pool retraction mirror on this node.

    ``host_pool`` mirrors the device KV pool 1:1 (``hicache_ratio`` 1.0) and
    every scheduler on the node allocates its own mirror, so the node
    requirement is the per-rank device bytes times the local scheduler count.
    The host mirror also packs the MTP/EAGLE draft caches into its own
    ``layer_num`` (``MHATokenToKVPoolHost.get_size_per_token``), so the draft
    device pools count toward the requirement too.

    Measured against ``virtual_memory().available`` -- the same quantity the
    real gate in ``HostKVCache.__init__`` reads. The verdict is local; callers
    reduce it with ``_agree_across_ranks`` so ranks cannot disagree.

    This is a lower bound, not the exact allocation: it does not model the host
    pool's page round-up nor SIDECAR draft host pools sized off
    ``host_pool_group.size``, and it counts device-side scale buffers the host
    mirror may not carry. A configuration that clears it can still fail the
    per-rank gate, which stays authoritative. The purpose is to turn the
    arithmetically-unsatisfiable case into a fallback rather than a crash.
    """
    per_rank_bytes = _kv_pool_bytes(kv_cache)
    for draft_pool in mtp_draft_device_pools or ():
        per_rank_bytes += _kv_pool_bytes(draft_pool)

    required_bytes = per_rank_bytes * _local_scheduler_count()
    available_bytes = (
        psutil.virtual_memory().available - HICACHE_HOST_MEMORY_RESERVE_BYTES
    )
    return required_bytes <= available_bytes, required_bytes, available_bytes


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
    # Drives the hicache_ratio default below; the capacity gate may move
    # ``backend`` off host_pool but must not move the ratio with it.
    ratio_backend = backend
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
        # Split the eligibility test by where its inputs come from. These five
        # read the config bags, so every rank evaluates them identically and
        # they can gate a collective. ``supports_host_pool`` cannot: pool class
        # and full-token capacity are per-rank, and gating the all-reduce on
        # per-rank state lets one rank skip it and hang its peers.
        host_pool_eligible = (
            disagg.disaggregation_mode == "decode"
            and not get_parallel().dcp_enabled
            and not disagg.disaggregation_decode_enable_radix_cache
            # KV offload already owns a host pool; a second one double-books host memory.
            and not disagg.disaggregation_decode_enable_offload_kvcache
            and not priority_preemption
        )
        backend = (
            "host_pool" if host_pool_eligible and supports_host_pool else "cpu_tensor"
        )
        ratio_backend = backend
        # Supported is not the same as affordable: a large device pool mirrored
        # across every scheduler on the node can exceed host RAM outright
        # (common when the HBM:DRAM ratio is high). Only an explicit
        # --disaggregation-decode-retraction-backup=host_pool should hard-fail
        # on that; an inferred default degrades instead of refusing to start.
        #
        # Keyed off the config-uniform half of the eligibility test, so a rank
        # whose pool cannot host the mirror still reaches the reduction and
        # cannot strand its peers in the all-reduce. Every other server --
        # prefill, embedding, non-PD -- skips the estimate entirely rather than
        # paying a psutil read and a world-group all-reduce it has no use for.
        if host_pool_eligible:
            local_schedulers = _local_scheduler_count()
            local_fits, required_bytes, available_bytes = _host_pool_retraction_fits(
                kv_cache, tp_worker.model_runner.mtp_draft_device_pools
            )
            fits = _agree_across_ranks(local_fits)
        else:
            local_schedulers, required_bytes, available_bytes, fits = 0, 0, 0, True
        if backend == "host_pool" and not fits:
            logger.warning(
                "Falling back to cpu_tensor retraction backup: host-pool "
                "retraction needs at least %.2f GB of host memory (device "
                "pool mirrored across %d scheduler(s) on this node) but "
                "only %.2f GB is available. Pass "
                "--disaggregation-decode-retraction-backup=host_pool to "
                "require it, or shrink the device pool with "
                "--max-total-tokens / --mem-fraction-static.",
                required_bytes / 1e9,
                local_schedulers,
                available_bytes / 1e9,
            )
            backend = "cpu_tensor"
        fields["disaggregation_decode_retraction_backup"] = backend

    if memory.hicache_ratio is None:
        # Only a decode server reaches resolution with the ratio unset; host-pool
        # retraction sizes the host pool 1:1 with the device pool, everything
        # else keeps the standard default. Keyed off the pre-fallback verdict so
        # the capacity gate cannot raise the ratio: enlarging the host pool
        # because host memory is short would be exactly backwards.
        fields["hicache_ratio"] = 1.0 if ratio_backend == "host_pool" else 2.0

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

    # Decode radix cache is unsupported with hybrid SWA/SSM models —
    # these use specialized memory pools incompatible with the
    # prefix-match-and-lock allocation path.
    if (
        get_disagg().disaggregation_decode_enable_radix_cache
        and get_disagg().disaggregation_mode == "decode"
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
