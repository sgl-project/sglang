"""Build the unified KV pool stack from declarations: plan, size, then construct.

`build_unified_pools` is the one entry point. Every plan gets one
`ChainAllocator` over its sub-pools; which KV facade it gets still depends on
its set of lifetimes, the `_BUILDERS` table below.
"""

import logging
from typing import Any, Optional, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.decl import (
    FullLifetime,
    KVLayerDecl,
    StateLifetime,
    WindowLifetime,
)
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.plan import KVStoragePlan, plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import (
    KVSizingInputs,
    SizedKVStorage,
    size_kv_storage,
)
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.kv.storage import KVStorage
from sglang.srt.mem_cache.layout.fused_draft import FusedDraftPlacement
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.unified_memory_pool import (
    GB,
    UnifiedHybridLinearKVPool,
    UnifiedHybridReqToTokenPool,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMHATokenToKVPool,
    UnifiedMLATokenToKVPool,
    UnifiedPoolBundle,
    UnifiedSWAKVPool,
    UnifiedSWAPoolBundle,
    _wire_mamba_slot_allocator,
)

logger = logging.getLogger(__name__)


class UnifiedBuildContext(msgspec.Struct, frozen=True, kw_only=True):
    """What the constructed objects need beyond the plan and its sizing."""

    device: str
    start_layer: int
    end_layer: int
    enable_memory_saver: bool
    need_sort: bool
    lazy_compaction: bool = False
    forward_stream: Optional[Any] = None  # torch.cuda.Stream
    post_capture_active: bool = False
    is_draft_worker: bool = False
    # The model's per-rank KV head count and head dim, recorded on the
    # full+state facade even when its full side is an MLA latent.
    head_num: Optional[int] = None
    head_dim: Optional[int] = None
    # The request table and the state storage of a plan with state.
    max_num_reqs: int = 0
    model_context_len: int = 0
    extra_max_context_len: int = 0
    state_cache_params: Optional[Any] = None
    enable_mamba_extra_buffer: bool = False
    enable_mamba_extra_buffer_lazy: bool = False
    speculative_num_draft_tokens: Optional[int] = None
    disable_overlap_schedule: bool = True
    decode_pre_alloc_size: int = 0
    # Logged only: the state share governs the total budget, not the split.
    mamba_full_memory_ratio: Optional[float] = None


def _attach_storage(
    plan: KVStoragePlan, token_to_kv_pool, allocator, ctx: UnifiedBuildContext
) -> None:
    """What host and peer transfers read: the plan's sub-pools through the KV
    pool's router and the allocator's state."""
    allocator.storage = KVStorage(
        plan=plan,
        router=token_to_kv_pool.layer_router,
        allocator=allocator,
        start_layer=ctx.start_layer,
    )


def _log_banner(*lines) -> None:
    rule = "[unified-memory-pool] " + "=" * 60
    logger.info(rule)
    for line in lines:
        logger.info("[unified-memory-pool] %s", line)
    logger.info(rule)


def _build_full_window(
    plan: KVStoragePlan,
    sized: SizedKVStorage,
    sizing: KVSizingInputs,
    ctx: UnifiedBuildContext,
) -> UnifiedSWAPoolBundle:
    """Full + window: both sub-allocators are page-aware, one virtual id space
    at page granularity, two physical sub-pools compacting independently."""
    full = plan.of_lifetime(FullLifetime)
    window = plan.of_lifetime(WindowLifetime)
    legacy_allocator_capacities = {}
    if sizing.token_budget_bytes is None:
        legacy_allocator_capacities = {
            "full_max_total_num_tokens": sizing.full_tokens,
            "swa_max_total_num_tokens": sizing.window_tokens,
        }
    shared_pool = UnifiedKVPool(
        total_bytes=sized.total_bytes,
        sub_pool_specs=plan.specs(),
        device=ctx.device,
        enable_memory_saver=ctx.enable_memory_saver,
        page_size=plan.page_size,
        post_capture_active=ctx.post_capture_active,
        bs1_floor_terms=list(sized.bs1_floor_terms),
        fused_draft=plan.fused_draft,
    )
    token_to_kv_pool = UnifiedSWAKVPool(
        unified_buffer=shared_pool,
        swa_attention_layer_ids=list(window.layer_ids),
        full_attention_layer_ids=list(full.layer_ids),
        page_size=plan.page_size,
        start_layer=ctx.start_layer,
        end_layer=ctx.end_layer,
        enable_memory_saver=ctx.enable_memory_saver,
    )
    allocator = ChainAllocator(
        unified_buffer=shared_pool,
        roles=ChainRoles(owner=full.name, window=window.name),
        kvcache=token_to_kv_pool,
        device=ctx.device,
        page_size=plan.page_size,
        need_sort=ctx.need_sort,
        forward_stream=ctx.forward_stream,
        lazy_compaction=ctx.lazy_compaction,
        **legacy_allocator_capacities,
    )
    _attach_storage(plan, token_to_kv_pool, allocator, ctx)
    _log_banner(
        "UNIFIED MEMORY POOL ENABLED -- path=SWA hybrid",
        "  token-major views, entry_bytes full=%d swa=%d B"
        % (full.spec.entry_bytes(), window.spec.entry_bytes()),
        "  full_layers=%d, swa_layers=%d, page_size=%d"
        % (len(full.layer_ids), len(window.layer_ids), plan.page_size),
        "  total_bytes=%d (=%.2f GB), full_capacity=%d, swa_capacity=%d, "
        "joint_available=%d slots"
        % (
            sized.total_bytes,
            sized.total_bytes / GB,
            allocator.size_full,
            allocator.size_swa,
            allocator.available_size(),
        ),
    )
    return UnifiedSWAPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
    )


def _build_state_request_table(
    plan: KVStoragePlan, ctx: UnifiedBuildContext
) -> UnifiedHybridReqToTokenPool:
    return UnifiedHybridReqToTokenPool(
        mamba_sub_pool_name=plan.of_lifetime(StateLifetime).name,
        size=ctx.max_num_reqs,
        max_context_len=ctx.model_context_len + ctx.extra_max_context_len,
        device=ctx.device,
        enable_memory_saver=ctx.enable_memory_saver,
        enable_mamba_extra_buffer=ctx.enable_mamba_extra_buffer,
        enable_mamba_extra_buffer_lazy=ctx.enable_mamba_extra_buffer_lazy,
        enable_overlap_schedule=not ctx.disable_overlap_schedule,
        start_layer=ctx.start_layer,
        pre_alloc_size=ctx.decode_pre_alloc_size,
    )


def _build_state_storage(
    plan: KVStoragePlan, shared_pool: UnifiedKVPool, ctx: UnifiedBuildContext
) -> StateStorage:
    """The state sub-pool's storage, without its slots: those come from the
    chain allocator's state band (`_wire_mamba_slot_allocator`). The PLE side
    states are built disabled; unified memory refuses them."""
    state = plan.of_lifetime(StateLifetime)
    layer_ids = list(state.layer_ids)
    assert len(ctx.state_cache_params.layers) >= len(layer_ids), (
        f"cache_params.layers ({len(ctx.state_cache_params.layers)}) cannot supply "
        f"{len(layer_ids)} mamba layer ids"
    )
    return StateStorage(
        sub_pool=state.name,
        pool=UnifiedMambaPool(
            unified_buffer=shared_pool,
            sub_pool_name=state.name,
            spec_state_size=ctx.max_num_reqs,  # outer dim of spec-decode intermediates
            mamba_layer_ids=layer_ids,
            enable_memory_saver=ctx.enable_memory_saver,
            speculative_num_draft_tokens=ctx.speculative_num_draft_tokens,
        ),
        slots=None,
        layer_map={layer_id: i for i, layer_id in enumerate(layer_ids)},
        short_conv_pool=ShortConvPool(
            size=0,
            state_shape=None,
            layer_ids=[],
            dtype=torch.bfloat16,
            device=ctx.device,
        ),
        ngram_pool=NGramPool(size=0, context_len=0, eos_token_id=0, device=ctx.device),
    )


def _build_full_state(
    plan: KVStoragePlan,
    sized: SizedKVStorage,
    sizing: KVSizingInputs,
    ctx: UnifiedBuildContext,
) -> UnifiedPoolBundle:
    """Full + state: the full sub-pool is page-aware, the state sub-pool is
    one slot per request."""
    full = plan.of_lifetime(FullLifetime)
    state = plan.of_lifetime(StateLifetime)
    use_mla = isinstance(full.spec, MLASubPoolSpec)
    if use_mla:
        assert not ctx.is_draft_worker, (
            "a draft worker binds views over the target's buffer instead of "
            "building one of its own"
        )
    kv_cache_dtype = full.geometry.kv_cache_dtype
    shared_pool = UnifiedKVPool(
        total_bytes=sized.total_bytes,
        sub_pool_specs=plan.specs(),
        device=ctx.device,
        enable_memory_saver=ctx.enable_memory_saver,
        page_size=plan.page_size,
        fused_draft=plan.fused_draft,
    )
    req_to_token_pool = _build_state_request_table(plan, ctx)
    state_storage = _build_state_storage(plan, shared_pool, ctx)
    if use_mla:
        # start_layer stays 0: HybridLinearKVPool addresses this view by the
        # contiguous 0..N-1 index of its full-attention layers.
        unified_full_kv_pool = UnifiedMLATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name=full.name,
            kv_cache_dtype=kv_cache_dtype,
            page_size=plan.page_size,
        )
    else:
        unified_full_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name=full.name,
            page_size=plan.page_size,
            start_layer=ctx.start_layer,
            end_layer=ctx.end_layer,
        )
    token_to_kv_pool = UnifiedHybridLinearKVPool(
        page_size=plan.page_size,
        size=sizing.full_tokens,
        dtype=kv_cache_dtype,
        head_num=ctx.head_num,
        head_dim=ctx.head_dim,
        full_attention_layer_ids=[0] if ctx.is_draft_worker else list(full.layer_ids),
        device=ctx.device,
        mamba_pool=state_storage.pool,
        enable_memory_saver=ctx.enable_memory_saver,
        use_mla=use_mla,
        start_layer=ctx.start_layer,
        full_kv_pool=unified_full_kv_pool,
    )
    allocator = ChainAllocator(
        unified_buffer=shared_pool,
        roles=ChainRoles(owner=full.name, state=state.name),
        kvcache=token_to_kv_pool,
        state_kvcache=state_storage.pool,
        device=ctx.device,
        page_size=plan.page_size,
        need_sort=ctx.need_sort,
        forward_stream=ctx.forward_stream,
        lazy_compaction=ctx.lazy_compaction,
    )
    # Size host storage from the configured token cap, not the dynamic buffer view.
    full_pool = token_to_kv_pool.full_kv_pool
    full_pool.host_capacity_tokens = sizing.full_tokens
    full_pool.host_capacity_bytes = (
        sizing.full_tokens * allocator.full_attn_allocator.entry_bytes
    )
    _wire_mamba_slot_allocator(
        mamba_end=allocator.mamba_allocator,
        take=allocator.alloc_state_slots,
        state=state_storage,
        device=ctx.device,
    )
    allocator.state = state_storage
    _attach_storage(plan, token_to_kv_pool, allocator, ctx)
    # Only HybridLinearKVPool's retraction CPU-copy path uses this hook.
    token_to_kv_pool._mamba_translate = allocator.state_ids(state.name).ids
    _log_banner(
        "UNIFIED MEMORY POOL ENABLED -- path=Mamba hybrid (%s full side)"
        % ("MLA" if use_mla else "MHA"),
        "  full_layers=%d, mamba_layers=%d, page_size=%d, is_draft_worker=%s "
        "(token-major views, entry_bytes=%d B)"
        % (
            len(full.layer_ids),
            len(state.layer_ids),
            plan.page_size,
            ctx.is_draft_worker,
            full.spec.entry_bytes(),
        ),
        "  total_bytes=%d, max_total_num_tokens=%d, max_mamba_cache_size=%d, "
        "max_num_reqs=%d, speculative_num_draft_tokens=%s"
        % (
            sized.total_bytes,
            sizing.full_tokens,
            sizing.state_slots,
            ctx.max_num_reqs,
            ctx.speculative_num_draft_tokens,
        ),
        *(
            ()
            if ctx.mamba_full_memory_ratio is None
            else (
                "  mamba_full_memory_ratio=%s governs the total budget only, "
                "not the runtime split." % ctx.mamba_full_memory_ratio,
            )
        ),
    )
    return UnifiedPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )


def _build_full_window_state(
    plan: KVStoragePlan,
    sized: SizedKVStorage,
    sizing: KVSizingInputs,
    ctx: UnifiedBuildContext,
) -> UnifiedPoolBundle:
    """Full + window + state: the window sub-pool floats between the two ends."""
    full = plan.of_lifetime(FullLifetime)
    window = plan.of_lifetime(WindowLifetime)
    state = plan.of_lifetime(StateLifetime)
    shared_pool = UnifiedKVPool(
        total_bytes=sized.total_bytes,
        sub_pool_specs=plan.specs(),
        device=ctx.device,
        enable_memory_saver=ctx.enable_memory_saver,
        page_size=plan.page_size,
        fused_draft=plan.fused_draft,
    )
    token_to_kv_pool = UnifiedSWAKVPool(
        unified_buffer=shared_pool,
        swa_attention_layer_ids=list(window.layer_ids),
        full_attention_layer_ids=list(full.layer_ids),
        page_size=plan.page_size,
        start_layer=ctx.start_layer,
        end_layer=ctx.end_layer,
        enable_memory_saver=ctx.enable_memory_saver,
    )
    req_to_token_pool = _build_state_request_table(plan, ctx)
    state_storage = _build_state_storage(plan, shared_pool, ctx)
    allocator = ChainAllocator(
        unified_buffer=shared_pool,
        roles=ChainRoles(owner=full.name, window=window.name, state=state.name),
        kvcache=token_to_kv_pool,
        state_kvcache=state_storage.pool,
        device=ctx.device,
        full_max_total_num_tokens=sizing.full_tokens,
        swa_max_total_num_tokens=sizing.window_tokens,
        page_size=plan.page_size,
        need_sort=ctx.need_sort,
        forward_stream=ctx.forward_stream,
        lazy_compaction=ctx.lazy_compaction,
    )
    _wire_mamba_slot_allocator(
        mamba_end=allocator.mamba_allocator,
        take=allocator.alloc_state_slots,
        state=state_storage,
        device=ctx.device,
    )
    allocator.state = state_storage
    _attach_storage(plan, token_to_kv_pool, allocator, ctx)
    _log_banner(
        "UNIFIED MEMORY POOL ENABLED -- path=SWA+Mamba tri-pool",
        "  full_layers=%d, swa_layers=%d, state_layers=%d, page_size=%d"
        % (
            len(full.layer_ids),
            len(window.layer_ids),
            len(state.layer_ids),
            plan.page_size,
        ),
        "  total_bytes=%d (=%.2f GB), full_max=%d, swa_max=%d, "
        "max_mamba_cache_size=%d, max_num_reqs=%d, joint_available=%d"
        % (
            sized.total_bytes,
            sized.total_bytes / GB,
            sizing.full_tokens,
            sizing.window_tokens,
            sizing.state_slots,
            ctx.max_num_reqs,
            allocator.available_size(),
        ),
    )
    return UnifiedPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )


# Which facade a plan gets, by its set of lifetimes.
_BUILDERS = {
    frozenset({FullLifetime, WindowLifetime}): _build_full_window,
    frozenset({FullLifetime, StateLifetime}): _build_full_state,
    frozenset({FullLifetime, WindowLifetime, StateLifetime}): _build_full_window_state,
}


def build_kv_storage(
    plan: KVStoragePlan,
    sized: SizedKVStorage,
    sizing: KVSizingInputs,
    ctx: UnifiedBuildContext,
):
    builder = _BUILDERS.get(plan.lifetime_types())
    if builder is None:
        raise ValueError(
            "--enable-unified-memory has no pool stack for sub-pools "
            f"{[s.name for s in plan.sub_pools]}"
        )
    return builder(plan, sized, sizing, ctx)


def build_unified_pools(
    *,
    decls: Sequence[KVLayerDecl],
    page_size: int,
    sizing: KVSizingInputs,
    ctx: UnifiedBuildContext,
    fused_draft: Optional[FusedDraftPlacement] = None,
):
    """Declarations -> plan -> sizing -> the unified pool stack."""
    plan = plan_kv_storage(decls, page_size=page_size, fused_draft=fused_draft)
    sized = size_kv_storage(plan, sizing)
    return build_kv_storage(plan, sized, sizing, ctx)
