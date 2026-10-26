"""Build the unified KV pool stack from declarations: plan, size, then construct.

`build_unified_pools` is the one entry point. Every plan gets one byte
buffer, a view per sub-pool, one `RoutedKVCache` over the token views and one
`ChainAllocator` over all of them.
"""

import logging
from typing import Any, Optional, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.decl import (
    KVLayerDecl,
    StateLifetime,
    WindowLifetime,
)
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.plan import (
    KVStoragePlan,
    PlannedSubPool,
    plan_kv_storage,
    resized_after_capture,
)
from sglang.srt.mem_cache.kv.routed_cache import RoutedKVCache
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
    UnifiedHybridReqToTokenPool,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMHATokenToKVPool,
    UnifiedMLATokenToKVPool,
    UnifiedPoolBundle,
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


def _log_banner(*lines) -> None:
    rule = "[unified-memory-pool] " + "=" * 60
    logger.info(rule)
    for line in lines:
        logger.info("[unified-memory-pool] %s", line)
    logger.info(rule)


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


def _build_token_view(
    sub_pool: PlannedSubPool,
    shared_pool: UnifiedKVPool,
    plan: KVStoragePlan,
    ctx: UnifiedBuildContext,
):
    """A token sub-pool's view in its layout."""
    if isinstance(sub_pool.spec, MLASubPoolSpec):
        # Addressed by the 0..N-1 index of its layers, so start_layer stays 0.
        return UnifiedMLATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name=sub_pool.name,
            kv_cache_dtype=sub_pool.geometry.kv_cache_dtype,
            page_size=plan.page_size,
        )
    return UnifiedMHATokenToKVPool(
        unified_buffer=shared_pool,
        sub_pool_name=sub_pool.name,
        page_size=plan.page_size,
        start_layer=ctx.start_layer,
        end_layer=ctx.end_layer,
    )


def build_kv_storage(
    plan: KVStoragePlan,
    sized: SizedKVStorage,
    sizing: KVSizingInputs,
    ctx: UnifiedBuildContext,
) -> UnifiedPoolBundle:
    """One byte buffer for the plan's sub-pools, a view per token sub-pool,
    the routed KV cache over them, and the chain allocator handing out their
    ids. A plan with state also gets its request table and state storage."""
    assert not ctx.is_draft_worker, (
        "a draft worker binds views over the target's buffer instead of "
        "building one of its own"
    )
    owner = plan.owner()
    window = plan.of_lifetime(WindowLifetime)
    state = plan.of_lifetime(StateLifetime)
    if len(plan.sub_pools) < 2:
        raise ValueError(
            "--enable-unified-memory has no pool stack for sub-pools "
            f"{[s.name for s in plan.sub_pools]}"
        )
    shared_pool = UnifiedKVPool(
        total_bytes=sized.total_bytes,
        sub_pool_specs=plan.specs(),
        device=ctx.device,
        enable_memory_saver=ctx.enable_memory_saver,
        page_size=plan.page_size,
        post_capture_active=(
            ctx.post_capture_active and resized_after_capture(plan.lifetime_types())
        ),
        bs1_floor_terms=list(sized.bs1_floor_terms),
        fused_draft=plan.fused_draft,
    )
    token_sub_pools = [s for s in plan.owner_first() if s is not state]
    views = {
        s.name: _build_token_view(s, shared_pool, plan, ctx) for s in token_sub_pools
    }
    for s in token_sub_pools:
        # Only the pool routing a layer knows its model id.
        if isinstance(views[s.name], UnifiedMHATokenToKVPool):
            views[s.name].model_layer_ids = tuple(s.layer_ids)
    # Today's token capacity of the pool: the owner's configured cap where
    # the owner is the only token sub-pool, else its slots (until sizing
    # follows one rule for every chain).
    kv_size = (
        sizing.full_tokens if window is None else shared_pool.max_slots(owner.name) - 1
    )
    owner_spec = owner.spec
    owner_is_mha = isinstance(owner_spec, MHASubPoolSpec)
    token_to_kv_pool = RoutedKVCache(
        unified_buffer=shared_pool,
        owner=owner.name,
        layer_maps=[
            (
                s.name,
                views[s.name],
                {layer_id: i for i, layer_id in enumerate(s.layer_ids)},
            )
            for s in token_sub_pools
        ],
        page_size=plan.page_size,
        start_layer=ctx.start_layer,
        end_layer=ctx.end_layer,
        size=kv_size,
        # An MHA owner's heads; an MLA latent records the model's.
        head_num=owner_spec.head_num if owner_is_mha else ctx.head_num,
        head_dim=owner_spec.head_dim if owner_is_mha else ctx.head_dim,
    )
    req_to_token_pool = None
    state_storage = None
    if state is not None:
        req_to_token_pool = _build_state_request_table(plan, ctx)
        state_storage = _build_state_storage(plan, shared_pool, ctx)
        views[state.name] = state_storage.pool
    legacy_capacities = {}
    if window is not None and (sizing.token_budget_bytes is None or state is not None):
        legacy_capacities = {
            "full_max_total_num_tokens": sizing.full_tokens,
            "swa_max_total_num_tokens": sizing.window_tokens,
        }
    allocator = ChainAllocator(
        unified_buffer=shared_pool,
        roles=ChainRoles(
            owner=owner.name,
            window=None if window is None else window.name,
            state=None if state is None else state.name,
        ),
        kvcache=token_to_kv_pool,
        views=views,
        device=ctx.device,
        page_size=plan.page_size,
        need_sort=ctx.need_sort,
        forward_stream=ctx.forward_stream,
        lazy_compaction=ctx.lazy_compaction,
        **legacy_capacities,
    )
    if window is None:
        # Host storage is sized from the configured token cap, not the
        # buffer's dynamic share.
        owner_view = views[owner.name]
        owner_view.host_capacity_tokens = sizing.full_tokens
        owner_view.host_capacity_bytes = (
            sizing.full_tokens * allocator.full_attn_allocator.entry_bytes
        )
    if state is not None:
        _wire_mamba_slot_allocator(
            mamba_end=allocator.mamba_allocator,
            take=allocator.alloc_state_slots,
            state=state_storage,
            device=ctx.device,
        )
        allocator.state = state_storage
    allocator.storage = KVStorage(
        plan=plan,
        router=token_to_kv_pool.layer_router,
        allocator=allocator,
        start_layer=ctx.start_layer,
    )
    _log_banner(
        "UNIFIED MEMORY POOL ENABLED -- sub-pools %s"
        % ", ".join(
            f"{s.name}({len(s.layer_ids)} layers, entry_bytes={s.spec.entry_bytes()})"
            for s in plan.owner_first()
        ),
        "  page_size=%d, total_bytes=%d (=%.2f GB), max_num_reqs=%d, "
        "joint_available=%d"
        % (
            plan.page_size,
            sized.total_bytes,
            sized.total_bytes / GB,
            ctx.max_num_reqs,
            allocator.available_size(),
        ),
        *(
            ()
            if ctx.mamba_full_memory_ratio is None or state is None
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
