"""Small unified-pool builders and fakes shared by allocator regression tests."""

from array import array
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.layout.fused_draft import FusedDraftPlacement
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
    UnifiedPoolBundle,
    UnifiedSWAKVPool,
    init_unified_mamba_pools,
    init_unified_mamba_swa_pools,
    init_unified_swa_pools,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def setup_allocator_context():
    reset_context()
    args = ServerArgs(model_path="dummy", device="cpu")
    # Mamba cache setup must not resolve the dummy model through Hugging Face.
    args._mamba_cache_chunk_size = 64
    publish(args, role="tokenizer")


def build_swa_pool(*, page_size=1, lazy=True, ratio=(1, 1), total_bytes=None):
    """Full + SWA composite over real CPU pools, 100 pages per side."""
    full_layers, swa_layers = ratio
    bundle = init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=page_size,
        start_layer=0,
        end_layer=full_layers + swa_layers,
        swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
        full_attention_layer_ids=list(range(full_layers)),
        full_max_total_num_tokens=100 * page_size,
        swa_max_total_num_tokens=100 * page_size,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=lazy,
        total_bytes=total_bytes,
    )
    return bundle.token_to_kv_pool_allocator


def build_tri_pool(*, page_size=1, lazy=True, ratio=(1, 1), temporal=(1, 4, 8)):
    """Full + SWA + Mamba composite over real CPU pools; returns the pool
    bundle and its allocator."""
    full_layers, swa_layers = ratio
    config = SimpleNamespace(
        shape=SimpleNamespace(conv=[(3, 8)], temporal=temporal),
        dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0],
    )
    bundle = init_unified_mamba_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=page_size,
        start_layer=0,
        end_layer=full_layers + swa_layers,
        swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
        full_attention_layer_ids=list(range(full_layers)),
        full_max_total_num_tokens=100 * page_size,
        swa_max_total_num_tokens=100 * page_size,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=lazy,
        mamba_layer_ids=[0],
        mamba2_cache_params=config,
        max_mamba_cache_size=8,
        model_context_len=256 * page_size,
        extra_max_context_len=1,
        max_num_reqs=4,
        enable_mamba_extra_buffer=False,
        enable_mamba_extra_buffer_lazy=False,
        disable_overlap_schedule=True,
        sliding_window_size=32,
    )
    return bundle, bundle.token_to_kv_pool_allocator


class FakeKVCache:
    """buf[p] == virtual id stored at physical slot p (-1 free); moves copy it."""

    def __init__(self, max_slots: int):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_physical_pages(
        self, dst_pages: torch.Tensor, src_pages: torch.Tensor, page_size: int
    ):
        slots = torch.arange(page_size, device=dst_pages.device)
        dst_loc = (dst_pages[:, None] * page_size + slots).flatten()
        src_loc = (src_pages[:, None] * page_size + slots).flatten()
        self.buf[dst_loc] = self.buf[src_loc].clone()


class FakeUnifiedSWAKVPool:
    class _SubKV(FakeKVCache):
        def __init__(self, max_slots):
            super().__init__(max_slots)
            self.allocator = None

        def attach_allocator(self, allocator):
            self.allocator = allocator

    def __init__(self, shared_pool: UnifiedKVPool):
        self.full_kv_pool = self._SubKV(shared_pool.max_slots("full"))
        self.swa_kv_pool = self._SubKV(shared_pool.max_slots("swa"))
        self._full_allocator = None
        self._swa_allocator = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


def tri_sub_pool_specs(
    full_layer_num=4, swa_layer_num=2, state_layer_num=2, head_num=2, head_dim=4
):
    full = MHASubPoolSpec(
        name="full",
        layer_num=full_layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction="down",
    )
    swa = MHASubPoolSpec(
        name="swa",
        layer_num=swa_layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction="float",
    )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=state_layer_num,
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(0, 0, 0),  # Inkling: conv-only, no SSM state
        temporal_dtype=torch.float32,
        grow_direction="up",
    )
    return full, swa, mamba


def build_swa_cache(
    *,
    occupancy=96,
    lazy=False,
    page_size=1,
    ratio=(1, 1),
    total_bytes=None,
    sessions=False,
    window=None,
):
    allocator = build_swa_pool(
        page_size=page_size, lazy=lazy, ratio=ratio, total_bytes=total_bytes
    )
    req_pool = ReqToTokenPool(
        size=4,
        max_context_len=256 * page_size,
        device="cpu",
        enable_memory_saver=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            enable_session_radix_cache=sessions,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=128 * page_size if window is None else window,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    slots = allocator.alloc(occupancy * page_size)
    assert slots is not None, "Fixture allocation must fit"
    for i in range(occupancy):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(i * 1000, i * 1000 + page_size))),
                value=slots[i * page_size : (i + 1) * page_size],
            )
        )
    return allocator, cache


def build_tri_cache(
    *,
    lazy=True,
    temporal=(0, 0, 0),
    page_size=1,
    ratio=(1, 1),
    sessions=False,
    state_cache=True,
):
    bundle, a = build_tri_pool(
        page_size=page_size, lazy=lazy, ratio=ratio, temporal=temporal
    )
    c = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=bundle.req_to_token_pool,
            token_to_kv_pool_allocator=a,
            page_size=page_size,
            sliding_window_size=128 * page_size,
            tree_components=(
                (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
                if state_cache
                else (ComponentType.FULL, ComponentType.SWA)
            ),
            enable_session_radix_cache=sessions,
        )
    )
    return bundle, a, c


class UnifiedGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """A small test model for the unified pool factories.

    ``chain`` picks the factory: "swa" (full + window), "mamba" (full + state)
    or "tri" (full + window + state). ``byte_budget`` is the factory's byte
    budget: the whole buffer for "swa", the token side for the state chains
    (their state slots ride on top); without it the token capacities size it.
    """

    chain: str
    full_layer_ids: Tuple[int, ...]
    swa_layer_ids: Tuple[int, ...] = ()
    state_layer_ids: Tuple[int, ...] = ()
    start_layer: int = 0
    end_layer: Optional[int] = None
    mla: bool = False
    head_num: int = 1
    head_dim: int = 8
    v_head_dim: Optional[int] = None
    swa_head_num: int = 1
    swa_head_dim: int = 8
    swa_v_head_dim: Optional[int] = None
    kv_lora_rank: int = 16
    qk_rope_head_dim: int = 8
    kv_cache_dtype: torch.dtype = torch.float16
    page_size: int = 1
    window: int = 32
    conv_shapes: Tuple[Tuple[int, ...], ...] = ((3, 8),)
    conv_dtype: torch.dtype = torch.bfloat16
    temporal_shape: Tuple[int, ...] = (1, 4, 8)
    temporal_dtype: torch.dtype = torch.float32
    # Set on the cache-params source only when not None, as model configs that
    # carry the attribute do.
    conv_slice_axis: Optional[int] = None
    fused_draft: Optional[FusedDraftPlacement] = None
    full_tokens: int = 64
    swa_tokens: int = 64
    state_slots: int = 8
    byte_budget: Optional[int] = None
    context_len: int = 256
    max_num_reqs: int = 4
    lazy: bool = True
    # Paged extend and decode launch device kernels; their callers need "cuda".
    device: str = "cpu"


def _state_cache_params(geometry: UnifiedGeometry) -> SimpleNamespace:
    shape = SimpleNamespace(
        conv=[tuple(s) for s in geometry.conv_shapes],
        temporal=tuple(geometry.temporal_shape),
    )
    if geometry.conv_slice_axis is not None:
        shape.conv_slice_axis = geometry.conv_slice_axis
    return SimpleNamespace(
        shape=shape,
        dtype=SimpleNamespace(
            conv=geometry.conv_dtype, temporal=geometry.temporal_dtype
        ),
        layers=list(geometry.state_layer_ids),
    )


def build_unified_bundle(geometry: UnifiedGeometry):
    """Build the unified pool stack a geometry describes, through the factory
    production uses for that chain."""
    g = geometry
    layer_ids = g.full_layer_ids + g.swa_layer_ids + g.state_layer_ids
    common = dict(
        device=g.device,
        kv_cache_dtype=g.kv_cache_dtype,
        page_size=g.page_size,
        start_layer=g.start_layer,
        end_layer=g.end_layer if g.end_layer is not None else max(layer_ids) + 1,
        full_attention_layer_ids=list(g.full_layer_ids),
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=g.lazy,
        fused_draft=g.fused_draft,
        head_num=g.head_num,
        head_dim=g.head_dim,
        v_head_dim=g.v_head_dim if g.v_head_dim is not None else g.head_dim,
    )
    swa_kwargs = dict(
        swa_head_num=g.swa_head_num,
        swa_head_dim=g.swa_head_dim,
        swa_v_head_dim=(
            g.swa_v_head_dim if g.swa_v_head_dim is not None else g.swa_head_dim
        ),
        swa_attention_layer_ids=list(g.swa_layer_ids),
        sliding_window_size=g.window,
    )
    state_kwargs = dict(
        mamba_layer_ids=list(g.state_layer_ids),
        mamba2_cache_params=_state_cache_params(g),
        max_mamba_cache_size=g.state_slots,
        model_context_len=g.context_len,
        extra_max_context_len=1,
        max_num_reqs=g.max_num_reqs,
        enable_mamba_extra_buffer=False,
        enable_mamba_extra_buffer_lazy=False,
        disable_overlap_schedule=True,
        speculative_num_draft_tokens=None,
        unified_total_bytes=g.byte_budget,
    )
    if g.chain == "swa":
        by_tokens = g.byte_budget is None
        return init_unified_swa_pools(
            **common,
            **swa_kwargs,
            full_max_total_num_tokens=g.full_tokens if by_tokens else None,
            swa_max_total_num_tokens=g.swa_tokens if by_tokens else None,
            total_bytes=g.byte_budget,
            model_context_len=g.context_len,
        )
    if g.chain == "mamba":
        return init_unified_mamba_pools(
            **common,
            **state_kwargs,
            is_draft_worker=False,
            use_mla_backend=g.mla,
            kv_lora_rank=g.kv_lora_rank if g.mla else None,
            qk_rope_head_dim=g.qk_rope_head_dim if g.mla else None,
            max_total_num_tokens=g.full_tokens,
        )
    assert g.chain == "tri", f"unknown chain {g.chain!r}"
    return init_unified_mamba_swa_pools(
        **common,
        **swa_kwargs,
        **state_kwargs,
        full_max_total_num_tokens=g.full_tokens,
        swa_max_total_num_tokens=g.swa_tokens,
    )


def build_mamba_pool(*, page_size=1, lazy=True, mla=False, ratio=(1, 1)):
    """Full + state composite over real CPU pools; returns the pool bundle and
    its allocator."""
    full_layers, state_layers = ratio
    bundle = build_unified_bundle(
        UnifiedGeometry(
            chain="mamba",
            full_layer_ids=tuple(range(full_layers)),
            state_layer_ids=tuple(range(full_layers, full_layers + state_layers)),
            mla=mla,
            page_size=page_size,
            full_tokens=100 * page_size,
            context_len=256 * page_size,
            lazy=lazy,
        )
    )
    return bundle, bundle.token_to_kv_pool_allocator


def build_mamba_cache(*, page_size=1, lazy=True, mla=False, ratio=(1, 1)):
    bundle, allocator = build_mamba_pool(
        page_size=page_size, lazy=lazy, mla=mla, ratio=ratio
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=bundle.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        )
    )
    return bundle, allocator, cache


def _view_summary(view: torch.Tensor) -> tuple:
    return (
        view.storage_offset() * view.element_size(),
        tuple(view.shape),
        tuple(view.stride()),
        str(view.dtype),
    )


def _ends_summary(views) -> tuple:
    """First and last layer of a per-layer view list."""
    return (_view_summary(views[0]), _view_summary(views[-1]))


def _sub_pool_summary(pool: UnifiedKVPool, spec) -> dict:
    out = {
        "kind": type(spec).__name__,
        "layer_num": spec.layer_num,
        "entry_bytes": spec.entry_bytes(),
        "host_entry_bytes": spec.host_entry_bytes(),
        "draft_offset": (
            None if spec.draft_region is None else spec.draft_offset_in_entry()
        ),
        "max_slots": pool.max_slots(spec.name),
        "min_slot_index": pool.min_slot_index(spec.name),
    }
    if isinstance(spec, MambaSubPoolSpec):
        conv, temporal = pool.mamba_views_for(spec.name)
        out["state"] = (
            tuple(tuple(s) for s in spec.conv_state_shapes),
            str(spec.conv_dtype),
            tuple(spec.temporal_state_shape),
            str(spec.temporal_dtype),
            spec.conv_slice_axis,
        )
        out["views"] = {
            "conv": tuple(_view_summary(v) for v in conv),
            "temporal": _view_summary(temporal),
        }
        if spec.draft_region is not None:
            d_conv, d_temporal = pool.build_draft_state_views(spec.name)
            out["views"]["draft_conv"] = tuple(_view_summary(v) for v in d_conv)
            out["views"]["draft_temporal"] = _view_summary(d_temporal)
        return out
    out["parts"] = tuple(
        (
            p.name,
            p.offset_bytes,
            p.layer_stride_bytes,
            p.layer_num,
            tuple(p.row_shape),
            str(p.dtype),
        )
        for p in spec.layout().parts
    )
    if isinstance(spec, MHASubPoolSpec):
        k, v = pool.mha_views_for(spec.name)
        out["views"] = {"k": _ends_summary(k), "v": _ends_summary(v)}
    else:
        assert isinstance(spec, MLASubPoolSpec), type(spec)
        out["views"] = {"kv": _ends_summary(pool.mla_views_for(spec.name))}
    if spec.draft_region is not None:
        d_k, d_v = pool.build_dense_draft_views(spec.name)
        out["views"]["draft_k"] = _ends_summary(d_k)
        out["views"]["draft_v"] = _ends_summary(d_v)
    return out


def _chain_members(allocator) -> Dict[str, object]:
    """The composite's band allocators, low byte end first, by walking the
    neighbour links its chain wiring set up."""
    member = allocator.full_attn_allocator
    while member.low_peer is not None:
        member = member.low_peer
    members = {}
    while member is not None:
        members[member.sub_pool_name] = member
        member = member.high_peer
    return members


def _member_summary(member) -> tuple:
    return (
        member.grow_direction,
        member.num_pages,
        member.min_page_index,
        member.num_virtual_ids,
        member.entry_bytes_per_page,
    )


def _kv_pools(bundle) -> Dict[str, object]:
    kv = bundle.token_to_kv_pool
    pools = {"full": kv.full_kv_pool}
    if isinstance(kv, UnifiedSWAKVPool):
        pools["swa"] = kv.swa_kv_pool
    if isinstance(bundle, UnifiedPoolBundle):
        pools["mamba"] = bundle.req_to_token_pool.mamba_pool
    return pools


def _pool_summary(pool, raw_base: int) -> dict:
    ptrs, lens, items = pool.get_contiguous_buf_infos()
    out = {
        "buf_infos": tuple(
            (ptr - raw_base, length, item)
            for ptr, length, item in zip(ptrs, lens, items)
        ),
        "host_capacity": (pool.host_capacity_tokens, pool.host_capacity_bytes),
    }
    if isinstance(pool, UnifiedMHATokenToKVPool):
        out["page_envelope"] = tuple(pool.get_page_envelope_buffer().shape)
    return out


def _layer_maps(bundle) -> dict:
    kv = bundle.token_to_kv_pool
    if isinstance(kv, UnifiedSWAKVPool):
        kv_map = tuple(
            sorted(
                (gid, idx, is_swa) for gid, (idx, is_swa) in kv.layers_mapping.items()
            )
        )
    else:
        kv_map = tuple(sorted(kv.full_attention_layer_id_mapping.items()))
    out = {"kv": kv_map}
    if isinstance(bundle, UnifiedPoolBundle):
        out["state"] = tuple(sorted(bundle.req_to_token_pool.mamba_map.items()))
    return out


def _capacity_summary(bundle) -> dict:
    allocator = bundle.token_to_kv_pool_allocator
    out = {
        "available": allocator.available_size(),
        "size_full": allocator.size_full,
        "draft_virtual_id_space": allocator.draft_virtual_id_space,
    }
    if isinstance(bundle.token_to_kv_pool, UnifiedSWAKVPool):
        out["size_swa"] = allocator.size_swa
    if isinstance(bundle, UnifiedPoolBundle):
        out["state_slots"] = bundle.req_to_token_pool.mamba_allocator.size
    return out


def unified_layout_summary(bundle) -> dict:
    """Plain data describing a built unified pool stack's byte layout: chain,
    entries, part offsets, views, envelopes, slot bounds and empty capacities."""
    pool = bundle.unified_memory_pool
    raw_base = pool._raw.data_ptr()
    out = {
        "chain": tuple((s.name, s.grow_direction) for s in pool.sub_pool_specs),
        "total_bytes": pool.total_bytes,
        "reserved_floor": pool._reserved_floor,
        "active_allocation_bytes": pool.active_allocation_bytes,
        "sub_pools": {s.name: _sub_pool_summary(pool, s) for s in pool.sub_pool_specs},
        "members": {
            name: _member_summary(m)
            for name, m in _chain_members(bundle.token_to_kv_pool_allocator).items()
        },
        "pools": {
            name: _pool_summary(p, raw_base) for name, p in _kv_pools(bundle).items()
        },
        "layer_maps": _layer_maps(bundle),
        "capacity": _capacity_summary(bundle),
    }
    if not isinstance(bundle, UnifiedPoolBundle):
        # Only the window chain's stored floor has a consumer (post-capture
        # backing re-checks it).
        out["bs1_floor_terms"] = tuple(pool._bs1_floor_terms)
    return out
