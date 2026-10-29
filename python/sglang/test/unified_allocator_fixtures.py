"""Small unified-pool builders and fakes shared by allocator regression tests."""

from array import array
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.alloc.chain import ChainRoles
from sglang.srt.mem_cache.kv.alloc.pricing import (
    ChainSnapshot,
    FloatChain,
    TokenEnds,
    snapshot_chain,
)
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_unified_pools
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    MLAGeometry,
    StateGeometry,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.memory.arena import ArenaBand, KVArena
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs
from sglang.srt.mem_cache.layout.fused_draft import FusedDraftPlacement
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.tree_slots import tree_slots_of


def chain_snapshot(allocator) -> ChainSnapshot:
    """A unified chain allocator's band snapshot, as its pricing reads it."""
    return snapshot_chain(
        allocator.bands,
        page_size=allocator.page_size,
        total_bytes=allocator.arena.total_bytes,
    )


def token_ends(allocator) -> TokenEnds:
    """The pricing constants of a chain of two token ends."""
    return allocator._policy._ends(allocator)


def float_chain(allocator) -> FloatChain:
    """The roles of a chain whose window floats between two ends."""
    roles = allocator.roles
    return FloatChain(owner=roles.owner, window=roles.window, state=roles.state)


def pin_every_band(allocator, busy) -> None:
    """Register ``busy`` as a pin source over every band of a chain allocator,
    as PD and HiCache register theirs over every sub-pool."""
    allocator.add_pin_source(tuple(m.sub_pool_name for m in allocator.members), busy)


def setup_allocator_context():
    reset_context()
    args = ServerArgs(model_path="dummy", device="cpu")
    # Mamba cache setup must not resolve the dummy model through Hugging Face.
    args._mamba_cache_chunk_size = 64
    publish(args, role="tokenizer")


def build_swa_pool(*, page_size=1, lazy=True, ratio=(1, 1), total_bytes=None):
    """Full + SWA composite over real CPU pools, 100 pages per side."""
    full_layers, swa_layers = ratio
    bundle = build_test_unified_pools(
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


def build_tri_pool(
    *, page_size=1, lazy=True, ratio=(1, 1), temporal=(1, 4, 8), total_bytes=None
):
    """Full + SWA + Mamba composite over real CPU pools; returns the pool
    bundle and its allocator."""
    full_layers, swa_layers = ratio
    config = SimpleNamespace(
        shape=SimpleNamespace(conv=[(3, 8)], temporal=temporal),
        dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0],
    )
    bundle = build_test_unified_pools(
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
        total_bytes=total_bytes,
    )
    return bundle, bundle.token_to_kv_pool_allocator


class FakeKVCache:
    """buf[p] == virtual id stored at physical slot p (-1 free); moves copy it."""

    def __init__(self, source, name=None):
        # A bare size for a pool no arena moves; else the band of ``source``
        # (a test arena) whose unit moves this buffer follows.
        if name is None:
            size = source
        else:
            size = source.num_entries(name)
            source.follow(name, self)
        # buf[p] == virtual id currently stored at physical slot p (-1 if free).
        self.buf = torch.full((size,), -1, dtype=torch.int64)

    def move_units(
        self, dst_pages: torch.Tensor, src_pages: torch.Tensor, page_size: int
    ):
        slots = torch.arange(page_size, device=dst_pages.device)
        dst_loc = (dst_pages[:, None] * page_size + slots).flatten()
        src_loc = (src_pages[:, None] * page_size + slots).flatten()
        self.buf[dst_loc] = self.buf[src_loc].clone()


class FakeUnifiedSWAKVPool:
    class _SubKV(FakeKVCache):
        def __init__(self, arena, name):
            super().__init__(arena, name)
            self.allocator = None

        def attach_allocator(self, allocator):
            self.allocator = allocator

    def __init__(self, arena: KVArena):
        self.full_kv_pool = self._SubKV(arena, "full")
        self.swa_kv_pool = self._SubKV(arena, "swa")


class SpecArena(KVArena):
    """A test's arena that also remembers the specs it was laid out from."""

    def __init__(self, specs, **kwargs):
        super().__init__(**kwargs)
        self._specs = {s.name: s for s in specs}
        self._followers: Dict[str, list] = {}

    def spec(self, name: str):
        return self._specs[name]

    def follow(self, name: str, fake) -> None:
        """Let a fake view's payload follow this band's unit moves."""
        self._followers.setdefault(name, []).append(fake)

    def move_units(self, name, dst, src) -> None:
        super().move_units(name, dst, src)
        for fake in self._followers.get(name, ()):
            fake.move_units(dst, src, self.band(name).unit_entries)


def arena_of(
    sub_pool_specs,
    *,
    total_bytes: int,
    device,
    page_size: int = 1,
    enable_memory_saver: bool = False,
    post_capture_active: bool = False,
    fused_draft: Optional[FusedDraftPlacement] = None,
) -> SpecArena:
    """An arena over hand-built sub-pool specs, as a plan of them lays it out:
    a state sub-pool's unit is one entry, every other one's a page."""
    for s in sub_pool_specs:
        expected = None if fused_draft is None else fused_draft.region(s.name)
        assert s.draft_region is expected, (
            f"sub-pool {s.name!r}: draft_region {s.draft_region} does not match "
            f"the fused draft placement's {expected}"
        )
    bands = [
        ArenaBand(
            name=s.name,
            entry_bytes=s.entry_bytes(),
            unit_entries=1 if isinstance(s, MambaSubPoolSpec) else page_size,
            grow_direction=s.grow_direction,
            zero_on_handout=s.zero_units_on_handout,
        )
        for s in sub_pool_specs
    ]
    return SpecArena(
        sub_pool_specs,
        bands=bands,
        total_bytes=total_bytes,
        sink_bytes=max(b.unit_bytes for b in bands),
        device=device,
        enable_memory_saver=enable_memory_saver,
        backed_after_capture=post_capture_active,
    )


def storage_parts(storage) -> SimpleNamespace:
    """A built KV storage's parts, under the names the runtime holds them by."""
    allocator = storage.allocator
    return SimpleNamespace(
        storage=storage,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=allocator.get_kvcache(),
        arena=allocator.arena,
        req_to_token_pool=storage.requests,
    )


def plan_stand_in(arena: SpecArena, fused_draft=None) -> SimpleNamespace:
    """What draft binding reads of a plan, over a test arena's specs."""
    return SimpleNamespace(
        fused_draft=fused_draft,
        sub_pool=lambda name: SimpleNamespace(spec=arena.spec(name)),
    )


def stand_in_views(roles: ChainRoles, kvcache, state=None) -> Dict[str, object]:
    """The sub-pool views a test's stand-in KV pool holds, by sub-pool name:
    what `ChainAllocator` takes as `views`."""
    views = {roles.owner: kvcache.full_kv_pool}
    if roles.window is not None:
        views[roles.window] = kvcache.swa_kv_pool
    if roles.state is not None:
        views[roles.state] = state
    return views


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
    if total_bytes is None:
        # These eviction fixtures reserve 100 pages of byte space including
        # sinks, leaving 99 joint pages for the pressure-sensitive scenarios.
        total_bytes = 100 * page_size * sum(ratio) * 32
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
            tree_slots=tree_slots_of(
                (ComponentType.FULL, ComponentType.SWA),
                window=128 * page_size if window is None else window,
            ),
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
    total_bytes=None,
):
    bundle, a = build_tri_pool(
        page_size=page_size,
        lazy=lazy,
        ratio=ratio,
        temporal=temporal,
        total_bytes=total_bytes,
    )
    c = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=bundle.req_to_token_pool,
            token_to_kv_pool_allocator=a,
            page_size=page_size,
            sliding_window_size=128 * page_size,
            tree_slots=tree_slots_of(
                (
                    (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
                    if state_cache
                    else (ComponentType.FULL, ComponentType.SWA)
                ),
                window=128 * page_size,
            ),
            enable_session_radix_cache=sessions,
        )
    )
    return bundle, a, c


class UnifiedGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """A small test model for the unified pool builder.

    ``chain`` names its kinds: "swa" (full + window), "mamba" (full + state) or
    "tri" (full + window + state); the layer ids given per kind must match.
    ``byte_budget`` is the complete device-storage budget;
    without it the token capacities size the buffer.
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


def build_test_unified_pools(
    *,
    device,
    kv_cache_dtype,
    page_size,
    start_layer,
    end_layer,
    full_attention_layer_ids,
    head_num,
    head_dim,
    v_head_dim=None,
    use_mla_backend=False,
    kv_lora_rank=None,
    qk_rope_head_dim=None,
    swa_attention_layer_ids=(),
    swa_head_num=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    sliding_window_size=None,
    mamba_layer_ids=(),
    mamba2_cache_params=None,
    max_mamba_cache_size=0,
    full_max_total_num_tokens=None,
    max_total_num_tokens=None,
    swa_max_total_num_tokens=None,
    total_bytes=None,
    unified_total_bytes=None,
    model_context_len=None,
    extra_max_context_len=0,
    max_num_reqs=1,
    enable_memory_saver=False,
    post_capture_active=False,
    need_sort=False,
    forward_stream=None,
    lazy_compaction=False,
    is_draft_worker=False,
    enable_mamba_extra_buffer=False,
    enable_mamba_extra_buffer_lazy=False,
    speculative_num_draft_tokens=None,
    disable_overlap_schedule=True,
    decode_pre_alloc_size=0,
    mamba_full_memory_ratio=None,
    fused_draft=None,
):
    """Build a unified pool stack from flat model arguments: the layer ids given
    for each kind (full, sliding-window, state) decide the sub-pools."""
    if use_mla_backend:
        full_geometry = MLAGeometry(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_cache_dtype=kv_cache_dtype,
        )
    else:
        full_geometry = MHAGeometry(
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim if v_head_dim is not None else head_dim,
            kv_cache_dtype=kv_cache_dtype,
        )
    window_geometry = None
    if swa_attention_layer_ids:
        window_geometry = MHAGeometry(
            head_num=swa_head_num,
            head_dim=swa_head_dim,
            v_head_dim=swa_v_head_dim if swa_v_head_dim is not None else swa_head_dim,
            kv_cache_dtype=kv_cache_dtype,
        )
    decls = kv_layer_decls(
        full_layer_ids=full_attention_layer_ids,
        full_geometry=full_geometry,
        window_layer_ids=swa_attention_layer_ids,
        window_geometry=window_geometry,
        window=sliding_window_size,
        state_layer_ids=mamba_layer_ids,
        state_geometry=(
            StateGeometry.of(mamba2_cache_params) if mamba_layer_ids else None
        ),
    )
    sizing = KVSizingInputs(
        budget_bytes=total_bytes if total_bytes is not None else unified_total_bytes,
        full_tokens=(
            full_max_total_num_tokens
            if full_max_total_num_tokens is not None
            else max_total_num_tokens
        ),
        window_tokens=swa_max_total_num_tokens,
        state_slots=max_mamba_cache_size,
        context_len=model_context_len,
        floor_window=sliding_window_size,
        alignment=1,
    )
    ctx = UnifiedBuildContext(
        device=device,
        start_layer=start_layer,
        end_layer=end_layer,
        enable_memory_saver=enable_memory_saver,
        need_sort=need_sort,
        lazy_compaction=lazy_compaction,
        forward_stream=forward_stream,
        post_capture_active=post_capture_active,
        is_draft_worker=is_draft_worker,
        head_num=head_num,
        head_dim=head_dim,
        max_num_reqs=max_num_reqs,
        model_context_len=model_context_len or 0,
        extra_max_context_len=extra_max_context_len,
        state_cache_params=mamba2_cache_params,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        enable_mamba_extra_buffer_lazy=enable_mamba_extra_buffer_lazy,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        disable_overlap_schedule=disable_overlap_schedule,
        decode_pre_alloc_size=decode_pre_alloc_size,
        mamba_full_memory_ratio=mamba_full_memory_ratio,
    )
    return storage_parts(
        build_unified_pools(
            decls=decls,
            page_size=page_size,
            sizing=sizing,
            ctx=ctx,
            fused_draft=fused_draft,
        )
    )


def build_unified_bundle(geometry: UnifiedGeometry):
    """Build the unified pool stack a geometry describes, through the
    declarations, plan, sizing and builder production uses."""
    g = geometry
    layer_ids = g.full_layer_ids + g.swa_layer_ids + g.state_layer_ids
    if g.mla:
        full_geometry = MLAGeometry(
            kv_lora_rank=g.kv_lora_rank,
            qk_rope_head_dim=g.qk_rope_head_dim,
            kv_cache_dtype=g.kv_cache_dtype,
        )
    else:
        full_geometry = MHAGeometry(
            head_num=g.head_num,
            head_dim=g.head_dim,
            v_head_dim=g.v_head_dim if g.v_head_dim is not None else g.head_dim,
            kv_cache_dtype=g.kv_cache_dtype,
        )
    cache_params = _state_cache_params(g) if g.state_layer_ids else None
    decls = kv_layer_decls(
        full_layer_ids=g.full_layer_ids,
        full_geometry=full_geometry,
        window_layer_ids=g.swa_layer_ids,
        window_geometry=MHAGeometry(
            head_num=g.swa_head_num,
            head_dim=g.swa_head_dim,
            v_head_dim=(
                g.swa_v_head_dim if g.swa_v_head_dim is not None else g.swa_head_dim
            ),
            kv_cache_dtype=g.kv_cache_dtype,
        ),
        window=g.window,
        state_layer_ids=g.state_layer_ids,
        state_geometry=(
            StateGeometry.of(cache_params) if cache_params is not None else None
        ),
    )
    by_tokens = g.byte_budget is None or bool(g.state_layer_ids)
    sizing = KVSizingInputs(
        budget_bytes=g.byte_budget,
        full_tokens=g.full_tokens if by_tokens else None,
        window_tokens=g.swa_tokens if by_tokens and g.swa_layer_ids else None,
        state_slots=g.state_slots if g.state_layer_ids else 0,
        context_len=g.context_len,
        floor_window=g.window if g.swa_layer_ids else None,
        alignment=1,
    )
    ctx = UnifiedBuildContext(
        device=g.device,
        start_layer=g.start_layer,
        end_layer=g.end_layer if g.end_layer is not None else max(layer_ids) + 1,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=g.lazy,
        head_num=g.head_num,
        head_dim=g.head_dim,
        max_num_reqs=g.max_num_reqs,
        model_context_len=g.context_len,
        extra_max_context_len=1,
        state_cache_params=cache_params,
        disable_overlap_schedule=True,
    )
    return storage_parts(
        build_unified_pools(
            decls=decls,
            page_size=g.page_size,
            sizing=sizing,
            ctx=ctx,
            fused_draft=g.fused_draft,
        )
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
            tree_slots=tree_slots_of((ComponentType.FULL, ComponentType.MAMBA)),
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


def _sub_pool_summary(arena: KVArena, spec) -> dict:
    out = {
        "kind": type(spec).__name__,
        "layer_num": spec.layer_num,
        "entry_bytes": spec.entry_bytes(),
        "host_entry_bytes": spec.host_entry_bytes(),
        "draft_offset": (
            None if spec.draft_region is None else spec.draft_offset_in_entry()
        ),
        "max_slots": arena.num_entries(spec.name),
        "min_slot_index": arena.first_entry(spec.name),
    }
    if isinstance(spec, MambaSubPoolSpec):
        conv, temporal = spec.views(arena)
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
            d_conv, d_temporal = spec.draft_views(arena)
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
        k, v = spec.views(arena)
        out["views"] = {"k": _ends_summary(k), "v": _ends_summary(v)}
    else:
        assert isinstance(spec, MLASubPoolSpec), type(spec)
        out["views"] = {"kv": _ends_summary(spec.views(arena))}
    if spec.draft_region is not None:
        d_k, d_v = spec.draft_views(arena)
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
        member.ids.num_virtual_ids,
        member.entry_bytes_per_page,
    )


def _kv_pools(bundle) -> Dict[str, object]:
    router = bundle.token_to_kv_pool.layer_router
    pools = {
        sub_pool.name: router.view(sub_pool.name) for sub_pool in router.sub_pools()
    }
    state = bundle.token_to_kv_pool_allocator.state
    if state is not None:
        pools[state.sub_pool] = state.pool
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
    if isinstance(pool, MHAView):
        out["page_envelope"] = tuple(pool.get_page_envelope_buffer().shape)
    return out


def _layer_maps(bundle) -> dict:
    """Each token layer's index in its view; with a window, and whether it
    is a window layer."""
    router = bundle.token_to_kv_pool.layer_router
    owner = router.sub_pools()[0].name
    routes = {gid: router.route(gid) for gid in router.layer_ids()}
    if len(router.sub_pools()) > 1:
        kv_map = tuple(
            sorted(
                (gid, route.index, route.sub_pool != owner)
                for gid, route in routes.items()
            )
        )
    else:
        kv_map = tuple(sorted((gid, route.index) for gid, route in routes.items()))
    out = {"kv": kv_map}
    state = bundle.token_to_kv_pool_allocator.state
    if state is not None:
        out["state"] = tuple(sorted(state.layer_map.items()))
    return out


def _capacity_summary(bundle) -> dict:
    allocator = bundle.token_to_kv_pool_allocator
    out = {
        "available": allocator.available_size(),
        "size_full": allocator.size_full,
        "draft_virtual_id_space": allocator.draft_virtual_id_space,
    }
    if allocator.has_window_pool:
        out["size_swa"] = allocator.size_swa
    if allocator.state is not None:
        out["state_slots"] = allocator.state_slots.size
    return out


def unified_layout_summary(bundle) -> dict:
    """Plain data describing a built unified pool stack's byte layout: chain,
    entries, part offsets, views, envelopes, slot bounds and empty capacities."""
    arena = bundle.arena
    plan = bundle.token_to_kv_pool_allocator.storage.plan
    specs = [plan.sub_pool(band.name).spec for band in arena.bands]
    raw_base = arena.raw.data_ptr()
    out = {
        "chain": tuple((b.name, b.grow_direction) for b in arena.bands),
        "total_bytes": arena.total_bytes,
        "reserved_floor": plan.sink_bytes(),
        "active_allocation_bytes": arena.accessible_bytes,
        "sub_pools": {s.name: _sub_pool_summary(arena, s) for s in specs},
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
    if bundle.token_to_kv_pool_allocator.state is None:
        # Only a plan without state resizes after capture, where the backing
        # re-checks the stored floor.
        out["bs1_floor_terms"] = tuple(bundle.token_to_kv_pool._bs1_floor_terms)
    return out
