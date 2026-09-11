"""KV pools a DRAFT runner binds over the draft lanes fused into the target's
unified pool: same pages, same slot ids, same v2p table as the target -- one
allocation, one free, one relocation.

`DRAFT_BINDERS` maps a `HostKind.family` to the binder that builds the
runner's pool over every host of that family; registering a binder is how a
new host kind reaches the draft worker.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.layout.fused_draft import (
    HOST_KINDS,
    FusedDraftPlacement,
    draft_swa_layer_ids,
)
from sglang.srt.mem_cache.memory_pool import (
    KVWriteLoc,
    MHATokenToKVPool,
    unwrap_write_loc,
    write_loc_is_physical,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool


class UnifiedDraftKVPool(MHATokenToKVPool):
    """Dense draft KV over the draft parts of one host sub-pool's entries.

    Per-layer `k_buffer` / `v_buffer` are views of the draft parts inside
    every slot of the host sub-pool; ``layer_lanes`` maps each of this
    runner's layer ids to its region lane. Locs are the target's PHYSICAL
    token ids, translated through `host_allocator`. Compaction moves whole
    page envelopes on the HOST pool, draft bytes included, so `move_kv_cache`
    raises rather than let a per-slot move corrupt the fused layout.
    """

    requires_physical_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        host_sub_pool_name: str,
        host_allocator,
        layer_lanes: Mapping[int, int],
        page_size: int = 1,
    ):
        region = unified_buffer.require_draft_host_spec(host_sub_pool_name).draft_region
        layer_ids = sorted(layer_lanes)
        assert layer_ids, "UnifiedDraftKVPool binds at least one layer"
        start_layer = layer_ids[0]
        assert layer_ids == list(range(start_layer, start_layer + len(layer_ids))), (
            f"fused draft layer ids must be contiguous; got {layer_ids}"
        )
        lanes = [layer_lanes[layer_id] for layer_id in layer_ids]
        assert len(set(lanes)) == len(lanes) and all(
            0 <= s < region.lane_num for s in lanes
        ), f"draft lanes {lanes} must be distinct and within range({region.lane_num})"
        k_views, v_views = unified_buffer.build_dense_draft_views(host_sub_pool_name)
        max_slots = unified_buffer.max_slots(host_sub_pool_name)

        self._unified_buffer = unified_buffer
        self._host_sub_pool_name = host_sub_pool_name
        self.host_allocator = host_allocator
        self.layer_lanes: Dict[int, int] = dict(layer_lanes)
        self._k_views: List[torch.Tensor] = [k_views[s] for s in lanes]
        self._v_views: List[torch.Tensor] = [v_views[s] for s in lanes]
        num_pages = max_slots // page_size

        super().__init__(
            size=num_pages * page_size - page_size,
            page_size=page_size,
            # The KV dtype, not its storage: `set_kv_buffer` casts to `dtype`
            # and views the result as `store_dtype` (uint8 for fp8).
            dtype=region.resolved_kv_dtype(),
            head_num=region.head_num,
            head_dim=region.head_dim,
            layer_num=len(layer_ids),
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            v_head_dim=region.resolved_v_head_dim(),
            start_layer=start_layer,
            end_layer=start_layer + len(layer_ids) - 1,
            enable_kv_cache_copy=False,
            kv_cache_layout="page_major",
        )

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views

    def _clear_buffers(self):
        pass  # lifetime owned by UnifiedKVPool

    def get_kv_size_bytes(self):
        return 0, 0  # fused into the host entries; UnifiedKVPool logs the total

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "fused draft KV relocates with the HOST pool's whole-page move; a "
            "draft-side per-slot move would corrupt the fused layout"
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported."
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")


class UnifiedDraftSWAKVPool(SWAKVPool):
    """A draft's KV pool when it has sliding-window layers: one dense draft
    pool per host sub-pool it fuses into ("full" and/or "swa"), routed per
    layer exactly like the target's `UnifiedSWAKVPool`, so the attention
    backends run their swa rail for it. Inherits `SWAKVPool` for `isinstance`
    only; never calls its `__init__` (it would allocate static pools)."""

    requires_physical_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        host_allocator,
        page_size: int,
        full_layer_lanes: Mapping[int, int],
        swa_layer_lanes: Mapping[int, int],
    ):
        assert swa_layer_lanes, (
            "UnifiedDraftSWAKVPool binds at least one window layer; a full-only "
            "draft binds UnifiedDraftKVPool"
        )
        self.unified_buffer = unified_buffer
        self.host_allocator = host_allocator
        self.page_size = page_size
        self.device = unified_buffer.device
        self.layer_transfer_counter = None
        self.full_layer_nums = len(full_layer_lanes)
        self.swa_layer_nums = len(swa_layer_lanes)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = min([*full_layer_lanes, *swa_layer_lanes])
        self.size = unified_buffer.max_slots("full") - 1
        self.size_swa = unified_buffer.max_slots("swa") - 1

        self.full_kv_pool: Optional[UnifiedDraftKVPool] = None
        if full_layer_lanes:
            self.full_kv_pool = self._side_pool("full", full_layer_lanes)
        self.swa_kv_pool = self._side_pool("swa", swa_layer_lanes)
        lead = self.full_kv_pool if self.full_kv_pool is not None else self.swa_kv_pool
        self.dtype = lead.dtype
        self.head_num = lead.head_num
        self.head_dim = lead.head_dim

        # {layer_id: (per-side index, is_swa_layer)}, sides indexed in layer order.
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for idx, layer_id in enumerate(sorted(full_layer_lanes)):
            self.layers_mapping[layer_id] = (idx, False)
        for idx, layer_id in enumerate(sorted(swa_layer_lanes)):
            self.layers_mapping[layer_id] = (idx, True)
        # None so dispatch routes through the host's v2p tables, never a
        # registered mapping.
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None
        self.mem_usage = 0.0  # fused into the host entries

    def _side_pool(
        self, host_sub_pool_name: str, layer_lanes: Mapping[int, int]
    ) -> UnifiedDraftKVPool:
        # The side is indexed 0..n-1 in layer order (`layer_id_override`).
        return UnifiedDraftKVPool(
            unified_buffer=self.unified_buffer,
            host_sub_pool_name=host_sub_pool_name,
            host_allocator=self.host_allocator,
            layer_lanes={
                idx: layer_lanes[layer_id]
                for idx, layer_id in enumerate(sorted(layer_lanes))
            },
            page_size=self.page_size,
        )

    def _side(self, layer_id: int) -> Tuple[UnifiedDraftKVPool, int]:
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        assert pool is not None, layer_id
        return pool, pool_layer_id

    # -- KVCache surface a side of None cannot answer --

    @property
    def post_capture_active(self) -> bool:
        return False  # the host buffer is fully backed at boot

    @property
    def post_capture_backed_bytes(self) -> int:
        return 0

    def finalize_backing(self, config) -> None:
        return

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    # -- BaseSWAKVPool ABC surface --

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        return  # the host's swa v2p table IS the mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        """Virtual token ids -> swa-physical token ids, through the host."""
        return self.host_allocator.translate_loc_from_full_to_swa(kv_indices)

    def get_state_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported."
        )

    # -- size/info --

    def get_kv_size_bytes(self):
        return 0, 0  # fused into the host entries; UnifiedKVPool logs the total

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported."
        )

    def get_v_head_dim(self):
        lead = self.full_kv_pool if self.full_kv_pool is not None else self.swa_kv_pool
        return lead.get_value_buffer(lead.start_layer).shape[-1]

    # -- buffer accessors --

    def get_key_buffer(self, layer_id: int):
        pool, pool_layer_id = self._side(layer_id)
        return pool.get_key_buffer(pool_layer_id)

    def get_value_buffer(self, layer_id: int):
        pool, pool_layer_id = self._side(layer_id)
        return pool.get_value_buffer(pool_layer_id)

    def get_kv_buffer(self, layer_id: int):
        pool, pool_layer_id = self._side(layer_id)
        return pool.get_kv_buffer(pool_layer_id)

    # -- kv writing --

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        """Route to the right side. Both locs are physical already (the
        backend derives `swa_loc` once per forward); never translates here.
        The full side writes through `loc`: the capture-stable `full_loc`
        alias is sized for the target's writes, not a per-step draft slice."""
        loc, swa_loc, _ = unwrap_write_loc(loc_info)
        physical = write_loc_is_physical(loc_info)
        pool, pool_layer_id = self._side(layer.layer_id)
        if pool is self.swa_kv_pool:
            assert swa_loc is not None, (
                "UnifiedDraftSWAKVPool.set_kv_buffer: window layer received no "
                "swa_loc; the attention backend must bundle "
                "forward_metadata.swa_out_cache_loc."
            )
            side_loc = swa_loc
        else:
            side_loc = loc
        pool.set_kv_buffer(
            None,
            KVWriteLoc(side_loc, physical=physical),
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=pool_layer_id,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "fused draft KV relocates with the HOST pool's whole-page move; a "
            "draft-side per-slot move would corrupt the fused layout"
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")


def fused_draft_host_allocator(token_to_kv_pool: Any) -> Optional[Any]:
    """The host allocator a fused draft pool translates through, or None for
    any other pool."""
    if isinstance(token_to_kv_pool, (UnifiedDraftKVPool, UnifiedDraftSWAKVPool)):
        return token_to_kv_pool.host_allocator
    return None


def draft_kv_layer_ids(model) -> List[int]:
    """Layer ids owning attention KV in the BUILT draft model, in layer order.
    Window layers count -- a window is a parameter of the attention layer;
    linear and recurrent layers have no attention module and are absent."""
    from sglang.srt.layers.radix_attention import RadixAttention

    return sorted(
        {m.layer_id for m in model.modules() if isinstance(m, RadixAttention)}
    )


def draft_state_layer_classes(model) -> List[str]:
    """Class names of the BUILT draft model's recurrent / linear-attention
    modules. A fused draft has KV lanes only, so any of these would run with
    no state pool of its own."""
    from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

    return sorted(
        {
            type(m).__name__
            for m in model.modules()
            if isinstance(m, (MambaMixer2, RadixLinearAttention))
        }
    )


class FusedDraftBinding(msgspec.Struct, frozen=True, kw_only=True):
    """What a draft runner binds: its KV pool over the fused lanes and the
    request table it reads."""

    token_to_kv_pool: Any = None
    req_to_token_pool: Any = None


class BindContext(msgspec.Struct, frozen=True, kw_only=True):
    """One runner's inputs to every binder."""

    placement: FusedDraftPlacement
    runner: int
    unified_buffer: Any
    host_allocator: Any
    model: Any
    model_config: Any
    page_size: int

    def hosts_of(self, family: str) -> Tuple[str, ...]:
        """This runner's hosts of ``family`` that hold lanes for it."""
        return tuple(
            host
            for host in self.placement.hosts()
            if HOST_KINDS[host].family == family
            and len(self.placement.lanes_for(self.runner, host)) > 0
        )


Binder = Callable[[BindContext, FusedDraftBinding], FusedDraftBinding]

DRAFT_BINDERS: Dict[str, Binder] = {}


def register_draft_binder(family: str, binder: Binder) -> Binder:
    assert family not in DRAFT_BINDERS, (
        f"a draft binder is already registered for host family {family!r}"
    )
    DRAFT_BINDERS[family] = binder
    return binder


def bind_fused_draft(
    *,
    placement: FusedDraftPlacement,
    runner: int,
    unified_buffer: UnifiedKVPool,
    host_allocator,
    model,
    model_config,
    req_to_token_pool,
    page_size: int,
) -> FusedDraftBinding:
    """Bind draft runner ``runner`` to its fused lanes: one binder per host
    family that holds lanes for it, in registry order, each refining the
    binding the previous one returned."""
    ctx = BindContext(
        placement=placement,
        runner=runner,
        unified_buffer=unified_buffer,
        host_allocator=host_allocator,
        model=model,
        model_config=model_config,
        page_size=page_size,
    )
    families: List[str] = []
    for host in placement.hosts():
        family = HOST_KINDS[host].family
        if family not in families and len(placement.lanes_for(runner, host)) > 0:
            families.append(family)
    assert families, f"draft runner {runner} holds no lane in any host"
    binding = FusedDraftBinding(req_to_token_pool=req_to_token_pool)
    for family in families:
        assert family in DRAFT_BINDERS, (
            f"no draft binder is registered for host family {family!r}"
        )
        binding = DRAFT_BINDERS[family](ctx, binding)
    assert binding.token_to_kv_pool is not None, (
        f"draft runner {runner}: no binder produced a KV pool"
    )
    return binding


def _bind_dense(ctx: BindContext, binding: FusedDraftBinding) -> FusedDraftBinding:
    """This runner's attention layers, in layer order, over its lanes in the
    dense hosts. The placement sized the lanes from the draft config; the
    model's real layer ids fill them, so a count mismatch is a loud boot
    failure, never a silent alias. Window layers follow the placement: they
    bind the swa sub-pool when it holds a region for them, else the full
    sub-pool, where a full lifetime covers any window."""
    hosts = ctx.hosts_of("dense")
    assert set(hosts) <= {"full", "swa"}, f"unknown dense host(s) in {hosts}"
    kv_layer_ids = draft_kv_layer_ids(ctx.model)
    full_lanes = ctx.placement.lanes_for(ctx.runner, "full")
    swa_lanes = ctx.placement.lanes_for(ctx.runner, "swa")
    if not swa_lanes:
        full_ids, swa_ids = list(kv_layer_ids), []
    elif not full_lanes:
        full_ids, swa_ids = [], list(kv_layer_ids)
    else:
        # Both kinds in one runner: a replicated head, whose config lists every
        # window layer (a per-depth runner's config is clipped to its block).
        window = set(draft_swa_layer_ids(ctx.model_config))
        full_ids = [layer_id for layer_id in kv_layer_ids if layer_id not in window]
        swa_ids = [layer_id for layer_id in kv_layer_ids if layer_id in window]
    assert len(full_ids) == len(full_lanes) and len(swa_ids) == len(swa_lanes), (
        f"draft runner {ctx.runner}: layers full={full_ids} swa={swa_ids} vs placed "
        f"lanes full={list(full_lanes)} swa={list(swa_lanes)}"
    )
    if not swa_ids:
        pool = UnifiedDraftKVPool(
            unified_buffer=ctx.unified_buffer,
            host_sub_pool_name="full",
            host_allocator=ctx.host_allocator,
            layer_lanes=dict(zip(full_ids, full_lanes)),
            page_size=ctx.page_size,
        )
    else:
        pool = UnifiedDraftSWAKVPool(
            unified_buffer=ctx.unified_buffer,
            host_allocator=ctx.host_allocator,
            page_size=ctx.page_size,
            full_layer_lanes=dict(zip(full_ids, full_lanes)),
            swa_layer_lanes=dict(zip(swa_ids, swa_lanes)),
        )
    return msgspec.structs.replace(binding, token_to_kv_pool=pool)


register_draft_binder("dense", _bind_dense)
