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

from sglang.srt.mem_cache.kv.routed_cache import RoutedKVCache
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.layout.fused_draft import (
    HOST_KINDS,
    LAYER_WINDOW,
    DraftStateRegion,
    FusedDraftPlacement,
    draft_state_layer_ids,
    draft_swa_layer_ids,
)
from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class UnifiedDraftKVPool(MHATokenToKVPool):
    """Dense draft KV over the draft parts of one host sub-pool's entries.

    Per-layer `k_buffer` / `v_buffer` are views of the draft parts inside
    every slot of the host sub-pool; ``layer_lanes`` maps each of this
    runner's layer ids to its region lane. Locs are the target's PHYSICAL
    token ids, translated through the id spaces of the allocator the draft
    runner shares with the target. Compaction moves whole page envelopes on
    the HOST pool, draft bytes included, so `move_kv_cache` raises rather
    than let a per-slot move corrupt the fused layout.
    """

    requires_physical_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        host_sub_pool_name: str,
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


class UnifiedDraftMambaPool(MambaPool):
    """A draft runner's recurrent state as views of the draft block fused into
    the host's state entries: `mamba_cache.conv[stream][layer]` is laid out
    exactly as a private `MambaPool`'s. Slot ids are the target's, translated
    by the shared allocator. Pure view: the host's whole-entry clear and copy
    already cover the draft block, so the block ops here act on the draft block alone, and it
    never calls `MambaPool.__init__` (no buffers of its own)."""

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        layer_lanes: Mapping[int, int],
    ):
        spec = unified_buffer.mamba_spec(sub_pool_name)
        region = spec.draft_region
        assert isinstance(region, DraftStateRegion), (
            f"UnifiedDraftMambaPool: sub-pool {sub_pool_name!r} carries no fused "
            "draft state region"
        )
        layer_ids = sorted(layer_lanes)
        assert layer_ids, "UnifiedDraftMambaPool binds at least one layer"
        lanes = [layer_lanes[layer_id] for layer_id in layer_ids]
        # A runner's lanes are one contiguous range, so its layers are one
        # strided slice of the region's views (a gather would copy).
        start = lanes[0]
        assert lanes == list(range(start, start + len(lanes))) and (
            0 <= start and start + len(lanes) <= region.lane_num
        ), (
            f"draft state lanes {lanes} must be a contiguous range within "
            f"{region.lane_num}"
        )
        conv_views, temporal_view = unified_buffer.build_draft_state_views(
            sub_pool_name
        )
        stop = start + len(lanes)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self.layer_lanes: Dict[int, int] = dict(layer_lanes)
        self.mamba_layer_ids = layer_ids
        self.num_mamba_layers = len(layer_ids)
        self._max_size = unified_buffer.max_slots(sub_pool_name) - 1
        self.size = self._max_size
        self.device = unified_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=False)
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        # Same disabled-state attributes `UnifiedMambaPool` replicates for the
        # base class's unconditional reads.
        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = 16
        self.replayssm_write_pos = None
        self.replayssm_is_kda = False
        self.enable_linear_replayssm_spec = False
        self.replayssm_spec_fold = False
        self.replayssm_cache_base = None
        self.replayssm_is_flush = None
        self.debug_memory_pool = False
        self.conv_shard_groups = None
        self.conv_slice_axis = spec.conv_slice_axis
        self.mamba_cache = self.State(
            conv=[view[start:stop] for view in conv_views],
            temporal=temporal_view[start:stop],
        )
        self.mem_usage = 0.0  # fused into the host entries

    def clear_slots(self, indices: torch.Tensor):
        """Zero THIS runner's state block in the given slots.

        The block only, never the whole entry: the host's own streams share
        the entry and must survive, since a new request takes over the
        draft's block while the target's state for that slot is still live.
        The host's whole-entry clear covers the block too, but only on a
        plain EXTEND forward: ModelRunner skips it on decode, target-verify
        and draft-extend, the modes the draft needs it in.
        """
        for view in self.mamba_cache.conv:
            view[:, indices] = 0
        self.mamba_cache.temporal[:, indices] = 0

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        """Copy-on-write THIS runner's state block, block only (see above)."""
        for view in self.mamba_cache.conv:
            view[:, dst_indices] = view[:, src_indices]
        self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
            :, src_indices
        ]

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "fused draft state relocates with the HOST pool's whole-entry move"
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft state has no contiguous region of its own; state "
            "transfer / disaggregation is unsupported."
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError(
            "CPU offloading is unsupported for fused draft state."
        )

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        raise NotImplementedError(
            "CPU offloading is unsupported for fused draft state."
        )


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
    modules. A runner the placement gives no state lanes would run any of
    these with no state pool of its own."""
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
    """What a draft runner binds: its KV pool over the fused lanes, the
    request table it reads (the target's), and the recurrent state its KV
    pool routes (the target's, until a state binder gives it its own)."""

    token_to_kv_pool: Any = None
    req_to_token_pool: Any = None
    state: Optional[StateStorage] = None


class BindContext(msgspec.Struct, frozen=True, kw_only=True):
    """One runner's inputs to every binder: the target's buffer, the name of
    its owner sub-pool (whose ids the batch's write locs carry), and its
    recurrent state."""

    placement: FusedDraftPlacement
    runner: int
    unified_buffer: Any
    owner: str
    target_state: Optional[StateStorage]
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
    owner: str,
    target_state: Optional[StateStorage],
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
        owner=owner,
        target_state=target_state,
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
    binding = FusedDraftBinding(req_to_token_pool=req_to_token_pool, state=target_state)
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
    bind the host that serves window layers when it holds lanes for this
    runner, else the full one, where a full lifetime covers any window.

    A runner whose layers all live in the owner sub-pool reads that sub-pool's
    view directly; any other runner routes each layer to its host's view, as
    the target's pool does, so the attention backends read a window layer
    through the window sub-pool's ids."""
    hosts = ctx.hosts_of("dense")
    kv_layer_ids = draft_kv_layer_ids(ctx.model)
    if len(hosts) == 1:
        layers_by_host = {hosts[0]: list(kv_layer_ids)}
    else:
        # Both kinds in one runner: a replicated head, whose config lists every
        # window layer (a per-depth runner's config is clipped to its block).
        window = set(draft_swa_layer_ids(ctx.model_config))
        layers_by_host = {
            host: [
                layer_id
                for layer_id in kv_layer_ids
                if (layer_id in window) == (HOST_KINDS[host].serves == LAYER_WINDOW)
            ]
            for host in hosts
        }
    lanes_by_host = {host: ctx.placement.lanes_for(ctx.runner, host) for host in hosts}
    assert all(
        len(layers_by_host[host]) == len(lanes_by_host[host]) for host in hosts
    ), (
        f"draft runner {ctx.runner}: layers {layers_by_host} vs placed lanes "
        f"{ {host: list(lanes) for host, lanes in lanes_by_host.items()} }"
    )
    if hosts == (ctx.owner,):
        pool = UnifiedDraftKVPool(
            unified_buffer=ctx.unified_buffer,
            host_sub_pool_name=ctx.owner,
            layer_lanes=dict(zip(layers_by_host[ctx.owner], lanes_by_host[ctx.owner])),
            page_size=ctx.page_size,
        )
        return msgspec.structs.replace(binding, token_to_kv_pool=pool)
    pool = routed_draft_pool(
        unified_buffer=ctx.unified_buffer,
        owner=ctx.owner,
        page_size=ctx.page_size,
        lanes_by_host={
            host: dict(zip(layers_by_host[host], lanes_by_host[host])) for host in hosts
        },
    )
    return msgspec.structs.replace(binding, token_to_kv_pool=pool)


def routed_draft_pool(
    *,
    unified_buffer: UnifiedKVPool,
    owner: str,
    page_size: int,
    lanes_by_host: Mapping[str, Mapping[int, int]],
) -> RoutedKVCache:
    """A draft runner's pool over several host sub-pools (or one other than
    the owner): each host's draft view holds the runner's layers there, and
    each layer routes to its host's view by its index among them, as the
    target's pool routes its own layers. ``lanes_by_host`` maps each host, in
    plan order, to the runner's model layer ids there and their lanes."""
    layer_maps = []
    for host, layer_lanes in lanes_by_host.items():
        layer_ids = sorted(layer_lanes)
        view = UnifiedDraftKVPool(
            unified_buffer=unified_buffer,
            host_sub_pool_name=host,
            layer_lanes={
                idx: layer_lanes[layer_id] for idx, layer_id in enumerate(layer_ids)
            },
            page_size=page_size,
        )
        layer_maps.append(
            (host, view, {layer_id: idx for idx, layer_id in enumerate(layer_ids)})
        )
    layer_ids = [layer_id for _, _, indices in layer_maps for layer_id in indices]
    lead = layer_maps[0][1]
    return RoutedKVCache(
        unified_buffer=unified_buffer,
        owner=owner,
        layer_maps=layer_maps,
        page_size=page_size,
        start_layer=min(layer_ids),
        end_layer=max(layer_ids),
        size=unified_buffer.max_slots(owner) - 1,
        head_num=lead.head_num,
        head_dim=lead.head_dim,
    )


register_draft_binder("dense", _bind_dense)


def _runner_state_layer_ids(ctx: BindContext) -> Tuple[int, ...]:
    """The recurrent-state layer ids this runner serves: its own depth for a
    per-depth head, every state layer otherwise."""
    mc = ctx.model_config
    num_depths = mc.num_nextn_predict_layers
    num_depths = 1 if num_depths is None else int(num_depths)
    layer_ids = draft_state_layer_ids(mc, num_depths=num_depths)
    if num_depths > 1:
        return (ctx.runner,) if ctx.runner in layer_ids else ()
    return layer_ids


def fused_draft_state(
    *,
    target: StateStorage,
    unified_buffer: UnifiedKVPool,
    layer_lanes: Mapping[int, int],
) -> StateStorage:
    """A draft runner's recurrent state over the draft block fused into the
    target's state entries: the target's slots (the draft's slot ids ARE the
    target's), and a view of the block keyed by the runner's own layer ids."""
    return StateStorage(
        sub_pool=target.sub_pool,
        pool=UnifiedDraftMambaPool(
            unified_buffer=unified_buffer,
            sub_pool_name=target.sub_pool,
            layer_lanes=layer_lanes,
        ),
        slots=target.slots,
        layer_map={layer_id: idx for idx, layer_id in enumerate(sorted(layer_lanes))},
        short_conv_pool=target.short_conv_pool,
        ngram_pool=target.ngram_pool,
    )


def _bind_state(ctx: BindContext, binding: FusedDraftBinding) -> FusedDraftBinding:
    """This runner's recurrent-state block, over the target's slots."""
    hosts = ctx.hosts_of("state")
    assert hosts == ("mamba",), (
        f"the state binder serves the 'mamba' host alone; got {hosts}"
    )
    target = ctx.target_state
    assert target is not None, (
        "a fused draft state block needs the target's unified state pool"
    )
    layer_ids = _runner_state_layer_ids(ctx)
    lanes = ctx.placement.lanes_for(ctx.runner, "mamba")
    assert len(layer_ids) == len(lanes), (
        f"draft runner {ctx.runner}: state layers {list(layer_ids)} vs placed "
        f"lanes {list(lanes)}"
    )
    state = fused_draft_state(
        target=target,
        unified_buffer=ctx.unified_buffer,
        layer_lanes=dict(zip(layer_ids, lanes)),
    )
    return msgspec.structs.replace(binding, state=state)


register_draft_binder("state", _bind_state)
