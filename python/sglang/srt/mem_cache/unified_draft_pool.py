"""KV pools a DRAFT runner binds over the draft lanes fused into the target's
unified pool: same pages, same slot ids, same v2p table as the target -- one
allocation, one free, one relocation.

`DRAFT_BINDERS` maps a `HostKind.family` to the binder that builds the
runner's pool over every host of that family; registering a binder is how a
new host kind reaches the draft worker.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import msgspec

from sglang.srt.mem_cache.kv.decl import WindowLifetime
from sglang.srt.mem_cache.kv.layout.mha import DraftMHAView
from sglang.srt.mem_cache.kv.layout.mla import DraftMLAView
from sglang.srt.mem_cache.kv.layout.state import DraftStateView, MambaSubPoolSpec
from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.mem_cache.kv.plan import KVStoragePlan
from sglang.srt.mem_cache.kv.routed_cache import RoutedKVCache
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_WINDOW,
    FusedDraftPlacement,
    MLADraftRegion,
    draft_state_layer_ids,
    draft_swa_layer_ids,
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
    """One runner's inputs to every binder: the target's arena and plan, the
    name of its owner sub-pool (whose ids the batch's write locs carry), and
    its recurrent state."""

    placement: FusedDraftPlacement
    runner: int
    arena: Any  # KVArena
    plan: Any  # KVStoragePlan
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
            if self.placement.host_kind(host).family == family
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
    arena: KVArena,
    plan: KVStoragePlan,
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
        arena=arena,
        plan=plan,
        owner=owner,
        target_state=target_state,
        model=model,
        model_config=model_config,
        page_size=page_size,
    )
    families: List[str] = []
    for host in placement.hosts():
        family = ctx.placement.host_kind(host).family
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
    """Bind draft model layer ids to the lanes sized from its configuration.

    Reject count mismatches to prevent aliasing. Window layers use the band
    whose retention matches their declared lifetime.
    Owner-only drafts use that view directly; other drafts route each layer to
    its host view and id space.
    """
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
                if (layer_id in window)
                == (ctx.placement.host_kind(host).serves == LAYER_WINDOW)
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
    if hosts == (ctx.owner,) and not isinstance(
        ctx.placement.region(ctx.owner), MLADraftRegion
    ):
        pool = DraftMHAView(
            arena=ctx.arena,
            host_spec=ctx.plan.sub_pool(ctx.owner).spec,
            layer_lanes=dict(zip(layers_by_host[ctx.owner], lanes_by_host[ctx.owner])),
            page_size=ctx.page_size,
        )
        return msgspec.structs.replace(binding, token_to_kv_pool=pool)
    pool = routed_draft_pool(
        arena=ctx.arena,
        plan=ctx.plan,
        owner=ctx.owner,
        page_size=ctx.page_size,
        lanes_by_host={
            host: dict(zip(layers_by_host[host], lanes_by_host[host])) for host in hosts
        },
    )
    return msgspec.structs.replace(binding, token_to_kv_pool=pool)


def routed_draft_pool(
    *,
    arena: KVArena,
    plan: KVStoragePlan,
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
        view_type = (
            DraftMLAView
            if isinstance(plan.sub_pool(host).spec.draft_region, MLADraftRegion)
            else DraftMHAView
        )
        view = view_type(
            arena=arena,
            host_spec=plan.sub_pool(host).spec,
            layer_lanes={
                idx: layer_lanes[layer_id] for idx, layer_id in enumerate(layer_ids)
            },
            page_size=page_size,
        )
        # Backends name the runner's window role "swa". Its view still names
        # the actual storage band, which RunnerIds resolves to that band's ids.
        route_name = (
            "swa" if isinstance(plan.sub_pool(host).lifetime, WindowLifetime) else host
        )
        layer_maps.append(
            (
                route_name,
                view,
                {layer_id: idx for idx, layer_id in enumerate(layer_ids)},
            )
        )
    layer_ids = [layer_id for _, _, indices in layer_maps for layer_id in indices]
    lead = layer_maps[0][1]
    return RoutedKVCache(
        arena=arena,
        owner=owner,
        layer_maps=layer_maps,
        page_size=page_size,
        start_layer=min(layer_ids),
        end_layer=max(layer_ids),
        size=arena.num_entries(owner) - 1,
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
    arena: KVArena,
    spec: MambaSubPoolSpec,
    layer_lanes: Mapping[int, int],
) -> StateStorage:
    """A draft runner's recurrent state over the draft block fused into the
    target's state entries: the target's slots (the draft's slot ids ARE the
    target's), and a view of the block keyed by the runner's own layer ids."""
    return StateStorage(
        sub_pool=target.sub_pool,
        pool=DraftStateView(
            arena=arena,
            spec=spec,
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
        arena=ctx.arena,
        spec=ctx.plan.sub_pool(target.sub_pool).spec,
        layer_lanes=dict(zip(layer_ids, lanes)),
    )
    return msgspec.structs.replace(binding, state=state)


register_draft_binder("state", _bind_state)
