"""The sub-pools a model's declarations need, their layouts and chain order.

Parts with the same lifetime and allocation unit share a sub-pool. In the
unified buffer the sub-pools form one chain ordered by lifetime rank: the
first grows up from the buffer's low end, the last grows down from its high
end, any between float.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Type

import msgspec

from sglang.srt.mem_cache.kv.decl import (
    DerivedPartDecl,
    FullLifetime,
    KVLayerDecl,
    KVLifetime,
    MHAGeometry,
    MLAGeometry,
    PartGeometry,
    RequestPartDecl,
    ScratchDecl,
    StateGeometry,
    StateLifetime,
    TensorPartDecl,
    WindowLifetime,
)
from sglang.srt.mem_cache.kv.layout.base import SubPoolSpec, _store_dtype_for
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.router import KV_PART, STATE_PART
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_STATE,
    LAYER_WINDOW,
    DenseDraftRegion,
    DraftStateRegion,
    FusedDraftPlacement,
    MLADraftRegion,
)

# The draft placement and the id spaces name sub-pools by these.
_SUB_POOL_NAMES: Dict[Type, str] = {
    FullLifetime: "full",
    WindowLifetime: "swa",
    StateLifetime: "mamba",
}


class SubPoolExternal(msgspec.Struct, frozen=True, kw_only=True):
    """The names a sub-pool goes by outside this process, at the values peers,
    host stores, the radix tree and metrics already use: its HiCache host pool
    (also the suffix of its L3 keys), its radix tree component, the family its
    usage is reported under, and the PD state type it travels as (None for the
    sub-pool whose token KV PD registers as the transfer's KV entries)."""

    host_pool: str
    tree_component: str
    metrics: str
    wire: Optional[str] = None


_EXTERNAL: Dict[Type, SubPoolExternal] = {
    FullLifetime: SubPoolExternal(
        host_pool="kv", tree_component="FULL", metrics="full"
    ),
    WindowLifetime: SubPoolExternal(
        host_pool="swa", tree_component="SWA", metrics="swa", wire="swa"
    ),
    StateLifetime: SubPoolExternal(
        host_pool="mamba", tree_component="MAMBA", metrics="mamba", wire="mamba"
    ),
}

# The order PD peers list state entries in: a window's before recurrent state.
_WIRE_ORDER = (
    "minimax_index_k",
    "minimax_dense_kv",
    "swa",
    "auxiliary_swa",
    "block_scale",
    "block_scale_swa",
    "swa_ring",
    "c128_state",
    "mamba",
    "dsa",
    "dsa_tail",
    "qsa_pending",
    "qsa_compressed",
    "request_row",
)

# How a prefix tree keeps a sub-pool for a cached prefix: every token, the
# trailing window of tokens, one state checkpoint per grid step, or fixed-size
# blocks. The tree cores pick a slot's behavior by its family.
TREE_TOKEN = "token"
TREE_WINDOW = "window"
TREE_STATE = "state"
TREE_BLOCK = "block"

_TREE_FAMILY: Dict[Type, str] = {
    FullLifetime: TREE_TOKEN,
    WindowLifetime: TREE_WINDOW,
    StateLifetime: TREE_STATE,
}


class TreeSlot(msgspec.Struct, frozen=True, kw_only=True):
    """One slot of a prefix tree: the id the tree, its KV events and its host
    stores know it by (an external name), the family that decides how a cached
    prefix keeps it, and the numbers that family keeps it by."""

    component: str
    family: str
    # Window family: the tokens a prefix keeps back from its end, which every
    # reader of the window needs (its layers' attention window, widened by a
    # reader that reaches further back, such as a fused draft).
    window: Optional[int] = None
    # State family: a cached state is valid after a multiple of this many tokens.
    checkpoint_interval: Optional[int] = None


class WireEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One PD state entry: the sub-pool part it carries and its state type."""

    sub_pool: str
    part: str
    state_type: str


class PlannedSubPool(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    lifetime: KVLifetime
    # Entries per allocation unit: the page for token sub-pools, one entry (a
    # request's slot) for state.
    unit: int
    layer_ids: Tuple[int, ...]
    geometry: PartGeometry
    spec: SubPoolSpec
    external: SubPoolExternal


class KVStoragePlan(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    # Chain order: the grow-up end first, the grow-down end last.
    sub_pools: Tuple[PlannedSubPool, ...]
    fused_draft: Optional[FusedDraftPlacement] = None
    scratch: Tuple[ScratchDecl, ...] = ()
    request_parts: Tuple[RequestPartDecl, ...] = ()
    derived_parts: Tuple[DerivedPartDecl, ...] = ()

    def storage_format_tag(self) -> Optional[str]:
        """Separate persisted payloads by their layout and producer identity."""
        if self.fused_draft is None:
            return None
        import hashlib

        descriptor = repr((self.page_size, self.sub_pools, self.fused_draft.source))
        return "declared-" + hashlib.sha256(descriptor.encode()).hexdigest()[:24]

    def sub_pool(self, name: str) -> PlannedSubPool:
        for sub_pool in self.sub_pools:
            if sub_pool.name == name:
                return sub_pool
        raise KeyError(f"the plan has no sub-pool {name!r}")

    def specs(self) -> List[SubPoolSpec]:
        """Specs in the order the unified buffer takes them: the owner first."""
        return [s.spec for s in reversed(self.sub_pools)]

    def sink_bytes(self) -> int:
        """The padding sink at the low end of the buffer: every sub-pool's
        dummy writes (entry 0) land in it, so it spans the largest unit."""
        return max(s.unit * s.spec.entry_bytes() for s in self.sub_pools)

    def lifetime_types(self) -> frozenset:
        return frozenset(type(s.lifetime) for s in self.sub_pools)

    def of_lifetime(self, lifetime_type: Type) -> Optional[PlannedSubPool]:
        found = [s for s in self.sub_pools if isinstance(s.lifetime, lifetime_type)]
        if lifetime_type is WindowLifetime and len(found) > 1:
            return next(s for s in found if s.name == "swa")
        assert len(found) <= 1, f"more than one {lifetime_type.__name__} sub-pool"
        return found[0] if found else None

    def owner(self) -> PlannedSubPool:
        """The sub-pool that owns the token id space."""
        return self.of_lifetime(FullLifetime)

    def owner_first(self) -> Tuple[PlannedSubPool, ...]:
        """The sub-pools from the owner to the grow-up end: the order host
        pools and transfers list them in."""
        return tuple(reversed(self.sub_pools))

    def drive_order(self) -> Tuple[PlannedSubPool, ...]:
        """The sub-pools from the longest retention to the shortest: the owner,
        then windows from the largest, then state; ties by unit, then name.
        A prefix tree drives eviction and lists its components in this order."""
        return tuple(
            sorted(
                self.sub_pools,
                key=lambda s: (
                    tuple(-r for r in s.lifetime.retention()),
                    s.unit,
                    s.name,
                ),
            )
        )

    def tree_slots(self, *, window_reach: int = 0) -> Tuple[TreeSlot, ...]:
        """The prefix tree's slots, one per sub-pool, in drive order.
        ``window_reach`` is how far back from a request's end a reader beyond
        the window's own layers (a fused draft) reads the window."""
        return tuple(
            TreeSlot(
                component=s.external.tree_component,
                family=_TREE_FAMILY[type(s.lifetime)],
                window=(
                    max(s.lifetime.window, window_reach)
                    if isinstance(s.lifetime, WindowLifetime)
                    and s.lifetime.window is not None
                    else None
                ),
                checkpoint_interval=(
                    s.lifetime.checkpoint_interval
                    if isinstance(s.lifetime, StateLifetime)
                    else None
                ),
            )
            for s in self.drive_order()
        )

    def wire_entries(self) -> Tuple[WireEntry, ...]:
        """The PD state entries, in the order peers list them. The owner's
        token KV is the transfer's KV entries, not a state entry."""
        entries = [
            WireEntry(
                sub_pool=s.name,
                part=STATE_PART if isinstance(s.lifetime, StateLifetime) else KV_PART,
                state_type=s.external.wire,
            )
            for s in self.sub_pools
            if s.external.wire is not None
        ]
        groups = {}
        for declared in self.derived_parts:
            if declared.wire is not None:
                key = (declared.sub_pool, declared.wire)
                groups.setdefault(
                    key,
                    WireEntry(
                        sub_pool=declared.sub_pool,
                        part=declared.wire,
                        state_type=declared.wire,
                    ),
                )
        for declared in self.request_parts:
            if declared.wire is not None:
                key = (self.owner().name, declared.wire)
                groups.setdefault(
                    key,
                    WireEntry(
                        sub_pool=self.owner().name,
                        part=declared.wire,
                        state_type=declared.wire,
                    ),
                )
        entries.extend(groups.values())
        return tuple(sorted(entries, key=lambda e: _WIRE_ORDER.index(e.state_type)))


def _unit(lifetime: KVLifetime, page_size: int) -> int:
    return 1 if isinstance(lifetime, StateLifetime) else page_size


def _chain_key(key: Tuple[KVLifetime, int]) -> tuple:
    lifetime, unit = key
    rank, window = lifetime.retention()
    return (rank, -window, unit, _SUB_POOL_NAMES[type(lifetime)])


def _group_parts(
    decls: Sequence[KVLayerDecl], page_size: int
) -> Dict[Tuple[KVLifetime, int], Tuple[List[int], PartGeometry]]:
    groups: Dict[Tuple[KVLifetime, int], Tuple[List[int], PartGeometry]] = {}
    for decl in decls:
        for part in decl.parts:
            key = (part.lifetime, _unit(part.lifetime, page_size))
            layer_ids, geometry = groups.setdefault(key, ([], part.geometry))
            if geometry != part.geometry:
                raise ValueError(
                    f"layers of one {type(part.lifetime).__name__} sub-pool declare "
                    f"two geometries ({geometry} and {part.geometry}); the unified "
                    "pool lays out one geometry per sub-pool"
                )
            layer_ids.append(decl.layer_id)
    return groups


def sub_pool_names(decls: Sequence[KVLayerDecl]) -> Tuple[str, ...]:
    """The sub-pools a plan of these declarations holds, owner first: what the
    fused-draft placement needs before the plan exists."""
    keys = _group_parts(decls, page_size=1)
    return tuple(_SUB_POOL_NAMES[type(lt)] for lt, _ in sorted(keys, key=_chain_key))[
        ::-1
    ]


def _spec_for(
    *,
    name: str,
    geometry: PartGeometry,
    layer_num: int,
    grow_direction: str,
    fused_draft: Optional[FusedDraftPlacement],
) -> SubPoolSpec:
    draft_region = None if fused_draft is None else fused_draft.region(name)
    if isinstance(geometry, MHAGeometry):
        return MHASubPoolSpec(
            name=name,
            layer_num=layer_num,
            head_num=geometry.head_num,
            head_dim=geometry.head_dim,
            v_head_dim=geometry.v_head_dim,
            store_dtype=_store_dtype_for(geometry.kv_cache_dtype),
            kv_cache_dtype=geometry.kv_cache_dtype,
            grow_direction=grow_direction,
            draft_region=draft_region,
        )
    if isinstance(geometry, MLAGeometry):
        return MLASubPoolSpec(
            name=name,
            layer_num=layer_num,
            kv_lora_rank=geometry.kv_lora_rank,
            qk_rope_head_dim=geometry.qk_rope_head_dim,
            store_dtype=_store_dtype_for(geometry.kv_cache_dtype),
            grow_direction=grow_direction,
            draft_region=draft_region,
        )
    assert isinstance(geometry, StateGeometry), f"unknown geometry {geometry}"
    return MambaSubPoolSpec(
        name=name,
        layer_num=layer_num,
        conv_state_shapes=geometry.conv_state_shapes,
        conv_dtype=geometry.conv_dtype,
        temporal_state_shape=geometry.temporal_state_shape,
        temporal_dtype=geometry.temporal_dtype,
        conv_slice_axis=geometry.conv_slice_axis,
        extra_parts=geometry.extra_parts,
        replay=geometry.replay,
        grow_direction=grow_direction,
        draft_region=draft_region,
    )


def plan_kv_storage(
    decls: Sequence[KVLayerDecl],
    *,
    page_size: int,
    fused_draft: Optional[FusedDraftPlacement] = None,
    speculative_num_draft_tokens: Optional[int] = None,
    request_parts: Sequence[RequestPartDecl] = (),
    derived_parts: Sequence[DerivedPartDecl] = (),
) -> KVStoragePlan:
    """The unified-buffer plan for these declarations."""
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    groups = _group_parts(decls, page_size)
    if not any(isinstance(lt, FullLifetime) for lt, _ in groups):
        raise ValueError("unified storage needs a FULL virtual-id owner")
    if len({lt for lt, _ in groups if isinstance(lt, WindowLifetime)}) > 1:
        raise ValueError("multiple target windows require named model declarations")
    names_by_key = {key: _SUB_POOL_NAMES[type(key[0])] for key in groups}
    if fused_draft is not None:
        for name, band in fused_draft.new_bands.items():
            region = fused_draft.region(name)
            if band.kind == LAYER_WINDOW:
                lifetime = WindowLifetime(window=band.window)
                assert isinstance(region, (DenseDraftRegion, MLADraftRegion))
                geometry = (
                    MLAGeometry(
                        kv_lora_rank=region.kv_lora_rank,
                        qk_rope_head_dim=region.qk_rope_head_dim,
                        kv_cache_dtype=region.resolved_kv_dtype(),
                    )
                    if isinstance(region, MLADraftRegion)
                    else MHAGeometry(
                        head_num=region.head_num,
                        head_dim=region.head_dim,
                        v_head_dim=region.resolved_v_head_dim(),
                        kv_cache_dtype=region.resolved_kv_dtype(),
                    )
                )
            else:
                assert band.kind == LAYER_STATE and isinstance(region, DraftStateRegion)
                lifetime = StateLifetime()
                geometry = StateGeometry(
                    conv_state_shapes=region.state.conv_state_shapes,
                    conv_dtype=region.state.conv_dtype,
                    temporal_state_shape=region.state.temporal_state_shape,
                    temporal_dtype=region.state.temporal_dtype,
                )
            key = (lifetime, _unit(lifetime, page_size))
            if key in groups:
                raise ValueError(f"draft band {name!r} duplicates a target lifetime")
            groups[key] = ([], geometry)
            names_by_key[key] = name
    chain = sorted(groups, key=_chain_key)
    sub_pools = []
    for i, key in enumerate(chain):
        lifetime, unit = key
        layer_ids, geometry = groups[key]
        name = names_by_key[key]
        grow_direction = "down" if i == len(chain) - 1 else "up" if i == 0 else "float"
        sub_pools.append(
            PlannedSubPool(
                name=name,
                lifetime=lifetime,
                unit=unit,
                layer_ids=tuple(layer_ids),
                geometry=geometry,
                spec=_spec_for(
                    name=name,
                    geometry=geometry,
                    layer_num=len(layer_ids),
                    grow_direction=grow_direction,
                    fused_draft=fused_draft,
                ),
                external=(
                    SubPoolExternal(
                        host_pool="auxiliary_swa",
                        tree_component="AUXILIARY_SWA",
                        metrics="swa",
                        wire="auxiliary_swa",
                    )
                    if name == "draft_swa"
                    else _EXTERNAL[type(lifetime)]
                ),
            )
        )
    request_parts = list(request_parts)
    for sub_pool in sub_pools:
        if isinstance(sub_pool.geometry, StateGeometry):
            request_parts.extend(
                RequestPartDecl(
                    part=msgspec.structs.replace(
                        part, name=f"{sub_pool.name}.{part.name}"
                    )
                )
                for part in sub_pool.geometry.request_parts
            )
    scratch = [
        ScratchDecl(
            part=msgspec.structs.replace(part, name=f"{sub_pool.name}.{part.name}")
        )
        for sub_pool in sub_pools
        if isinstance(sub_pool.geometry, StateGeometry)
        for part in sub_pool.geometry.scratch_parts
    ]
    if speculative_num_draft_tokens is not None:
        for sub_pool in sub_pools:
            if (
                not isinstance(sub_pool.geometry, StateGeometry)
                or not sub_pool.layer_ids
            ):
                continue
            geometry = sub_pool.geometry
            for name, shape, dtype in (
                (
                    "intermediate_ssm",
                    geometry.temporal_state_shape,
                    geometry.temporal_dtype,
                ),
                *(
                    (f"intermediate_conv_window.{i}", shape, geometry.conv_dtype)
                    for i, shape in enumerate(geometry.conv_state_shapes)
                ),
            ):
                if (
                    name == "intermediate_ssm"
                    and geometry.replay is not None
                    and geometry.replay.speculative
                ):
                    continue
                scratch.append(
                    ScratchDecl(
                        part=TensorPartDecl(
                            name=f"{sub_pool.name}.{name}",
                            shape=(speculative_num_draft_tokens, *shape),
                            dtype=dtype,
                            layer_ids=sub_pool.layer_ids,
                        )
                    )
                )
    names = {s.name for s in sub_pools}
    for derived in derived_parts:
        if derived.sub_pool not in names:
            raise ValueError(
                f"derived part {derived.part.name!r} has no owning sub-pool"
            )
    wire_owners = {}
    for declared in (*derived_parts, *request_parts):
        if declared.wire is None:
            continue
        if declared.wire not in _WIRE_ORDER:
            raise ValueError(f"unknown transfer wire name {declared.wire!r}")
        owner = (
            ("page", declared.sub_pool)
            if isinstance(declared, DerivedPartDecl)
            else ("request",)
        )
        if wire_owners.setdefault(declared.wire, owner) != owner:
            raise ValueError(
                f"wire component {declared.wire!r} mixes allocation domains"
            )
    return KVStoragePlan(
        page_size=page_size,
        sub_pools=tuple(sub_pools),
        fused_draft=fused_draft,
        scratch=tuple(scratch),
        request_parts=tuple(request_parts),
        derived_parts=tuple(derived_parts),
    )
