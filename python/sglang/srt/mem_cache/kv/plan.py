"""The sub-pools a model's declarations need, their layouts and chain order.

Parts with the same lifetime and allocation unit share a sub-pool. In the
unified buffer the sub-pools form one chain ordered by lifetime rank: the
first grows up from the buffer's low end, the last grows down from its high
end, any between float.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Type

import msgspec

from sglang.srt.mem_cache.kv.decl import (
    FullLifetime,
    KVLayerDecl,
    KVLifetime,
    MHAGeometry,
    MLAGeometry,
    PartGeometry,
    StateGeometry,
    StateLifetime,
    WindowLifetime,
)
from sglang.srt.mem_cache.kv.layout.base import SubPoolSpec, _store_dtype_for
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.router import KV_PART, STATE_PART
from sglang.srt.mem_cache.layout.fused_draft import FusedDraftPlacement

# The one chain whose builder resizes the buffer after CUDA graph capture.
_RESIZED_AFTER_CAPTURE = frozenset({FullLifetime, WindowLifetime})


def resized_after_capture(lifetime_types: frozenset) -> bool:
    """Whether the unified pool of a plan with these lifetimes can resize its
    buffer after CUDA graph capture."""
    return lifetime_types == _RESIZED_AFTER_CAPTURE


# The draft placement and the id spaces name sub-pools by these.
_SUB_POOL_NAMES: Dict[Type, str] = {
    FullLifetime: "full",
    WindowLifetime: "swa",
    StateLifetime: "mamba",
}


class SubPoolExternal(msgspec.Struct, frozen=True, kw_only=True):
    """The names a sub-pool goes by outside this process, at the values peers,
    host stores and the radix tree already use: its HiCache host pool (also the
    suffix of its L3 keys), its radix tree component, and the PD state type it
    travels as (None for the sub-pool whose token KV PD registers as the
    transfer's KV entries)."""

    host_pool: str
    tree_component: str
    wire: Optional[str] = None


_EXTERNAL: Dict[Type, SubPoolExternal] = {
    FullLifetime: SubPoolExternal(host_pool="kv", tree_component="FULL"),
    WindowLifetime: SubPoolExternal(host_pool="swa", tree_component="SWA", wire="swa"),
    StateLifetime: SubPoolExternal(
        host_pool="mamba", tree_component="MAMBA", wire="mamba"
    ),
}

# The order PD peers list state entries in: a window's before recurrent state.
_WIRE_ORDER = ("swa", "mamba")


class WireEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One PD state entry: the sub-pool part it carries and its state type."""

    sub_pool: str
    part: str
    state_type: str


class PlannedSubPool(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    lifetime: KVLifetime
    layer_ids: Tuple[int, ...]
    geometry: PartGeometry
    spec: SubPoolSpec
    external: SubPoolExternal


class KVStoragePlan(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    # Chain order: the grow-up end first, the grow-down end last.
    sub_pools: Tuple[PlannedSubPool, ...]
    fused_draft: Optional[FusedDraftPlacement] = None

    def sub_pool(self, name: str) -> PlannedSubPool:
        for sub_pool in self.sub_pools:
            if sub_pool.name == name:
                return sub_pool
        raise KeyError(f"the plan has no sub-pool {name!r}")

    def specs(self) -> List[SubPoolSpec]:
        """Specs in the order the unified buffer takes them: the owner first."""
        return [s.spec for s in reversed(self.sub_pools)]

    def lifetime_types(self) -> frozenset:
        return frozenset(type(s.lifetime) for s in self.sub_pools)

    def of_lifetime(self, lifetime_type: Type) -> Optional[PlannedSubPool]:
        found = [s for s in self.sub_pools if isinstance(s.lifetime, lifetime_type)]
        assert len(found) <= 1, f"more than one {lifetime_type.__name__} sub-pool"
        return found[0] if found else None

    def owner(self) -> PlannedSubPool:
        """The sub-pool that owns the token id space."""
        return self.of_lifetime(FullLifetime)

    def owner_first(self) -> Tuple[PlannedSubPool, ...]:
        """The sub-pools from the owner to the grow-up end: the order host
        pools and transfers list them in."""
        return tuple(reversed(self.sub_pools))

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
        grow_direction=grow_direction,
        draft_region=draft_region,
    )


def plan_kv_storage(
    decls: Sequence[KVLayerDecl],
    *,
    page_size: int,
    fused_draft: Optional[FusedDraftPlacement] = None,
) -> KVStoragePlan:
    """The unified-buffer plan for these declarations."""
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    groups = _group_parts(decls, page_size)
    if len(groups) < 2:
        raise ValueError(
            "--enable-unified-memory shares one buffer between sub-pools of "
            f"different lifetimes; this model declares {len(groups)} "
            "(a plain or pure-window model)"
        )
    windows = {lt for lt, _ in groups if isinstance(lt, WindowLifetime)}
    if len(windows) > 1:
        raise ValueError(
            "--enable-unified-memory supports one sliding-window size per model; "
            f"this model declares {sorted(w.window for w in windows)}"
        )
    chain = sorted(groups, key=_chain_key)
    sub_pools = []
    for i, key in enumerate(chain):
        lifetime, _ = key
        layer_ids, geometry = groups[key]
        name = _SUB_POOL_NAMES[type(lifetime)]
        grow_direction = "up" if i == 0 else "down" if i == len(chain) - 1 else "float"
        sub_pools.append(
            PlannedSubPool(
                name=name,
                lifetime=lifetime,
                layer_ids=tuple(layer_ids),
                geometry=geometry,
                spec=_spec_for(
                    name=name,
                    geometry=geometry,
                    layer_num=len(layer_ids),
                    grow_direction=grow_direction,
                    fused_draft=fused_draft,
                ),
                external=_EXTERNAL[type(lifetime)],
            )
        )
    return KVStoragePlan(
        page_size=page_size, sub_pools=tuple(sub_pools), fused_draft=fused_draft
    )
