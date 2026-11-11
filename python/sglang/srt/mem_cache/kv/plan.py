"""The sub-pools a model's declarations need, their layouts and chain order.

Parts with the same lifetime and allocation unit share a sub-pool. A sub-pool
whose layers declare one geometry gets that geometry's layout; one whose
layers declare several (a layer's KV and a part it keeps beside it) gets a
composite unit holding each, in declaration order. In the unified buffer the
sub-pools form one chain ordered by lifetime rank: the first grows up from the
buffer's low end, the last grows down from its high end, any between float.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Type

import msgspec

from sglang.srt.mem_cache.kv.decl import (
    KV_PART,
    STATE_PART,
    DerivedPartDecl,
    FullLifetime,
    KVLayerDecl,
    KVLifetime,
    MHAGeometry,
    MLAGeometry,
    PartGeometry,
    RequestGeometry,
    RequestLifetime,
    ScratchDecl,
    StateGeometry,
    StateLifetime,
    TensorPartDecl,
    WindowLifetime,
    primary_window,
    window_sub_pool_name,
)
from sglang.srt.mem_cache.kv.layout.base import SubPoolSpec, _store_dtype_for
from sglang.srt.mem_cache.kv.layout.composite import (
    CompositeSubPoolSpec,
    TokenRows,
    UnitPart,
    UnitPartGeometry,
)
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.mla import MLASubPoolSpec
from sglang.srt.mem_cache.kv.layout.request import RequestSubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_STATE,
    LAYER_WINDOW,
    DenseDraftRegion,
    DraftStateRegion,
    FusedDraftPlacement,
    MLADraftRegion,
)

# The draft placement and the id spaces name sub-pools by these; windows are
# named by `window_sub_pool_name`.
_SUB_POOL_NAMES: Dict[Type, str] = {
    FullLifetime: "full",
    StateLifetime: "mamba",
    RequestLifetime: "request",
}


class SubPoolExternal(msgspec.Struct, frozen=True, kw_only=True):
    """The names a sub-pool goes by outside this process, at the values peers,
    host stores, the radix tree and metrics already use: its HiCache host pool
    (also the suffix of its L3 keys), its radix tree component, the family its
    usage is reported under and the label within it, the PD state type it
    travels as (None for the sub-pool whose token KV PD registers as the
    transfer's KV entries), and the key PD peers tell its entries apart by.
    Renaming a sub-pool inside this process changes none of them unless they
    are derived from its name."""

    host_pool: str
    tree_component: str
    metrics: str
    label: str
    wire: Optional[str] = None
    wire_key: Optional[str] = None


_EXTERNAL: Dict[Type, SubPoolExternal] = {
    FullLifetime: SubPoolExternal(
        host_pool="kv",
        tree_component="FULL",
        metrics="full",
        label="full",
        wire_key="full",
    ),
    StateLifetime: SubPoolExternal(
        host_pool="mamba",
        tree_component="MAMBA",
        metrics="mamba",
        label="mamba",
        wire="mamba",
        wire_key="mamba",
    ),
    # Never cached, stored on the host or sent to a PD peer: its names exist for
    # metrics only.
    RequestLifetime: SubPoolExternal(
        host_pool="request",
        tree_component="REQUEST",
        metrics="request",
        label="request",
    ),
}

# A window's prefix tree component and host pool are named by its sub-pool.

# The PD state type every window travels as: its trailing pages. A peer tells
# one window's entry from another's by the entry's sub-pool.
WINDOW_WIRE = "swa"


def window_tree_component(name: str) -> str:
    """The prefix tree component a window sub-pool goes by."""
    return name.upper()


def _window_external(name: str) -> SubPoolExternal:
    return SubPoolExternal(
        host_pool=name,
        tree_component=window_tree_component(name),
        metrics="window",
        label=name,
        wire=WINDOW_WIRE,
        wire_key=name,
    )


# The order PD peers list state entries in: a window's before recurrent state.
_WIRE_ORDER = (
    "minimax_index_k",
    "minimax_dense_kv",
    WINDOW_WIRE,
    "block_scale",
    "block_scale_swa",
    "swa_ring",
    "c128_state",
    "mamba",
    "dsa",
    "dsa_tail",
    "qsa_pending",
    "qsa_compressed",
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
    # The host pool keeping the slot's HiCache copies, by name; None means the
    # family's own (the window host pool for a window).
    host_pool: Optional[str] = None


class WireEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One PD state entry: the sub-pool part it carries, its state type, and
    the key peers tell entries of one type apart by (the sub-pool's)."""

    sub_pool: str
    part: str
    state_type: str
    key: Optional[str] = None


class PlannedPart(msgspec.Struct, frozen=True, kw_only=True):
    """One geometry a composite sub-pool holds: the part name it is routed by,
    and the layers that declare it."""

    name: str
    geometry: PartGeometry
    layer_ids: Tuple[int, ...]


class PlannedSubPool(
    msgspec.Struct, frozen=True, kw_only=True, repr_omit_defaults=True
):
    name: str
    lifetime: KVLifetime
    # Entries per allocation unit: the page for token sub-pools, one entry (a
    # request's slot or row) for state and request sub-pools.
    unit: int
    # The sub-pool's first geometry (its token KV) and the layers declaring it.
    layer_ids: Tuple[int, ...]
    geometry: PartGeometry
    spec: SubPoolSpec
    external: SubPoolExternal
    # Every geometry of a composite sub-pool, the first one included; empty
    # when the sub-pool's layers declare one geometry.
    parts: Tuple[PlannedPart, ...] = ()

    def part(self, name: str) -> PlannedPart:
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"sub-pool {self.name!r} holds no part {name!r}")


class KVStoragePlan(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    # Chain order: the grow-up end first, the grow-down end last.
    sub_pools: Tuple[PlannedSubPool, ...]
    fused_draft: Optional[FusedDraftPlacement] = None
    scratch: Tuple[ScratchDecl, ...] = ()
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
        """The plan's one sub-pool of a lifetime type; windows can be several,
        see `windows` and `primary_window`."""
        assert lifetime_type is not WindowLifetime, "a plan may hold several windows"
        found = [s for s in self.sub_pools if isinstance(s.lifetime, lifetime_type)]
        assert len(found) <= 1, f"more than one {lifetime_type.__name__} sub-pool"
        return found[0] if found else None

    def windows(self) -> Tuple[PlannedSubPool, ...]:
        """The window sub-pools in window-slot order: the primary window, then
        the rest in drive order."""
        primary = self.primary_window()
        rest = [
            s
            for s in self.drive_order()
            if isinstance(s.lifetime, WindowLifetime) and s is not primary
        ]
        return tuple([primary] * (primary is not None) + rest)

    def primary_window(self) -> Optional[PlannedSubPool]:
        """The longest window the target's layers declare, or the longest
        window when the target declares none: the window a single-window reader
        sizes and retains by."""
        drafts = set() if self.fused_draft is None else set(self.fused_draft.new_bands)
        windows = [s for s in self.sub_pools if isinstance(s.lifetime, WindowLifetime)]
        candidates = [s for s in windows if s.name not in drafts] or windows
        if not candidates:
            return None
        length = primary_window(s.lifetime.window for s in candidates)
        return next(s for s in candidates if s.lifetime.window == length)

    def owner(self) -> PlannedSubPool:
        """The sub-pool that owns the token id space."""
        return self.of_lifetime(FullLifetime)

    def owner_first(self) -> Tuple[PlannedSubPool, ...]:
        """The sub-pools from the owner to the grow-up end: the order host
        pools and transfers list them in."""
        return tuple(reversed(self.sub_pools))

    def requests(self) -> Tuple[PlannedSubPool, ...]:
        """The request sub-pools: one unit per bound request row, outside the
        prefix tree, host stores and PD transfers."""
        return tuple(
            s for s in self.sub_pools if isinstance(s.lifetime, RequestLifetime)
        )

    def cached_sub_pools(self) -> Tuple[PlannedSubPool, ...]:
        """The sub-pools a prefix tree caches, in chain order: all but the
        request sub-pools."""
        return tuple(
            s for s in self.sub_pools if not isinstance(s.lifetime, RequestLifetime)
        )

    def drive_order(self) -> Tuple[PlannedSubPool, ...]:
        """The cached sub-pools from the longest retention to the shortest: the
        owner, then windows from the largest, then state; ties by unit, then
        name. A prefix tree drives eviction and lists its components in this
        order."""
        return tuple(
            sorted(
                self.cached_sub_pools(),
                key=lambda s: (
                    tuple(-r for r in s.lifetime.retention()),
                    s.unit,
                    s.name,
                ),
            )
        )

    def tree_slots(
        self, *, window_lengths: Optional[Dict[str, int]] = None
    ) -> Tuple[TreeSlot, ...]:
        """The prefix tree's slots, one per sub-pool, in drive order with the
        windows in `windows` order: the tree's first window slot is the
        target's window. A window slot keeps ``window_lengths[name]``, the
        tail a running request keeps of it (`KVStorage.retained_window_lengths`);
        without it, the window its layers declare (for a slot's identity
        only)."""
        for sub_pool in self.sub_pools:
            lifetime = sub_pool.lifetime
            if isinstance(lifetime, StateLifetime) and (
                lifetime.checkpoint_interval is None
                or lifetime.checkpoint_interval <= 0
            ):
                raise ValueError(
                    f"state sub-pool {sub_pool.name!r} needs a positive checkpoint "
                    "interval before it can be advertised to a prefix cache"
                )
        return tuple(
            TreeSlot(
                component=s.external.tree_component,
                family=_TREE_FAMILY[type(s.lifetime)],
                window=(
                    (window_lengths or {}).get(s.name, s.lifetime.window)
                    if isinstance(s.lifetime, WindowLifetime)
                    else None
                ),
                checkpoint_interval=(
                    s.lifetime.checkpoint_interval
                    if isinstance(s.lifetime, StateLifetime)
                    else None
                ),
                host_pool=s.external.host_pool,
            )
            for s in self._tree_order()
        )

    def _tree_order(self) -> Tuple[PlannedSubPool, ...]:
        windows = iter(self.windows())
        return tuple(
            next(windows) if isinstance(s.lifetime, WindowLifetime) else s
            for s in self.drive_order()
        )

    def wire_entries(self) -> Tuple[WireEntry, ...]:
        """The PD state entries in peer order: by state type, then key, so
        entries of one type line up on both peers. Owner token KV is not one."""
        entries = [
            WireEntry(
                sub_pool=s.name,
                part=STATE_PART if isinstance(s.lifetime, StateLifetime) else KV_PART,
                state_type=s.external.wire,
                key=s.external.wire_key,
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
                        key=self.sub_pool(declared.sub_pool).external.wire_key,
                    ),
                )
        entries.extend(groups.values())
        unknown = [e.state_type for e in entries if e.state_type not in _WIRE_ORDER]
        if unknown:
            raise ValueError(f"PD transfers have no state type for {unknown}")
        return tuple(
            sorted(
                entries,
                key=lambda e: (_WIRE_ORDER.index(e.state_type), e.key),
            )
        )


def _unit(lifetime: KVLifetime, page_size: int) -> int:
    return 1 if isinstance(lifetime, (StateLifetime, RequestLifetime)) else page_size


def _chain_key(key: Tuple[KVLifetime, int]) -> tuple:
    lifetime, unit = key
    rank, window = lifetime.retention()
    return (rank, -window, unit)


def _sub_pool_name(lifetime: KVLifetime) -> str:
    if isinstance(lifetime, WindowLifetime):
        return window_sub_pool_name(lifetime.window)
    return _SUB_POOL_NAMES[type(lifetime)]


# Per (lifetime, unit) group: each geometry its layers declare, in order of
# first declaration, with the layers declaring it.
_Groups = Dict[Tuple[KVLifetime, int], List[Tuple[PartGeometry, List[int]]]]


def _group_parts(decls: Sequence[KVLayerDecl], page_size: int) -> _Groups:
    groups: _Groups = {}
    for decl in decls:
        for part in decl.parts:
            key = (part.lifetime, _unit(part.lifetime, page_size))
            entries = groups.setdefault(key, [])
            for geometry, layer_ids in entries:
                if geometry == part.geometry:
                    layer_ids.append(decl.layer_id)
                    break
            else:
                entries.append((part.geometry, [decl.layer_id]))
    request = _request_group(groups)
    if request is not None:
        geometry, layer_ids = request
        groups[(RequestLifetime(), 1)] = [(geometry, layer_ids)]
    return groups


def _request_group(groups: _Groups) -> Optional[Tuple[RequestGeometry, List[int]]]:
    """The request sub-pool the state sub-pools' request parts form, each part
    named by its state sub-pool; None when no state declares any."""
    parts, layer_ids = [], set()
    for (lifetime, _), entries in groups.items():
        for geometry, layers in entries:
            if not isinstance(geometry, StateGeometry) or not geometry.request_parts:
                continue
            name = _sub_pool_name(lifetime)
            parts.extend(
                msgspec.structs.replace(part, name=f"{name}.{part.name}")
                for part in geometry.request_parts
            )
            layer_ids.update(layers)
    if not parts:
        return None
    return RequestGeometry(parts=tuple(parts)), sorted(layer_ids)


def sub_pool_names(decls: Sequence[KVLayerDecl]) -> Tuple[str, ...]:
    """The sub-pools a plan of these declarations holds, owner first: what the
    fused-draft placement needs before the plan exists."""
    keys = _group_parts(decls, page_size=1)
    return tuple(_sub_pool_name(lt) for lt, _ in sorted(keys, key=_chain_key))[::-1]


def _needs_unit_layout(geometry: PartGeometry) -> bool:
    """Whether a geometry is laid out only inside a composite unit, even alone."""
    return isinstance(geometry, UnitPartGeometry) and geometry.requires_unit_layout()


def _part_name(geometry: PartGeometry) -> str:
    """The name a composite sub-pool routes a geometry's layers by."""
    if isinstance(geometry, UnitPartGeometry):
        return geometry.part_name
    return KV_PART


def _unit_part(geometry: PartGeometry, *, name: str, layer_num: int) -> UnitPart:
    """A geometry's bytes in a composite unit."""
    if isinstance(geometry, UnitPartGeometry):
        return geometry.unit_part(name=name, layer_num=layer_num)
    # Token-major rows: their backends read one run at the token entry stride.
    if isinstance(geometry, MLAGeometry):
        row_bytes = (
            geometry.kv_lora_rank + geometry.qk_rope_head_dim
        ) * geometry.kv_cache_dtype.itemsize
    else:
        assert isinstance(geometry, MHAGeometry), geometry
        row_bytes = (
            geometry.head_num
            * (geometry.head_dim + geometry.v_head_dim)
            * geometry.kv_cache_dtype.itemsize
        )
    return TokenRows(name=name, layer_num=layer_num, row_bytes=row_bytes)


def _planned_parts(
    entries: Sequence[Tuple[PartGeometry, Sequence[int]]],
) -> Tuple[PlannedPart, ...]:
    for geometry, _ in entries:
        if isinstance(geometry, (StateGeometry, RequestGeometry)):
            raise ValueError(
                f"a {type(geometry).__name__} has no part in a composite unit; "
                "only token layers' KV and the parts they keep beside it share one"
            )
    parts = tuple(
        PlannedPart(
            name=_part_name(geometry), geometry=geometry, layer_ids=tuple(layer_ids)
        )
        for geometry, layer_ids in entries
    )
    names = [p.name for p in parts]
    if len(set(names)) != len(names) or names[0] != KV_PART:
        raise ValueError(
            "layers of one sub-pool declare their KV in more than one geometry "
            f"({[type(p.geometry).__name__ for p in parts]}); a sub-pool holds "
            "one KV layout, beside which its layers may keep other parts"
        )
    kv_layers = set(parts[0].layer_ids)
    for part in parts[1:]:
        stray = set(part.layer_ids) - kv_layers
        if stray:
            raise ValueError(
                f"layers {sorted(stray)} keep a {part.name!r} part without KV in "
                "the same sub-pool"
            )
    return parts


def _composite_spec(
    *,
    name: str,
    parts: Tuple[PlannedPart, ...],
    page_size: int,
    grow_direction: str,
    fused_draft: Optional[FusedDraftPlacement],
) -> CompositeSubPoolSpec:
    if fused_draft is not None and fused_draft.region(name) is not None:
        raise ValueError(
            f"a fused draft cannot ride in composite sub-pool {name!r} yet"
        )
    return CompositeSubPoolSpec(
        name=name,
        layer_num=len(parts[0].layer_ids),
        grow_direction=grow_direction,
        page_size=page_size,
        parts=tuple(
            _unit_part(p.geometry, name=p.name, layer_num=len(p.layer_ids))
            for p in parts
        ),
    )


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
    if isinstance(geometry, RequestGeometry):
        assert fused_draft is None or fused_draft.region(name) is None, (
            "a fused draft rides in no request entry"
        )
        return RequestSubPoolSpec(
            name=name,
            layer_num=layer_num,
            grow_direction=grow_direction,
            parts=geometry.parts,
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
    derived_parts: Sequence[DerivedPartDecl] = (),
) -> KVStoragePlan:
    """The unified-buffer plan for these declarations."""
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    groups = _group_parts(decls, page_size)
    if not any(isinstance(lt, FullLifetime) for lt, _ in groups):
        raise ValueError("unified storage needs a FULL virtual-id owner")
    names_by_key = {key: _sub_pool_name(key[0]) for key in groups}
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
            if name in names_by_key.values():
                raise ValueError(f"draft band {name!r} takes a target sub-pool's name")
            groups[key] = [(geometry, [])]
            names_by_key[key] = name
    chain = sorted(groups, key=lambda key: (*_chain_key(key), names_by_key[key]))

    sub_pools = []
    for i, key in enumerate(chain):
        lifetime, unit = key
        entries = groups[key]
        geometry, layer_ids = entries[0]
        name = names_by_key[key]
        grow_direction = "down" if i == len(chain) - 1 else "up" if i == 0 else "float"
        parts = ()
        if len(entries) > 1 or _needs_unit_layout(geometry):
            parts = _planned_parts(entries)
            spec = _composite_spec(
                name=name,
                parts=parts,
                page_size=unit,
                grow_direction=grow_direction,
                fused_draft=fused_draft,
            )
        else:
            spec = _spec_for(
                name=name,
                geometry=geometry,
                layer_num=len(layer_ids),
                grow_direction=grow_direction,
                fused_draft=fused_draft,
            )
        sub_pools.append(
            PlannedSubPool(
                name=name,
                lifetime=lifetime,
                unit=unit,
                layer_ids=tuple(layer_ids),
                geometry=geometry,
                spec=spec,
                parts=parts,
                external=(
                    _window_external(name)
                    if isinstance(lifetime, WindowLifetime)
                    else _EXTERNAL[type(lifetime)]
                ),
            )
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
    for declared in derived_parts:
        if declared.wire is None:
            continue
        if declared.wire not in _WIRE_ORDER:
            raise ValueError(f"unknown transfer wire name {declared.wire!r}")
        owner = ("page", declared.sub_pool)
        if wire_owners.setdefault(declared.wire, owner) != owner:
            raise ValueError(
                f"wire component {declared.wire!r} mixes allocation domains"
            )
    return KVStoragePlan(
        page_size=page_size,
        sub_pools=tuple(sub_pools),
        fused_draft=fused_draft,
        scratch=tuple(scratch),
        derived_parts=tuple(derived_parts),
    )
