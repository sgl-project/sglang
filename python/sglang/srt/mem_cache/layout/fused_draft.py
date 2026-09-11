"""Draft-model regions nested inside unified sub-pools' slot entries.

The nesting, outermost first:

  sub-pool  a named pool inside the unified pool ("full", "swa", "mamba"),
            with its own grow direction and frontier. HOST is the role it
            plays when its entries also carry draft bytes.
  slot      one token entry of that sub-pool. The draft's KV is extra parts of
            every slot, after the host's own parts, so ONE slot id per token
            covers target and draft: one allocation, one free, one whole-page
            relocation carry both.
  region    the draft geometry inside one host's entries. A region is not a
            `SubPoolSpec`: no grow direction, no frontier logic, only geometry.
  lane      one position in the region's layer dimension -- one draft layer of
            one runner, present in every slot.
  range     the contiguous lanes one runner owns in one host.

A lane is not a layer. Runners are the draft's execution copies: a
replicated head gives EVERY runner all the draft's layers, on lanes of its own
so runners do not clobber each other's KV (3 runners x 2 layers = 6 lanes),
while a per-depth head (one block per predicted position) gives each runner
ONE depth (8 layers across 2 runners = 2 lanes). The runners' ranges tile
each region in runner order, with no gap and no overlap.

A draft LAYER kind (`LAYER_*`) is what the draft checkpoint has; a HOST kind
(`HostKind`, keyed by sub-pool name) is what a sub-pool can carry in its
entries. `place_fused_draft` consults the `HOST_KINDS` registry, and the draft
worker binds each host through its family's binder
(`unified_draft_pool.DRAFT_BINDERS`). When a kind's host turns it away, its
layers FOLD into that host's `fallback_host` if the one-region-one-geometry
rule allows.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.layout.token_major import ROW_ALIGN_BYTES, DensePart

LAYER_FULL = "full"
LAYER_WINDOW = "window"
LAYER_STATE = "state"

_LAYER_KIND_NOUN = {
    LAYER_FULL: "full-attention",
    LAYER_WINDOW: "sliding-window",
    LAYER_STATE: "recurrent-state",
}


class DenseDraftRegion(msgspec.Struct, frozen=True, kw_only=True):
    """Geometry of the DRAFT model's K/V rows fused into every slot of a host
    sub-pool. The draft is a separate checkpoint, so its head geometry and
    layer count can differ from the host's; its K and V rows are two more parts
    of the host entry, indexed by the host's physical token id."""

    lane_num: int
    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None
    # The KV cache dtype the rows hold, where it differs from how they are
    # stored: fp8 rows are stored as uint8.
    kv_dtype: Optional[torch.dtype] = None

    def validate(self) -> None:
        assert self.lane_num > 0, f"lane_num must be positive; got {self.lane_num}"
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        v = self.resolved_v_head_dim()
        assert v > 0, f"v_head_dim must be positive; got {v}"

    def resolved_v_head_dim(self) -> int:
        return self.head_dim if self.v_head_dim is None else self.v_head_dim

    def resolved_kv_dtype(self) -> torch.dtype:
        return self.store_dtype if self.kv_dtype is None else self.kv_dtype

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.resolved_v_head_dim() * self.store_dtype.itemsize

    def entry_bytes(self) -> int:
        """Draft bytes per slot, before the host entry's alignment."""
        return self.lane_num * (self.k_row_bytes() + self.v_row_bytes())

    def parts(self, offset_bytes: int) -> Tuple[DensePart, DensePart]:
        """The draft's K and V parts, laid out from ``offset_bytes`` inside
        the host entry."""
        layer_stride = self.k_row_bytes() + self.v_row_bytes()
        return (
            DensePart(
                name="draft_k",
                offset_bytes=offset_bytes,
                layer_stride_bytes=layer_stride,
                layer_num=self.lane_num,
                row_shape=(self.head_num, self.head_dim),
                dtype=self.store_dtype,
            ),
            DensePart(
                name="draft_v",
                offset_bytes=offset_bytes + self.k_row_bytes(),
                layer_stride_bytes=layer_stride,
                layer_num=self.lane_num,
                row_shape=(self.head_num, self.resolved_v_head_dim()),
                dtype=self.store_dtype,
            ),
        )


# Widened to a Union once a host kind places a region of another shape.
DraftRegion = DenseDraftRegion


class DraftKVGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """Per-GPU K/V row geometry of one kind of draft attention layer."""

    head_num: int
    head_dim: int
    v_head_dim: int


class DraftStateGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """Per-GPU shapes of one recurrent-state layer of the draft: one conv
    tensor per stream plus the temporal state, as `MambaPool.State` lays them
    out. Every stream is carried, even ones a depth never touches, because
    the backend indexes ``conv[stream]`` by absolute stream number."""

    conv_state_shapes: Tuple[Tuple[int, ...], ...]
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype

    def conv_row_bytes(self, idx: int) -> int:
        return math.prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return math.prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def layer_bytes(self) -> int:
        conv = sum(self.conv_row_bytes(i) for i in range(len(self.conv_state_shapes)))
        return conv + self.temporal_row_bytes()


class DraftLayerSet(msgspec.Struct, frozen=True, kw_only=True):
    """The draft's layers of one kind: layer ids for a replicated head, depth
    ids for a per-depth head, with the per-GPU geometry they share."""

    layer_ids: Tuple[int, ...]
    geometry: Any
    # LAYER_WINDOW only: how many tokens back the layers read.
    window: Optional[int] = None


class DraftKVProfile(msgspec.Struct, frozen=True, kw_only=True):
    """What the draft checkpoint asks of the host sub-pools, by layer kind.

    ``num_depths`` > 1 marks a per-depth head (one transformer block per MTP
    depth, served by one runner each under multi-layer EAGLE), whose layer
    ids are depth ids; otherwise every runner serves all ``num_layers``
    layers. A kind is present only when the draft has layers of it.
    """

    num_layers: int
    num_depths: int = 1
    kinds: Dict[str, DraftLayerSet] = {}


class PlacementContext(msgspec.Struct, frozen=True, kw_only=True):
    """Target-side facts a host kind consults when placing a layer set:
    the sub-pools the host builds, the target's window, whether every
    resolved backend carries v_head_dim, and the backends the draft worker
    may run on (its explicit one, else the target's)."""

    host_names: Tuple[str, ...]
    target_window: Optional[int] = None
    asymmetric_rows_ok: bool = False
    draft_backends: Tuple[str, ...] = ()


class HostKind(msgspec.Struct, frozen=True, kw_only=True):
    """A unified sub-pool that carries one kind of draft layer in its entries.

    ``serves`` is the layer kind this host is the first choice for and
    ``family`` picks the draft worker's binder. ``fallback_host`` carries
    ``serves`` when this host cannot, under the one-region-one-geometry fold
    rule.
    """

    name: str
    serves: str
    family: str
    fallback_host: Optional[str] = None

    def prerequisite(
        self, *, layers: DraftLayerSet, ctx: PlacementContext
    ) -> Optional[str]:
        """What the runtime must carry for ``layers`` to be served fused at
        all, whichever host takes them; a reason here declines the draft."""
        return None

    def admits(self, *, layers: DraftLayerSet, ctx: PlacementContext) -> Optional[str]:
        """None when this host's entries can carry ``layers``, else why not."""
        if self.name not in ctx.host_names:
            return f"the host has no {self.name!r} sub-pool"
        return None

    def region_for(
        self,
        *,
        layers: DraftLayerSet,
        lane_num: int,
        store_dtype: torch.dtype,
        kv_dtype: Optional[torch.dtype],
    ) -> DraftRegion:
        raise NotImplementedError

    def describe(self, *, region: DraftRegion, lanes: Sequence[Tuple[int, ...]]) -> str:
        """The boot-log line for ``region`` and each runner's lanes in it."""
        raise NotImplementedError


class DenseHostKind(HostKind):
    """A host whose entries take the draft's K/V rows as two more dense parts."""

    def region_for(
        self,
        *,
        layers: DraftLayerSet,
        lane_num: int,
        store_dtype: torch.dtype,
        kv_dtype: Optional[torch.dtype],
    ) -> DenseDraftRegion:
        geometry = layers.geometry
        assert isinstance(geometry, DraftKVGeometry), geometry
        return DenseDraftRegion(
            lane_num=lane_num,
            head_num=geometry.head_num,
            head_dim=geometry.head_dim,
            v_head_dim=geometry.v_head_dim,
            store_dtype=store_dtype,
            kv_dtype=kv_dtype,
        )

    def describe(self, *, region: DraftRegion, lanes: Sequence[Tuple[int, ...]]) -> str:
        kv_dtype = region.resolved_kv_dtype()
        stored = (
            kv_dtype
            if kv_dtype == region.store_dtype
            else f"{kv_dtype} (stored as {region.store_dtype})"
        )
        return (
            f"fused draft region in {self.name!r}: {region.lane_num} lane(s) x "
            f"{region.head_num} kv head(s) x {region.head_dim}/"
            f"{region.resolved_v_head_dim()} k/v head_dim @ {stored} "
            f"= {region.entry_bytes()} B/token; runner lanes {list(lanes)}"
        )


# Registration order is the order placements report their hosts in.
HOST_KINDS: Dict[str, HostKind] = {}


def host_for(layer_kind: str) -> Optional[HostKind]:
    """The registered host that serves ``layer_kind`` first, if any."""
    for kind in HOST_KINDS.values():
        if kind.serves == layer_kind:
            return kind
    return None


def register_host_kind(kind: HostKind) -> HostKind:
    assert kind.name not in HOST_KINDS, f"host kind {kind.name!r} is already registered"
    primary = host_for(kind.serves)
    assert primary is None, (
        f"layer kind {kind.serves!r} is already served by host {primary.name!r}"
    )
    assert kind.fallback_host is None or kind.fallback_host in HOST_KINDS, (
        f"host kind {kind.name!r} falls back to unregistered host "
        f"{kind.fallback_host!r}"
    )
    HOST_KINDS[kind.name] = kind
    return kind


FULL_HOST = register_host_kind(
    DenseHostKind(name="full", serves=LAYER_FULL, family="dense")
)

# Multi-step draft backends that build the per-step sliding-window read and
# write rails a fused window layer needs (TritonMultiStepDraftBackend).
WINDOW_RAIL_BACKENDS = frozenset({"triton"})


class WindowHostKind(DenseHostKind):
    """The swa sub-pool: a draft window layer rides there when its window fits
    the target's; eviction keeps any further reach the draft worker declares."""

    def prerequisite(
        self, *, layers: DraftLayerSet, ctx: PlacementContext
    ) -> Optional[str]:
        unrailed = sorted(set(ctx.draft_backends) - WINDOW_RAIL_BACKENDS)
        if unrailed:
            return (
                f"the draft's attention backend {unrailed} carries no per-step "
                f"sliding-window rail (only {sorted(WINDOW_RAIL_BACKENDS)} does)"
            )
        return None

    def admits(self, *, layers: DraftLayerSet, ctx: PlacementContext) -> Optional[str]:
        reason = super().admits(layers=layers, ctx=ctx)
        if reason is not None:
            return reason
        if layers.window is None:
            return "the draft declares no sliding window size"
        if ctx.target_window is None:
            return "the target declares no sliding window size"
        if layers.window > ctx.target_window:
            return (
                f"its window {layers.window} exceeds the target's window "
                f"{ctx.target_window}"
            )
        return None


SWA_HOST = register_host_kind(
    WindowHostKind(
        name="swa", serves=LAYER_WINDOW, family="dense", fallback_host="full"
    )
)


class RunnerLanes(msgspec.Struct, frozen=True, kw_only=True):
    """One draft runner's lanes inside each host region: host -> (start, count)."""

    ranges: Dict[str, Tuple[int, int]] = {}

    def range_for(self, host: str) -> range:
        start, count = self.ranges.get(host, (0, 0))
        return range(start, start + count)


class FusedDraftPlacement(msgspec.Struct, frozen=True, kw_only=True):
    """Where every draft runner's layers live inside the host sub-pools.

    One region per host sub-pool that carries draft lanes plus each runner's
    lane range into it. Built once on the target, stored on the
    `UnifiedKVPool`, and read back by each draft runner, so the two sides
    cannot disagree on a lane.
    """

    runners: Tuple[RunnerLanes, ...]
    regions: Dict[str, DraftRegion] = {}

    def __post_init__(self):
        assert len(self.runners) > 0, "a placement needs at least one draft runner"
        hosts = set(self.regions).union(*(r.ranges for r in self.runners))
        for host in sorted(hosts):
            assert host in HOST_KINDS, (
                f"no host kind is registered for sub-pool {host!r}"
            )
            region = self.regions.get(host)
            lanes = [s for r in self.runners for s in r.range_for(host)]
            if region is None:
                assert not lanes, f"host {host!r} has draft lanes but no region"
                continue
            assert lanes == list(range(region.lane_num)), (
                f"host {host!r}: runner lane ranges {lanes} must tile "
                f"range({region.lane_num}) in runner order"
            )

    def region(self, host: str) -> Optional[DraftRegion]:
        return self.regions.get(host)

    def hosts(self) -> Tuple[str, ...]:
        """The hosts carrying draft lanes, in registry order."""
        return tuple(h for h in HOST_KINDS if h in self.regions)

    def lanes_for(self, runner: int, host: str) -> range:
        return self.runners[runner].range_for(host)

    @classmethod
    def from_counts(
        cls,
        *,
        counts: Mapping[str, Sequence[int]],
        regions: Mapping[str, DraftRegion],
    ) -> "FusedDraftPlacement":
        """Tile each host's region with the runners' layer counts, in runner order."""
        num_runners = {len(per_runner) for per_runner in counts.values()}
        assert len(num_runners) == 1, (
            f"every host needs one layer count per runner; got {dict(counts)}"
        )
        ranges: List[Dict[str, Tuple[int, int]]] = [
            {} for _ in range(num_runners.pop())
        ]
        for host, per_runner in counts.items():
            assert host in regions, f"host {host!r} has layer counts but no region"
            start = 0
            for runner, count in enumerate(per_runner):
                if count:
                    ranges[runner][host] = (start, count)
                start += count
        return cls(
            runners=tuple(RunnerLanes(ranges=r) for r in ranges),
            regions=dict(regions),
        )


def draft_swa_layer_ids(draft_model_config) -> Tuple[int, ...]:
    """The draft layer ids a hybrid-SWA pool routes to its swa side. This is
    the pool-routing convention (`SWAKVPool.layers_mapping`), not a per-layer
    kernel window: an attention-only draft window stays full-kind."""
    mc = draft_model_config
    if mc.is_hybrid_swa and not mc.is_deepseek_v4_arch:
        return tuple(int(i) for i in mc.swa_attention_layer_ids)
    return ()


def draft_state_layer_ids(draft_model_config, *, num_depths: int) -> Tuple[int, ...]:
    """The draft depth ids that own a recurrent-state block of their own.

    Only a per-depth conv-chain MTP head carries state: the baseline serves
    it with a state cache cloned beside the target's. A plain NEXTN head of
    a linear-attention trunk inherits the trunk's mamba-ish config class but
    is a full-attention block; its config lists the TRUNK's state layers,
    and the head carries none.
    """
    from sglang.srt.configs.hybrid_arch import mambaish_config

    mc = draft_model_config
    if mambaish_config(mc) is None or mc.num_nextn_predict_layers is None:
        return ()
    if getattr(mc.hf_text_config, "mtp_local_layer_ids", None) is None:
        return ()
    return tuple(range(num_depths))


def draft_kv_profile(
    draft_model_config,
    *,
    num_layers: int,
    attn_tp_size: int,
    num_depths: Optional[int] = None,
) -> DraftKVProfile:
    """The profile of a draft `ModelConfig`, heads divided by attn_tp the way
    the target divides its own (drafts never join the DCP group).
    ``num_depths`` overrides the config's `num_nextn_predict_layers`."""
    from sglang.srt.configs.hybrid_arch import mambaish_config

    mc = draft_model_config
    if num_depths is None:
        num_depths = mc.num_nextn_predict_layers
    num_depths = 1 if num_depths is None else int(num_depths)
    num_layers = int(num_layers)
    window_ids = draft_swa_layer_ids(mc)
    assert set(window_ids) <= set(range(num_layers)), (
        f"draft window layer ids {window_ids} outside range({num_layers})"
    )
    kinds: Dict[str, DraftLayerSet] = {}
    full_ids = tuple(i for i in range(num_layers) if i not in window_ids)
    if full_ids:
        kinds[LAYER_FULL] = DraftLayerSet(
            layer_ids=full_ids,
            geometry=DraftKVGeometry(
                head_num=int(mc.get_num_kv_heads(attn_tp_size)),
                head_dim=int(mc.head_dim),
                v_head_dim=int(mc.v_head_dim),
            ),
        )
    if window_ids:
        kinds[LAYER_WINDOW] = DraftLayerSet(
            layer_ids=window_ids,
            geometry=DraftKVGeometry(
                head_num=int(mc.get_swa_num_kv_heads(attn_tp_size)),
                head_dim=int(mc.swa_head_dim),
                v_head_dim=int(mc.swa_v_head_dim),
            ),
            window=(
                None if mc.sliding_window_size is None else int(mc.sliding_window_size)
            ),
        )
    state_ids = draft_state_layer_ids(mc, num_depths=num_depths)
    if state_ids:
        cp = mambaish_config(mc).mamba2_cache_params
        kinds[LAYER_STATE] = DraftLayerSet(
            layer_ids=state_ids,
            geometry=DraftStateGeometry(
                conv_state_shapes=tuple(
                    tuple(int(x) for x in s) for s in cp.shape.conv
                ),
                conv_dtype=cp.dtype.conv,
                temporal_state_shape=tuple(int(x) for x in cp.shape.temporal),
                temporal_dtype=cp.dtype.temporal,
            ),
        )
    return DraftKVProfile(num_layers=num_layers, num_depths=num_depths, kinds=kinds)


class FusedDraftDecision(msgspec.Struct, frozen=True, kw_only=True):
    """`place_fused_draft`'s answer: the placement, or why the draft cannot
    fuse. Neither means fusion does not apply. ``note`` says why a placed
    layer kind did not get its first-choice host."""

    placement: Optional[FusedDraftPlacement] = None
    declined: Optional[str] = None
    note: Optional[str] = None


def _runner_layer_counts(
    profile: DraftKVProfile, num_runners: int
) -> Tuple[Optional[Dict[str, List[int]]], Optional[str]]:
    """Per layer kind, each runner's layer count; or why no runner layout exists."""
    if profile.num_depths <= 1:
        return {
            kind: [len(layers.layer_ids)] * num_runners
            for kind, layers in profile.kinds.items()
        }, None
    if num_runners == 1:
        return None, (
            f"a per-depth draft head ({profile.num_depths} depths) needs one "
            "runner per depth (multi-layer EAGLE)"
        )
    if num_runners > profile.num_depths:
        return None, (
            f"{num_runners} draft runners exceed the head's {profile.num_depths} depths"
        )
    return {
        kind: [1 if r in layers.layer_ids else 0 for r in range(num_runners)]
        for kind, layers in profile.kinds.items()
    }, None


def _kv_rows_declined(
    geometry: Any, *, store_dtype: torch.dtype, asymmetric_rows_ok: bool
) -> Optional[str]:
    """Why one kind's K/V rows cannot be fused entry parts, or None."""
    if not isinstance(geometry, DraftKVGeometry):
        return None
    if geometry.head_dim != geometry.v_head_dim and not asymmetric_rows_ok:
        return (
            "the draft's K/V rows are asymmetric "
            f"(head_dim={geometry.head_dim}, v_head_dim={geometry.v_head_dim}) "
            "and a resolved attention backend does not carry v_head_dim "
            "through to the kernel"
        )
    for dim in (geometry.head_dim, geometry.v_head_dim):
        row_bytes = geometry.head_num * dim * store_dtype.itemsize
        if row_bytes % ROW_ALIGN_BYTES:
            return (
                f"the draft's K/V rows are {row_bytes} B, not a multiple of "
                f"the {ROW_ALIGN_BYTES}-B row alignment a fused entry part needs"
            )
    return None


def _fold_refused(
    profile: DraftKVProfile, *, kind: str, into: HostKind
) -> Optional[str]:
    """One region holds one row geometry: ``kind`` folds into ``into`` only
    when the draft has no layers of the kind ``into`` serves, or both kinds
    share a geometry."""
    own = profile.kinds.get(into.serves)
    if own is None or own.geometry == profile.kinds[kind].geometry:
        return None
    return (
        f"its {_LAYER_KIND_NOUN[kind]} layers' rows differ from its "
        f"{_LAYER_KIND_NOUN[into.serves]} layers' rows, so they cannot share the "
        f"{into.name!r} sub-pool"
    )


def place_fused_draft(
    *,
    profile: DraftKVProfile,
    num_runners: int,
    ctx: PlacementContext,
    store_dtype: torch.dtype,
    kv_dtype: Optional[torch.dtype] = None,
) -> FusedDraftDecision:
    """Assign every draft layer of every runner to a host sub-pool through the
    registry: a layer kind goes to the host that serves it, else to that
    host's fallback under the fold rule. A kind no host takes declines the
    whole draft, as does a kind whose runtime prerequisite is missing,
    asymmetric K/V rows unless ``ctx`` vouches that every attention backend
    carries v_head_dim through to the kernel, and rows off the entry-part
    alignment."""
    counts, reason = _runner_layer_counts(profile, num_runners)
    if counts is None:
        return FusedDraftDecision(declined=reason)
    for kind, layers in profile.kinds.items():
        what = f"the draft's {len(layers.layer_ids)} {_LAYER_KIND_NOUN[kind]} layer(s)"
        host = host_for(kind)
        if host is None:
            return FusedDraftDecision(declined=f"{what}: no host kind serves them")
        reason = host.prerequisite(layers=layers, ctx=ctx)
        if reason is not None:
            return FusedDraftDecision(declined=f"{what}: {reason}")
        reason = _kv_rows_declined(
            layers.geometry,
            store_dtype=store_dtype,
            asymmetric_rows_ok=ctx.asymmetric_rows_ok,
        )
        if reason is not None:
            return FusedDraftDecision(declined=reason)

    host_counts: Dict[str, List[int]] = {}
    host_layers: Dict[str, DraftLayerSet] = {}
    notes: List[str] = []
    for kind, layers in profile.kinds.items():
        host = host_for(kind)
        what = f"the draft's {len(layers.layer_ids)} {_LAYER_KIND_NOUN[kind]} layer(s)"
        reason = host.admits(layers=layers, ctx=ctx)
        if reason is not None:
            fallback = (
                None if host.fallback_host is None else HOST_KINDS[host.fallback_host]
            )
            fold = None
            if fallback is not None:
                fold = _fold_refused(profile, kind=kind, into=fallback)
                if fold is None:
                    fold = fallback.admits(layers=layers, ctx=ctx)
            if fallback is None or fold is not None:
                if fold is not None:
                    reason = f"{reason}, and {fold}"
                return FusedDraftDecision(
                    declined=(
                        f"{what} cannot ride in the {host.name!r} sub-pool: {reason}"
                    )
                )
            notes.append(f"{what} ride in the {fallback.name!r} sub-pool: {reason}")
            host = fallback
        per_runner = host_counts.setdefault(host.name, [0] * num_runners)
        for runner, count in enumerate(counts[kind]):
            per_runner[runner] += count
        # The fold rule made every kind sharing a host share its geometry.
        host_layers.setdefault(host.name, layers)

    placed = {host: c for host, c in host_counts.items() if sum(c) > 0}
    regions = {
        host: HOST_KINDS[host].region_for(
            layers=host_layers[host],
            lane_num=sum(c),
            store_dtype=store_dtype,
            kv_dtype=kv_dtype,
        )
        for host, c in placed.items()
    }
    return FusedDraftDecision(
        placement=FusedDraftPlacement.from_counts(counts=placed, regions=regions),
        note="; ".join(notes) or None,
    )
