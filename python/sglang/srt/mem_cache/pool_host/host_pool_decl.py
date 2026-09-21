"""Declarations of the host pools a device pool needs mirrored by HiCache.

A device pool declares them (``HostPoolDecl``); the assembler binds each
declaration to transfer layers (``HostPoolPlan``) and builds entries from the
plan, so every declared pool gets an entry by construction.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)


class HostPoolLayout(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer storage bytes of one token-addressed host pool. This is what the
    mirror allocates from, not a description of the kernel-facing layout."""

    bytes_per_token_per_layer: int
    dtype: torch.dtype

    def page_bytes(self, page_size: int) -> int:
        """Bytes of one page of one layer."""
        return self.bytes_per_token_per_layer * page_size

    def host_bytes(self, *, page_num: int, layer_num: int, page_size: int) -> int:
        return page_num * layer_num * self.page_bytes(page_size)


class LayerBinding(msgspec.Struct, frozen=True, kw_only=True):
    """Transfer layer (what the controller iterates) to device pool layer.

    Host compact indices stay inside the mirror; packed draft remapping stays
    inside the controller.
    """

    transfer_to_device: dict[int, int]
    transfer_layer_id_max: int


class MirrorAdapter(Protocol):
    def build(
        self,
        *,
        decl: HostPoolDecl,
        anchor_host: Any,
        allocator_type: str,
        packed_draft_device_pools: tuple[Any, ...],
    ) -> Any: ...


class HostPoolDecl(msgspec.Struct, frozen=True, kw_only=True):
    """One host pool a device pool asks HiCache to mirror. Pool-intrinsic: no layer binding."""

    name: PoolName
    # The pool object that owns this state's device buffers. For a composite
    # pool the KV state lives on a sub-pool while dependent states live on the
    # composite itself, so each declaration names its own owner.
    device_pool: Any
    # Whose host page indices this state reuses when transferred. None: primary.
    index_source: Optional[PoolName]
    # Whose mirror decides this state's capacity and layout. None: self.
    layout_source: Optional[PoolName]
    # None for a layout root (KV, DRAFT): its mirror is built by the assembler
    # from the device pool, so no byte facts are declared for it.
    layout: Optional[HostPoolLayout]
    mirror: Optional[MirrorAdapter]
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    # Local device layers that own buffers for this pool; None means every
    # layer. Layers outside it are neither mirrored nor transferred.
    device_layers: Optional[tuple[int, ...]] = None

    @property
    def is_primary(self) -> bool:
        return self.index_source is None

    @property
    def is_layout_root(self) -> bool:
        return self.layout_source is None

    def sidecar_spec(self) -> SidecarPoolSpec:
        if self.index_source is None:
            raise ValueError(f"{self.name} is the primary pool and has no sidecar spec")
        return SidecarPoolSpec(
            pool_name=self.name,
            indices_from_pool=self.index_source,
            hit_policy=self.hit_policy,
        )


class HostPoolPlan(msgspec.Struct, frozen=True, kw_only=True):
    """A declaration bound to a stack: the unit the assembler builds entries from."""

    decl: HostPoolDecl
    layers: LayerBinding
    # Draft pools whose same-named state is appended as tail layers, in depth order.
    packed_draft_device_pools: tuple[Any, ...] = ()


def kv_pool_decl(pool: Any) -> HostPoolDecl:
    """The primary KV pool every device pool declares; its mirror is built by
    the assembler, so no layout is declared here."""
    return HostPoolDecl(
        name=PoolName.KV,
        device_pool=pool,
        index_source=None,
        layout_source=None,
        layout=None,
        mirror=None,
    )


# Separate (non-packed) drafts mirror each target-role pool under its own name
# while reusing the target KV transfer indices.
_DRAFT_NAMES = {
    PoolName.KV: PoolName.DRAFT,
    PoolName.INDEXER: PoolName.DRAFT_INDEXER,
}


def draft_sidecar_decls(
    draft_decls: tuple[HostPoolDecl, ...],
) -> tuple[HostPoolDecl, ...]:
    """Rename a draft pool's declarations into the target's sidecar namespace:
    KV -> DRAFT (own capacity), INDEXER -> DRAFT_INDEXER laid out on DRAFT,
    every transfer index taken from target KV."""
    out = []
    for d in draft_decls:
        if d.name not in _DRAFT_NAMES:
            raise ValueError(f"no separate-draft sidecar defined for {d.name}")
        out.append(
            msgspec.structs.replace(
                d,
                name=_DRAFT_NAMES[d.name],
                index_source=PoolName.KV,
                layout_source=(
                    None if d.layout_source is None else _DRAFT_NAMES[d.layout_source]
                ),
            )
        )
    return tuple(out)


def packed_draft_pools(
    target_decls: tuple[HostPoolDecl, ...], draft_pools: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Validate that every draft can be appended as tail layers of the target
    mirrors: it declares the same pools with the same per-layer layout and
    every packed layer owns its buffer. The draft plan already chose packing,
    so a draft that cannot be packed is an error, not a silent skip."""
    targets = {d.name: d for d in target_decls}
    for pool in draft_pools:
        drafts = {d.name: d for d in pool.host_pool_decls()}
        if set(drafts) != set(targets):
            raise ValueError(
                f"packed draft {type(pool).__name__} declares "
                f"{sorted(d.value for d in drafts)} but the target declares "
                f"{sorted(d.value for d in targets)}; every target pool needs a "
                "draft counterpart or the draft state is not restored"
            )
        for name, target in targets.items():
            draft = drafts[name]
            if draft.layout != target.layout:
                raise ValueError(
                    f"packed draft {name.value} layout {draft.layout} differs from "
                    f"target {target.layout}"
                )
            owned = draft.device_layers
            if owned is not None and len(owned) != draft.device_pool.layer_num:
                raise ValueError(
                    f"packed draft {name.value} owns buffers on {len(owned)} of "
                    f"{draft.device_pool.layer_num} layers; every packed layer "
                    "must own its buffer"
                )
    return tuple(draft_pools)


def layout_root(decls: tuple[HostPoolDecl, ...]) -> HostPoolDecl:
    """The one declaration whose mirror decides the group's capacity."""
    roots = [d for d in decls if d.is_layout_root]
    if len(roots) != 1:
        raise ValueError(
            f"expected exactly one layout root, got {[d.name for d in roots]}"
        )
    return roots[0]


def _decl_named(decls: tuple[HostPoolDecl, ...], name: PoolName) -> HostPoolDecl:
    return next(d for d in decls if d.name == name)


def plan_host_pools(
    *,
    decls: tuple[HostPoolDecl, ...],
    full_layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    packed_draft_pools: tuple[Any, ...] = (),
    index_primary: Optional[PoolName] = None,
) -> tuple[HostPoolPlan, ...]:
    """Bind declarations to a stack. A target group contains its own primary
    KV pool; a draft sidecar group reuses an external ``index_primary``. Either
    way exactly one layout root anchors the others' capacity, and sidecar
    indices come from one real source (HostPoolGroup resolves no chains).

    ``packed_draft_pools`` are drafts accepted by packed_draft_pools; each
    plan carries the draft objects that own its same-named state."""
    names = [d.name for d in decls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate host pool names: {names}")
    root = layout_root(decls)
    if index_primary is None:
        if not root.is_primary or root.name != PoolName.KV:
            raise ValueError(
                f"expected the layout root to be the primary KV pool, got {root.name}"
            )
        index_primary = root.name
    elif any(d.is_primary for d in decls):
        raise ValueError("a sidecar group must take every index from the target")
    for d in decls:
        if d.is_primary:
            continue
        if d.index_source != index_primary:
            raise ValueError(
                f"{d.name}.index_source must be {index_primary}, got {d.index_source}"
            )
        if d.is_layout_root:
            continue
        if d.layout_source == d.name:
            raise ValueError(f"{d.name}.layout_source must name another pool")
        if d.layout_source not in names:
            raise ValueError(f"{d.name} references undeclared pool {d.layout_source}")
    draft_decls = [pool.host_pool_decls() for pool in packed_draft_pools]
    return tuple(
        HostPoolPlan(
            decl=d,
            layers=LayerBinding(
                transfer_to_device=_owned_layer_mapping(
                    full_layer_mapping, d.device_layers, root.device_pool.layer_num
                ),
                transfer_layer_id_max=transfer_layer_id_max,
            ),
            packed_draft_device_pools=tuple(
                _decl_named(decls_of_draft, d.name).device_pool
                for decls_of_draft in draft_decls
            ),
        )
        for d in decls
    )


def _owned_layer_mapping(
    mapping: dict[int, int],
    device_layers: Optional[tuple[int, ...]],
    target_layer_num: int,
) -> dict[int, int]:
    """Drop transfer layers whose device layer owns no buffer for this pool.
    Packed draft tails (device index >= target layer count) are kept."""
    if device_layers is None:
        return mapping
    owned = set(device_layers)
    return {
        t: dev for t, dev in mapping.items() if dev >= target_layer_num or dev in owned
    }
