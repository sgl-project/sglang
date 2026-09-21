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
        device_pool: Any,
        anchor_host: Any,
        allocator_type: str,
    ) -> Any: ...


class HostPoolDecl(msgspec.Struct, frozen=True, kw_only=True):
    """One host pool a device pool asks HiCache to mirror. Pool-intrinsic: no layer binding."""

    name: PoolName
    # Whose host page indices this state reuses when transferred. None: primary.
    index_source: Optional[PoolName]
    # Whose mirror decides this state's capacity and layout. None: self.
    layout_source: Optional[PoolName]
    # None for a layout root (KV, DRAFT): its mirror is built by the assembler
    # from the device pool, so no byte facts are declared for it.
    layout: Optional[HostPoolLayout]
    mirror: Optional[MirrorAdapter]
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES

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
    device_pool: Any
    layers: LayerBinding
    packed_draft_device_pools: tuple[Any, ...] = ()


def kv_pool_decl() -> HostPoolDecl:
    """The primary KV pool every device pool declares; its mirror is built by
    the assembler, so no layout is declared here."""
    return HostPoolDecl(
        name=PoolName.KV,
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


def packable_draft_pools(
    target_decls: tuple[HostPoolDecl, ...], draft_pools: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Draft pools whose declarations cover every target pool with the same
    per-layer layout, so their layers can be appended to the target mirrors.
    A draft that declares fewer pools is skipped; a layout mismatch is an error."""
    targets = {d.name: d for d in target_decls}
    packable = []
    for pool in draft_pools:
        drafts = {d.name: d for d in pool.host_pool_decls()}
        if set(drafts) != set(targets):
            continue
        for name, target in targets.items():
            if drafts[name].layout != target.layout:
                raise ValueError(
                    f"packed draft {name} layout {drafts[name].layout} differs from "
                    f"target {target.layout}"
                )
        packable.append(pool)
    return tuple(packable)


def plan_host_pools(
    *,
    decls: tuple[HostPoolDecl, ...],
    device_pool: Any,
    full_layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    packed_draft_device_pools: tuple[Any, ...] = (),
    index_primary: Optional[PoolName] = None,
) -> tuple[HostPoolPlan, ...]:
    """Bind declarations to a stack. A target group contains its own primary
    KV pool; a draft sidecar group reuses an external ``index_primary``. Either
    way exactly one layout root anchors the others' capacity, and sidecar
    indices come from one real source (HostPoolGroup resolves no chains)."""
    names = [d.name for d in decls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate host pool names: {names}")
    roots = [d for d in decls if d.is_layout_root]
    if len(roots) != 1:
        raise ValueError(
            f"expected exactly one layout root, got {[d.name for d in roots]}"
        )
    root = roots[0]
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
    layers = LayerBinding(
        transfer_to_device=full_layer_mapping,
        transfer_layer_id_max=transfer_layer_id_max,
    )
    return tuple(
        HostPoolPlan(
            decl=d,
            device_pool=device_pool,
            layers=layers,
            packed_draft_device_pools=packed_draft_device_pools,
        )
        for d in decls
    )
