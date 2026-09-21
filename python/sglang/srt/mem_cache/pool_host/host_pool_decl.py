"""Declarations of the host pools a device pool needs HiCache to keep.

A device pool declares them (``HostPoolDecl``); the assembler binds each
declaration to transfer layers (``HostPoolBuildConfig``) and builds one entry
per config, so every declared pool gets an entry by construction.
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


class HostPoolStorageInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Bytes one host pool stores per token per layer, and their dtype.

    Two equal values mean equal byte counts and dtype only; page geometry,
    scale placement and kernel layout are checked by the host pool builder.
    """

    bytes_per_token_per_layer: int
    dtype: torch.dtype

    def page_bytes(self, page_size: int) -> int:
        """Bytes of one page of one layer."""
        return self.bytes_per_token_per_layer * page_size

    def host_bytes(self, *, page_num: int, layer_num: int, page_size: int) -> int:
        return page_num * layer_num * self.page_bytes(page_size)


class LayerBinding(msgspec.Struct, frozen=True, kw_only=True):
    """Transfer layer (what the controller iterates) to device pool layer.

    Compact host layer indices stay inside the host pool; packed draft
    remapping stays inside the controller.
    """

    transfer_to_device: dict[int, int]
    transfer_layer_id_max: int


class HostPoolBuilder(Protocol):
    def build(
        self,
        *,
        decl: HostPoolDecl,
        anchor_host: Any,
        allocator_type: str,
        packed_draft_device_pools: tuple[Any, ...],
    ) -> Any: ...


class HostPoolDecl(msgspec.Struct, frozen=True, kw_only=True):
    """One host pool a device pool asks HiCache to keep. Pool-intrinsic: no layer binding."""

    pool_name: PoolName
    # The pool object that owns the device buffers behind this host pool. A
    # composite pool keeps its KV buffers on a sub-pool and, e.g., its
    # compressed index keys on itself, so each declaration names its owner.
    device_pool: Any
    # Whose host page indices this pool reuses when transferred. None: primary.
    indices_from_pool: Optional[PoolName]
    # Whose host pool decides this pool's capacity and memory layout. None: self.
    layout_source: Optional[PoolName]
    # None for a layout root (KV, DRAFT): the assembler builds its host pool
    # from the device pool, so no byte facts are declared for it.
    storage_info: Optional[HostPoolStorageInfo]
    host_pool_builder: Optional[HostPoolBuilder]
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    # Local device layers that own buffers for this pool; None means every
    # layer. Layers outside it get no host layer and no transfer.
    owned_device_layers: Optional[tuple[int, ...]] = None

    @property
    def is_primary(self) -> bool:
        return self.indices_from_pool is None

    @property
    def is_layout_root(self) -> bool:
        return self.layout_source is None

    def sidecar_spec(self) -> SidecarPoolSpec:
        if self.indices_from_pool is None:
            raise ValueError(
                f"{self.pool_name} is the primary pool and has no sidecar spec"
            )
        return SidecarPoolSpec(
            pool_name=self.pool_name,
            indices_from_pool=self.indices_from_pool,
            hit_policy=self.hit_policy,
        )


class HostPoolBuildConfig(msgspec.Struct, frozen=True, kw_only=True):
    """A declaration bound to one stack: what the assembler builds an entry from."""

    decl: HostPoolDecl
    layer_binding: LayerBinding
    # Draft pools whose same-named buffers are appended as tail layers, in depth order.
    packed_draft_device_pools: tuple[Any, ...] = ()


def kv_pool_decl(pool: Any) -> HostPoolDecl:
    """The primary KV pool every device pool declares; the assembler builds its
    host pool, so no storage info is declared here."""
    return HostPoolDecl(
        pool_name=PoolName.KV,
        device_pool=pool,
        indices_from_pool=None,
        layout_source=None,
        storage_info=None,
        host_pool_builder=None,
    )


# Separate (non-packed) drafts keep each target-role pool under its own name
# while reusing the target KV transfer indices.
_DRAFT_NAMES = {
    PoolName.KV: PoolName.DRAFT,
    PoolName.INDEXER: PoolName.DRAFT_INDEXER,
}


def make_draft_sidecar_decls(
    draft_decls: tuple[HostPoolDecl, ...],
) -> tuple[HostPoolDecl, ...]:
    """Rename a draft pool's declarations into the target's sidecar namespace:
    KV -> DRAFT (own capacity), INDEXER -> DRAFT_INDEXER laid out on DRAFT,
    every transfer index taken from target KV."""
    out = []
    for d in draft_decls:
        if d.pool_name not in _DRAFT_NAMES:
            raise ValueError(f"no separate-draft sidecar defined for {d.pool_name}")
        out.append(
            msgspec.structs.replace(
                d,
                pool_name=_DRAFT_NAMES[d.pool_name],
                indices_from_pool=PoolName.KV,
                layout_source=(
                    None if d.layout_source is None else _DRAFT_NAMES[d.layout_source]
                ),
            )
        )
    return tuple(out)


def validate_packed_draft_pools(
    target_decls: tuple[HostPoolDecl, ...], draft_pools: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Check that every draft can be appended as tail layers of the target host
    pools: it declares the same pools with the same storage info and every
    packed layer owns its buffer. The draft plan already chose packing, so a
    draft that cannot be packed is an error, not a silent skip."""
    targets = {d.pool_name: d for d in target_decls}
    for pool in draft_pools:
        drafts = {d.pool_name: d for d in pool.host_pool_decls()}
        if set(drafts) != set(targets):
            raise ValueError(
                f"packed draft {type(pool).__name__} declares "
                f"{sorted(d.value for d in drafts)} but the target declares "
                f"{sorted(d.value for d in targets)}; every target pool needs a "
                "draft counterpart or the draft buffers are not restored"
            )
        for name, target in targets.items():
            draft = drafts[name]
            if draft.storage_info != target.storage_info:
                raise ValueError(
                    f"packed draft {name.value} storage {draft.storage_info} "
                    f"differs from target {target.storage_info}"
                )
            owned = draft.owned_device_layers
            if owned is not None and len(owned) != draft.device_pool.layer_num:
                raise ValueError(
                    f"packed draft {name.value} owns buffers on {len(owned)} of "
                    f"{draft.device_pool.layer_num} layers; every packed layer "
                    "must own its buffer"
                )
    return tuple(draft_pools)


def layout_root(decls: tuple[HostPoolDecl, ...]) -> HostPoolDecl:
    """The one declaration whose host pool decides the group's capacity."""
    roots = [d for d in decls if d.is_layout_root]
    if len(roots) != 1:
        raise ValueError(
            f"expected exactly one layout root, got {[d.pool_name for d in roots]}"
        )
    return roots[0]


def _find_pool_decl(decls: tuple[HostPoolDecl, ...], name: PoolName) -> HostPoolDecl:
    return next(d for d in decls if d.pool_name == name)


def prepare_host_pool_configs(
    *,
    decls: tuple[HostPoolDecl, ...],
    full_layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    packed_draft_pools: tuple[Any, ...] = (),
    index_primary: Optional[PoolName] = None,
) -> tuple[HostPoolBuildConfig, ...]:
    """Bind declarations to a stack. A target group contains its own primary
    KV pool; a draft sidecar group reuses an external ``index_primary``. Either
    way exactly one layout root anchors the others' capacity, and sidecar
    indices come from one real source (HostPoolGroup resolves no chains).

    ``packed_draft_pools`` have passed validate_packed_draft_pools; each config
    carries the draft objects that own its same-named buffers."""
    names = [d.pool_name for d in decls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate host pool names: {names}")
    root = layout_root(decls)
    if index_primary is None:
        if not root.is_primary or root.pool_name != PoolName.KV:
            raise ValueError(
                f"expected the layout root to be the primary KV pool, got {root.pool_name}"
            )
        index_primary = root.pool_name
    elif any(d.is_primary for d in decls):
        raise ValueError("a sidecar group must take every index from the target")
    for d in decls:
        if d.is_primary:
            continue
        if d.indices_from_pool != index_primary:
            raise ValueError(
                f"{d.pool_name}.indices_from_pool must be {index_primary}, "
                f"got {d.indices_from_pool}"
            )
        if d.is_layout_root:
            continue
        if d.layout_source == d.pool_name:
            raise ValueError(f"{d.pool_name}.layout_source must name another pool")
        if d.layout_source not in names:
            raise ValueError(
                f"{d.pool_name} references undeclared pool {d.layout_source}"
            )
    draft_decls = [pool.host_pool_decls() for pool in packed_draft_pools]
    return tuple(
        HostPoolBuildConfig(
            decl=d,
            layer_binding=LayerBinding(
                transfer_to_device=_filter_owned_layer_mapping(
                    full_layer_mapping,
                    d.owned_device_layers,
                    root.device_pool.layer_num,
                ),
                transfer_layer_id_max=transfer_layer_id_max,
            ),
            packed_draft_device_pools=tuple(
                _find_pool_decl(decls_of_draft, d.pool_name).device_pool
                for decls_of_draft in draft_decls
            ),
        )
        for d in decls
    )


def _filter_owned_layer_mapping(
    mapping: dict[int, int],
    owned_device_layers: Optional[tuple[int, ...]],
    target_layer_num: int,
) -> dict[int, int]:
    """Drop transfer layers whose device layer owns no buffer for this pool.
    Packed draft tails (device index >= target layer count) are kept."""
    if owned_device_layers is None:
        return mapping
    owned = set(owned_device_layers)
    return {
        t: dev for t, dev in mapping.items() if dev >= target_layer_num or dev in owned
    }
