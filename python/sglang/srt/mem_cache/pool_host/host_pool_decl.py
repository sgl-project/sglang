"""Declarations of the host pools a device pool needs HiCache to keep.

A device pool declares them (``HostPoolDecl``) through ``host_pool_decls()``;
the assembler (hybrid_pool_assembler) binds each declaration to transfer layers
and builds one entry per declaration. This module holds only the declaration
types and their basic constructors.
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


class HostPoolBuilder(Protocol):
    def validate(
        self,
        *,
        decl: HostPoolDecl,
        page_size: int,
        packed_draft_device_pools: tuple[Any, ...],
    ) -> None: ...

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


def make_kv_pool_decl(pool: Any) -> HostPoolDecl:
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
