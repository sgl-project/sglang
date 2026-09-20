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
    layout: HostPoolLayout
    # None only for the primary KV pool, which the assembler builds itself.
    mirror: Optional[MirrorAdapter]
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES

    @property
    def is_primary(self) -> bool:
        return self.index_source is None

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


def plan_host_pools(
    *,
    decls: tuple[HostPoolDecl, ...],
    device_pool: Any,
    full_layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    packed_draft_device_pools: tuple[Any, ...],
) -> tuple[HostPoolPlan, ...]:
    """Bind declarations to a stack after checking they form one primary KV
    pool plus sidecars that resolve their indices from it."""
    names = [d.name for d in decls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate host pool names: {names}")
    primaries = [d for d in decls if d.is_primary]
    if len(primaries) != 1 or primaries[0].name != PoolName.KV:
        raise ValueError(
            f"expected exactly one primary KV pool, got {[d.name for d in primaries]}"
        )
    primary = primaries[0].name
    for d in decls:
        if d.is_primary:
            continue
        # HostPoolGroup resolves sidecar indices from one real source, so no
        # self-reference and no sidecar-to-sidecar chains.
        if d.index_source != primary:
            raise ValueError(
                f"{d.name}.index_source must be the primary pool {primary}, "
                f"got {d.index_source}"
            )
        if d.layout_source is None or d.layout_source == d.name:
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
