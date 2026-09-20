"""Declarations of the host-mirrored state a device pool carries.

A pool declares the states HiCache must mirror (``HostStateDecl``); the
assembler binds each declaration to transfer layers (``HostStatePlan``) and
builds entries from the plan, so every declared state gets an entry by
construction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Protocol

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    SidecarPoolSpec,
)


class StateKind(str, Enum):
    """Pairs target and draft states of the same role (KV with KV, ...)."""

    KV = "kv"
    INDEXER = "indexer"
    SWA = "swa"


class RowFamily(str, Enum):
    TOKEN_ROWS = "token_rows"
    PAGE_ROWS = "page_rows"


class StateLayout(msgspec.Struct, frozen=True, kw_only=True):
    row_family: RowFamily
    # token_rows: bytes per token per layer. page_rows: bytes per page per layer.
    bytes_per_row: int
    dtype: torch.dtype

    def page_stride_bytes(self, page_size: int) -> int:
        if self.row_family is RowFamily.PAGE_ROWS:
            return self.bytes_per_row
        return self.bytes_per_row * page_size

    def host_bytes(self, *, page_num: int, layer_num: int, page_size: int) -> int:
        return page_num * layer_num * self.page_stride_bytes(page_size)


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
        decl: HostStateDecl,
        device_pool: Any,
        anchor_host: Any,
        allocator_type: str,
    ) -> Any: ...


class HostStateDecl(msgspec.Struct, frozen=True, kw_only=True):
    """What a device pool asks HiCache to mirror. Pool-intrinsic: no layer binding."""

    name: PoolName
    kind: StateKind
    # Whose host page indices this state reuses when transferred. None: primary.
    index_source: Optional[PoolName]
    # Whose mirror decides this state's capacity and layout. None: self.
    layout_source: Optional[PoolName]
    layout: StateLayout
    # None only for the primary KV state, which the assembler builds itself.
    mirror: Optional[Any]
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES

    @property
    def is_primary(self) -> bool:
        return self.index_source is None

    def sidecar_spec(self) -> SidecarPoolSpec:
        if self.index_source is None:
            raise ValueError(f"{self.name} is a primary state and has no sidecar spec")
        return SidecarPoolSpec(
            pool_name=self.name,
            indices_from_pool=self.index_source,
            hit_policy=self.hit_policy,
        )


class HostStatePlan(msgspec.Struct, frozen=True, kw_only=True):
    """A declaration bound to a stack: the unit the assembler builds entries from."""

    decl: HostStateDecl
    device_pool: Any
    layers: LayerBinding
    packed_draft_device_pools: tuple[Any, ...] = ()
