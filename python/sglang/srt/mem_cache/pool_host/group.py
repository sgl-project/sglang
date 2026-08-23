from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer


@dataclass
class PoolEntry:
    name: PoolName
    host_pool: Any
    device_pool: Any
    layer_mapper: Callable[[int], int | None]
    is_primary_index_anchor: bool = False
    host_evict_fn: Callable[[int], Any] | None = None
    device_evict_fn: Callable[[int], Any] | None = None
    device_alloc_fn: Callable[[int], Any] | None = None
    device_free_fn: Callable[[Any], Any] | None = None
    packed_draft_device_pools: tuple[Any, ...] = ()


class HostPoolGroup:
    """Allocation facade for an anchor host pool and its side pools."""

    def __init__(self, entries: list[PoolEntry]):
        if not entries:
            raise ValueError("HostPoolGroup requires at least one pool entry.")
        if len({entry.name for entry in entries}) != len(entries):
            raise ValueError("HostPoolGroup pool names must be unique.")

        anchors = [entry for entry in entries if entry.is_primary_index_anchor]
        if len(anchors) > 1:
            raise ValueError("HostPoolGroup requires at most one anchor pool.")

        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        self.anchor_entry = anchors[0] if anchors else entries[0]

        self.layout = self.anchor_entry.host_pool.layout
        self.page_size = self.anchor_entry.host_pool.page_size
        self.device = self.anchor_entry.host_pool.device
        self.size = self.anchor_entry.host_pool.size
        self.logical_size = self.anchor_entry.host_pool.logical_size
        self._refresh_transfer_capabilities()

    def _refresh_transfer_capabilities(self) -> None:
        child_write_back_jit = [
            entry.host_pool.can_use_write_back_jit for entry in self.entries
        ]
        self.can_use_write_back_jit = all(child_write_back_jit)
        self.supports_per_pool_backup_indices = any(child_write_back_jit)

    def add_entry(self, entry: PoolEntry) -> None:
        if entry.name in self.entry_map:
            raise ValueError(f"Host pool {entry.name} is already registered.")
        if entry.is_primary_index_anchor:
            raise ValueError("Cannot replace the anchor of an existing HostPoolGroup.")
        self.entries.append(entry)
        self.entry_map[entry.name] = entry
        self._refresh_transfer_capabilities()

    def get_entry(self, name: PoolName | None = None) -> PoolEntry:
        return self.anchor_entry if name is None else self.entry_map[name]

    def get_pool(self, name: PoolName):
        return self.get_entry(name).host_pool

    def alloc(
        self,
        need_size: int,
        *,
        pool: PoolName | None = None,
        reclaim: Callable[[int], Any] | None = None,
    ) -> torch.Tensor | None:
        """Allocate from one pool, optionally reclaiming once before retrying."""
        host_pool = self.get_entry(pool).host_pool
        indices = host_pool.alloc(need_size)
        if indices is None and reclaim is not None:
            reclaim(need_size)
            indices = host_pool.alloc(need_size)
        return indices

    def free(self, indices: torch.Tensor, *, pool: PoolName | None = None) -> int:
        return self.get_entry(pool).host_pool.free(indices)

    def resolve_host_transfers(
        self,
        transfers: list[PoolTransfer] | None,
        *,
        primary_device_indices: torch.Tensor | None = None,
        primary_host_indices: torch.Tensor | None = None,
    ) -> list[PoolTransfer] | None:
        """Allocate unresolved side-pool host indices atomically.

        On failure, every allocation made by this call is released and the
        corresponding transfer is restored to its unresolved state.
        """
        if not transfers:
            return None

        allocated: list[tuple[PoolTransfer, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback() -> None:
            for transfer, indices in allocated:
                self.free(indices, pool=transfer.name)
                transfer.host_indices = None

        for transfer in transfers:
            if transfer.indices_from_pool is not None:
                derived_transfers.append(transfer)
                continue
            if transfer.host_indices is not None or transfer.device_indices is None:
                continue
            entry = self.entry_map.get(transfer.name)
            if entry is None:
                continue
            indices = self.alloc(
                len(transfer.device_indices),
                pool=transfer.name,
                reclaim=entry.host_evict_fn,
            )
            if indices is None:
                rollback()
                return None
            transfer.host_indices = indices
            allocated.append((transfer, indices))

        for transfer in derived_transfers:
            if transfer.indices_from_pool == self.anchor_entry.name:
                transfer.host_indices = primary_host_indices
                transfer.device_indices = primary_device_indices
                continue

            source = next(
                (
                    candidate
                    for candidate in transfers
                    if candidate.indices_from_pool is None
                    and candidate.name == transfer.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback()
                return None
            transfer.host_indices = source.host_indices
            transfer.device_indices = source.device_indices
        return transfers

    def release_transfers(self, transfers: list[PoolTransfer] | None) -> int:
        """Release independently allocated side-pool indices.

        Derived transfers share another pool's indices and are deliberately
        skipped so each allocation is released exactly once.
        """
        released = 0
        for transfer in transfers or []:
            if transfer.indices_from_pool is not None or transfer.host_indices is None:
                continue
            released += self.free(transfer.host_indices, pool=transfer.name)
        return released

    @property
    def size_per_token(self):
        return self.anchor_entry.host_pool.size_per_token

    def clear(self) -> None:
        for entry in self.entries:
            entry.host_pool.clear()

    def destroy(self) -> None:
        for entry in self.entries:
            entry.host_pool.destroy()

    def available_size(self, pool: PoolName | None = None):
        return self.get_entry(pool).host_pool.available_size()
