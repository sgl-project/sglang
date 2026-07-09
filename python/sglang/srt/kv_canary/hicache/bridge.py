from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.hicache.transfer import copy_canary_buffers_indexed
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheController


class CanaryHiCacheBridge:
    """L2 sidecar storage for the existing L1 canary buffers."""

    def __init__(
        self,
        *,
        buffer_groups: Sequence[CanaryBufferGroup],
        host_sizes: Mapping[PoolKind, int],
        pin_memory: bool,
    ) -> None:
        self._device_groups: dict[PoolKind, CanaryBufferGroup] = {}
        self._host_buffers: dict[PoolKind, tuple[torch.Tensor, ...]] = {}

        for group in buffer_groups:
            if group.kind in self._device_groups:
                raise ValueError(
                    f"kv-canary: duplicate HiCache buffer group {group.kind.name}"
                )
            device_buffers = _group_buffers(group)
            try:
                host_size = int(host_sizes[group.kind])
            except KeyError as exc:
                raise ValueError(
                    f"kv-canary: missing HiCache host size for {group.kind.name}"
                ) from exc
            if host_size <= 0:
                raise ValueError(
                    f"kv-canary: HiCache host size must be positive, got {host_size}"
                )
            slot_bytes = int(device_buffers[0].shape[1])
            self._device_groups[group.kind] = group
            self._host_buffers[group.kind] = tuple(
                torch.empty(
                    (host_size, slot_bytes),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for _ in device_buffers
            )

    @classmethod
    def from_cache_controller(
        cls,
        *,
        buffer_groups: Sequence[CanaryBufferGroup],
        cache_controller: HiCacheController,
    ) -> CanaryHiCacheBridge:
        if cache_controller.enable_storage:
            raise NotImplementedError(
                "kv-canary: HiCache L3 storage does not yet preserve canary sidecars"
            )

        host = cache_controller.mem_pool_host
        if hasattr(host, "entry_map"):
            full_host_pool = host.entry_map[PoolName.KV].host_pool
            host_sizes = {PoolKind.FULL: int(full_host_pool.size)}
            if any(group.kind is PoolKind.SWA for group in buffer_groups):
                host_sizes[PoolKind.SWA] = int(
                    host.entry_map[PoolName.SWA].host_pool.size
                )
            pin_memory = bool(full_host_pool.pin_memory)
        else:
            host_sizes = {PoolKind.FULL: int(host.size)}
            pin_memory = bool(host.pin_memory)

        return cls(
            buffer_groups=buffer_groups,
            host_sizes=host_sizes,
            pin_memory=pin_memory,
        )

    def backup(
        self,
        *,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
        pool_transfers: Optional[Sequence[PoolTransfer]] = None,
    ) -> None:
        self._copy(
            to_host=True,
            host_indices=host_indices,
            device_indices=device_indices,
            io_backend=io_backend,
            pool_transfers=pool_transfers,
        )

    def restore(
        self,
        *,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
        pool_transfers: Optional[Sequence[PoolTransfer]] = None,
    ) -> None:
        self._copy(
            to_host=False,
            host_indices=host_indices,
            device_indices=device_indices,
            io_backend=io_backend,
            pool_transfers=pool_transfers,
        )

    def destroy(self) -> None:
        self._host_buffers.clear()

    def _copy(
        self,
        *,
        to_host: bool,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
        pool_transfers: Optional[Sequence[PoolTransfer]],
    ) -> None:
        for kind, group in self._device_groups.items():
            kind_host_indices, kind_device_indices = _indices_for_kind(
                kind=kind,
                host_indices=host_indices,
                device_indices=device_indices,
                pool_transfers=pool_transfers,
            )
            device_buffers = _group_buffers(group)
            host_buffers = self._host_buffers[kind]
            copy_canary_buffers_indexed(
                src_buffers=device_buffers if to_host else host_buffers,
                dst_buffers=host_buffers if to_host else device_buffers,
                src_indices=kind_device_indices if to_host else kind_host_indices,
                dst_indices=kind_host_indices if to_host else kind_device_indices,
                io_backend=io_backend,
            )


def _group_buffers(group: CanaryBufferGroup) -> tuple[torch.Tensor, ...]:
    if group.v_head is None or group.v_tail is None:
        raise NotImplementedError(
            "kv-canary: HiCache canary sidecars currently require K/V buffer halves"
        )
    return group.k_head, group.k_tail, group.v_head, group.v_tail


def _indices_for_kind(
    *,
    kind: PoolKind,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    pool_transfers: Optional[Sequence[PoolTransfer]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind is PoolKind.FULL:
        return host_indices, device_indices

    for transfer in pool_transfers or ():
        if transfer.name == PoolName.SWA:
            if transfer.host_indices is None or transfer.device_indices is None:
                break
            return transfer.host_indices, transfer.device_indices
    raise RuntimeError("kv-canary: missing SWA PoolTransfer for HiCache canary")
