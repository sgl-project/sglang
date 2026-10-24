# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/peer_memory.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric peer memory for in-kernel TP reductions."""

import torch
from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

from sglang.srt.layers.mono.runtime.consensus import MonoUnsupported, tp_agree


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address, for a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """One uncached buffer per rank, every rank holding every peer's address
    (``addresses``, int64, indexed by TP rank). What lives where, and why no rank
    overwrites a peer's unread data, is the model's layout's to say.

    ``bytes`` is a torch view of the local buffer (for a runner that clears it).

    The constructor is collective over ``group`` (the handle exchange, then an
    agreement that every rank mapped every peer): every rank must reach it, so a
    runner reaches it only after ``tp_agree``. A failed mapping on any rank raises
    ``MonoUnsupported`` on all of them.
    """

    def __init__(
        self,
        nbytes: int,
        group,
        rank: int,
        npes: int,
        device: torch.device,
    ):
        self.local = UncachedIpcHeap.alloc_uncached(nbytes)
        # a view, not an owner: close() frees the memory under it
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self._opened: list[int] = []
        addresses = [self.local]
        if npes > 1:
            handles = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
            )
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = UncachedIpcHeap.open_mem_handle(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except RuntimeError as err:  # every rank must still reach the agreement
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise MonoUnsupported(
                    f"peer memory handshake failed on {'this' if failure else 'another'}"
                    " TP rank"
                ) from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Close the peer mappings and free the local buffer; idempotent. Only once no
        kernel of any rank can still touch it (the runner is being dropped)."""
        opened, self._opened = self._opened, []
        for base in opened:
            UncachedIpcHeap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            UncachedIpcHeap.free_device_mem(self.local)
            self.local = 0
