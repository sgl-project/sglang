# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/peer_memory.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric peer memory for in-kernel TP reductions."""

import torch
from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

from sglang.srt.layers.mono.runtime.consensus import MonoUnsupported, tp_agree
from sglang.srt.layers.mono.runtime.step_fence import FENCE_BYTES, step_fence


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

    ``bytes`` is the data region, the part cleared between steps; the step
    fence's slots sit past it (``fence``, called after every clear on the same
    stream: ``mailboxes.StepMailboxes.begin_step``). ``debug``: bounded fence
    waits, a rank that never arrived reported by ``missing_ranks``.

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
        *,
        debug: bool = False,
    ):
        self.local = UncachedIpcHeap.alloc_uncached(nbytes + FENCE_BYTES)
        # a view, not an owner: close() frees the memory under it
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self.rank = rank
        self._fence_off = nbytes
        self._epoch = torch.zeros(1, dtype=torch.int64, device=device)
        self._missing = (
            torch.zeros(npes, dtype=torch.int64, device=device) if debug else None
        )
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

    def kernel_args(self) -> dict:
        """A kernel's view of the group, by ABI name: this rank's buffer
        (``sym``), the table of every rank's (``peers``) and the TP ``rank``."""
        return {
            "sym": self.local,
            "peers": self.addresses.data_ptr(),
            "rank": self.rank,
        }

    def fence(self) -> None:
        """The step fence, after this rank's clear of ``bytes`` (same stream)."""
        step_fence(
            self._epoch, self.addresses, self._fence_off, self.rank, self._missing
        )

    def missing_ranks(self) -> list[int]:
        """A debug buffer's ranks the fences since the last call gave up on
        (synchronizes: eager only)."""
        if self._missing is None:
            return []
        missing = [r for r, m in enumerate(self._missing.tolist()) if m]
        self._missing.zero_()
        return missing

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
