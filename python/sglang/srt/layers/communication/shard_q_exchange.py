"""Symmetric-memory workspace for the DSA prefill shard-Q exchange.

One workspace per head-sharded group: a q buffer that receives every rank's
heads of this rank's row block, and an output buffer that receives this rank's
heads of every row. Peers push straight into them (kernels/ops/communication/
shard_q_push.py); a barrier before a push keeps a peer from overwriting rows
its owner still reads, and one after it publishes the writes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from sglang.kernels.ops.communication.shard_q_push import (
    push_cols_to_peers,
    push_rows_to_peers,
)


class ShardQExchange:
    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self._buffers: dict[str, tuple[torch.Tensor, object, torch.Tensor]] = {}

    def _buffer(self, name: str, nbytes: int, device: torch.device):
        """Symmetric byte buffer ``name`` of at least ``nbytes``, with its
        handle and every rank's base address. Growing it is collective."""
        entry = self._buffers.get(name)
        if entry is None or entry[0].numel() < nbytes:
            if entry is not None:
                entry[1].barrier(channel=0)
                torch.cuda.current_stream().synchronize()
            storage = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
            handle = symm_mem.rendezvous(storage, self.group)
            ptrs = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
            entry = (storage, handle, ptrs)
            self._buffers[name] = entry
        return entry

    def exchange_q(
        self, q: torch.Tensor, bounds: torch.Tensor, *, max_count: int, count: int
    ) -> torch.Tensor:
        """``[rows, H, D]`` head-sharded q -> this rank's ``count`` rows of
        ``bounds`` (device int32, ``world + 1``) with every rank's heads,
        ``[count, world * H, D]`` (a view of the buffer)."""
        _, heads, dim = q.shape
        row_bytes = heads * dim * q.element_size()
        storage, handle, ptrs = self._buffer(
            "q", max(max_count, 1) * self.world * row_bytes, q.device
        )
        handle.barrier(channel=0)
        push_rows_to_peers(
            q.contiguous(),
            ptrs,
            bounds,
            max_count=max_count,
            dst_stride_bytes=self.world * row_bytes,
            dst_col_bytes=self.rank * row_bytes,
        )
        handle.barrier(channel=0)
        return (
            storage[: count * self.world * row_bytes]
            .view(q.dtype)
            .view(count, self.world * heads, dim)
        )

    def exchange_out(self, out: torch.Tensor, *, row0: int, rows: int) -> torch.Tensor:
        """This rank's ``[count, world * H, Dv]`` rows, starting at batch row
        ``row0`` -> its heads of every row, ``[rows, H, Dv]`` (a view of the
        buffer, valid until the next call)."""
        count, all_heads, dim = out.shape
        heads = all_heads // self.world
        row_bytes = heads * dim * out.element_size()
        storage, handle, ptrs = self._buffer("out", rows * row_bytes, out.device)
        handle.barrier(channel=0)
        push_cols_to_peers(out.view(count, self.world, heads, dim), ptrs, row0=row0)
        handle.barrier(channel=0)
        return storage[: rows * row_bytes].view(out.dtype).view(rows, heads, dim)


_exchanges: dict[int, Optional[ShardQExchange]] = {}


def get_shard_q_exchange(group: dist.ProcessGroup) -> Optional[ShardQExchange]:
    """The group's exchange, or None without symmetric memory (NCCL fallback)."""
    key = id(group)
    if key not in _exchanges:
        try:
            probe = symm_mem.empty(16, dtype=torch.uint8, device="cuda")
            symm_mem.rendezvous(probe, group)
            _exchanges[key] = ShardQExchange(group)
        except Exception:
            _exchanges[key] = None
    return _exchanges[key]
