"""Shared-memory all-gather of a few small integers among the ranks of one host.

The scheduler loop runs two tiny CPU collectives on every pass (the request
broadcast size and the HiCache ready counts). Over the gloo backend each one
costs a few hundred microseconds even when every rank is ready. When all ranks
of the group share a host, this exchange replaces them with a shared-memory
table.

Every stored element carries the call's sequence tag in its high bits, and
readers check each element on its own. So the protocol needs no ordering
between stores of different elements, only single-copy atomic aligned 64-bit
loads and stores. The table is double-buffered by sequence parity: a rank can
reuse a slot only after every rank has published the next call, which a rank
does only after it has finished reading the previous call.

Enabled by SGLANG_Q35MF_EDGE_SHM_SYNC=1 (default off). Groups that span hosts
keep the stock gloo path.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Sequence
from unittest.mock import patch

import torch.distributed as dist

logger = logging.getLogger(__name__)

_TAG_BITS = 23
_VAL_BITS = 40
_TAG_MASK = (1 << _TAG_BITS) - 1
_VAL_MASK = (1 << _VAL_BITS) - 1


def edge_shm_sync_enabled() -> bool:
    return os.environ.get("SGLANG_Q35MF_EDGE_SHM_SYNC", "0") == "1"


class ShmTagExchange:
    def __init__(self, group: dist.ProcessGroup, kind: str, width: int = 8):
        from sglang.srt.utils.stale_shm_cleanup import make_shm_name

        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.width = width
        self.seq = 0
        self.timeout_s = float(
            os.environ.get("TORCH_DISTRIBUTED_DEFAULT_TIMEOUT", "1800")
        )
        nbytes = 2 * self.world * width * 8
        ranks = dist.get_process_group_ranks(group)
        self._creator = self.rank == 0
        if self._creator:
            self.shm = shared_memory.SharedMemory(
                create=True, size=nbytes, name=make_shm_name(f"edgex{kind}")
            )
            self.shm.buf[:nbytes] = bytes(nbytes)
            dist.broadcast_object_list([self.shm.name], src=ranks[0], group=group)
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=ranks[0], group=group)
            # The creator owns unlinking; do not let this process's tracker unlink it.
            with patch(
                "multiprocessing.resource_tracker.register", lambda *a, **k: None
            ):
                self.shm = shared_memory.SharedMemory(name=recv[0])
        self.table = (ctypes.c_int64 * (2 * self.world * width)).from_buffer(
            self.shm.buf
        )
        dist.barrier(group=group)

    def allgather(self, values: Sequence[int]) -> List[List[int]]:
        """Every rank passes the same number of values, each in [0, 2**40)."""
        n = len(values)
        assert n <= self.width
        self.seq += 1
        tag = self.seq & _TAG_MASK
        tagged = tag << _VAL_BITS
        base = (self.seq & 1) * self.world * self.width
        t = self.table
        mine = base + self.rank * self.width
        for i, v in enumerate(values):
            if not 0 <= v <= _VAL_MASK:
                raise ValueError(f"shm exchange value out of range: {v}")
            t[mine + i] = tagged | v
        out: List[List[int]] = []
        deadline = None
        for r in range(self.world):
            row = base + r * self.width
            vals = []
            for i in range(n):
                x = t[row + i]
                while (x >> _VAL_BITS) != tag:
                    if deadline is None:
                        deadline = time.monotonic() + self.timeout_s
                    elif time.monotonic() > deadline:
                        raise RuntimeError(
                            f"shm exchange timed out after {self.timeout_s:.0f}s waiting "
                            f"for rank {r} (seq {self.seq}); a peer scheduler is stuck or dead"
                        )
                    os.sched_yield()
                    x = t[row + i]
                vals.append(x & _VAL_MASK)
            out.append(vals)
        return out

    def close(self):
        try:
            del self.table
            self.shm.close()
            if self._creator:
                self.shm.unlink()
        except Exception:
            pass


_EXCHANGES: Dict[tuple, Optional[ShmTagExchange]] = {}


def get_shm_exchange(
    group: Optional[dist.ProcessGroup], kind: str
) -> Optional[ShmTagExchange]:
    """Collective on first call per (group, kind): every rank of the group must call it
    at the same point. Returns None (stock path) when disabled or the group spans hosts."""
    if group is None or not edge_shm_sync_enabled():
        return None
    key = (id(group), kind)
    if key not in _EXCHANGES:
        if dist.get_world_size(group) <= 1:
            _EXCHANGES[key] = None
        else:
            from sglang.srt.distributed.parallel_state import in_the_same_node_as

            same = in_the_same_node_as(group, source_rank=0)
            if all(same):
                _EXCHANGES[key] = ShmTagExchange(group, kind)
                logger.info(
                    "EDGE shm exchange '%s' enabled for a %d-rank group",
                    kind,
                    dist.get_world_size(group),
                )
            else:
                _EXCHANGES[key] = None
    return _EXCHANGES[key]


def fold_u64(x: int) -> List[int]:
    """Split an arbitrary int into two 32-bit halves of its low 64 bits."""
    x &= (1 << 64) - 1
    return [x & 0xFFFFFFFF, x >> 32]
