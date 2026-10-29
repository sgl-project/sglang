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
keep the stock gloo path unless SGLANG_Q35MF_EDGE_SHM_SYNC_MULTIHOST=1 is also
set: then each host exchanges through shared memory and only the host leaders
meet in one small gloo all_gather (HierTagExchange). The leaders swap rows over
TCP instead with SGLANG_Q35MF_EDGE_MULTIHOST_TCP=1 (two hosts) or
SGLANG_Q35MF_EDGE_MULTIHOST_TCP_MESH=1 (more than two hosts), and the request
broadcast's size takes the same path with SGLANG_Q35MF_EDGE_RECV_MULTIHOST=1.
"""

from __future__ import annotations

import ctypes
import logging
import os
import socket
import time
from array import array
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Sequence
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_device

logger = logging.getLogger(__name__)

_TAG_BITS = 23
_VAL_BITS = 40
_TAG_MASK = (1 << _TAG_BITS) - 1
_VAL_MASK = (1 << _VAL_BITS) - 1


def _leader_mesh_enabled() -> bool:
    return os.environ.get("SGLANG_Q35MF_EDGE_MULTIHOST_TCP_MESH", "0") == "1"


def recv_multihost_enabled() -> bool:
    return os.environ.get("SGLANG_Q35MF_EDGE_RECV_MULTIHOST", "0") == "1"


class ShmTagExchange:
    def __init__(
        self,
        group: dist.ProcessGroup,
        kind: str,
        width: int = 8,
        rows: Optional[int] = None,
    ):
        from sglang.srt.utils.stale_shm_cleanup import make_shm_name

        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.width = width
        # One row per rank for allgather; publish_rows / read_rows may use any row count.
        self.rows = self.world if rows is None else rows
        self.seq = 0
        self.timeout_s = float(
            os.environ.get("TORCH_DISTRIBUTED_DEFAULT_TIMEOUT", "1800")
        )
        nbytes = 2 * self.rows * width * 8
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
        self.table = (ctypes.c_int64 * (2 * self.rows * width)).from_buffer(
            self.shm.buf
        )
        dist.barrier(group=group)

    def allgather(self, values: Sequence[int]) -> List[List[int]]:
        """Every rank passes the same number of values, each in [0, 2**40)."""
        n = len(values)
        assert n <= self.width and self.rows == self.world
        self.seq += 1
        tag = self.seq & _TAG_MASK
        base = (self.seq & 1) * self.rows * self.width
        self._store(base + self.rank * self.width, values, tag)
        return self._wait_rows(base, n, tag)

    def publish_rows(self, table: Sequence[Sequence[int]]) -> None:
        """Single writer: stores every row of this call; the other ranks call read_rows."""
        self.seq += 1
        tag = self.seq & _TAG_MASK
        base = (self.seq & 1) * self.rows * self.width
        for r, row in enumerate(table):
            self._store(base + r * self.width, row, tag)

    def read_rows(self, n: int) -> List[List[int]]:
        self.seq += 1
        tag = self.seq & _TAG_MASK
        return self._wait_rows((self.seq & 1) * self.rows * self.width, n, tag)

    def _store(self, start: int, values: Sequence[int], tag: int) -> None:
        tagged = tag << _VAL_BITS
        for i, v in enumerate(values):
            if not 0 <= v <= _VAL_MASK:
                raise ValueError(f"shm exchange value out of range: {v}")
            self.table[start + i] = tagged | v

    def _wait_rows(self, base: int, n: int, tag: int) -> List[List[int]]:
        t = self.table
        out: List[List[int]] = []
        deadline = None
        for r in range(self.rows):
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


class _LeaderLink:
    """Persistent TCP connection between the two host leaders; one send and one exact
    receive per call (payloads are a few hundred bytes, so the sends never block)."""

    def __init__(self, leaders: dist.ProcessGroup, first: bool, timeout_s: float):
        from sglang.srt.utils.network import get_local_ip_auto

        server = None
        box: List[Optional[tuple]] = [None]
        if first:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("0.0.0.0", 0))
            server.listen(1)
            box[0] = (get_local_ip_auto(), server.getsockname()[1])
        src = dist.get_process_group_ranks(leaders)[0]
        dist.broadcast_object_list(box, src=src, group=leaders)
        if first:
            server.settimeout(timeout_s)
            self.sock, _ = server.accept()
            server.close()
        else:
            self.sock = socket.create_connection(box[0], timeout=timeout_s)
        self.sock.settimeout(timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.timeout_s = timeout_s
        # SGLANG_PERFLL_HOST_LINK_SPIN: spin on a non-blocking socket (no sleep / wakeup)
        self.spin = envs.SGLANG_PERFLL_HOST_LINK_SPIN.get()
        if self.spin:
            self.sock.setblocking(False)
            logger.info("PERF-LL host: LINK_SPIN on")

    def swap(self, payload: bytes) -> bytes:
        msg = len(payload).to_bytes(4, "little") + payload
        if self.spin:
            return self._spin_swap(msg)
        self.sock.sendall(msg)
        size = int.from_bytes(self._recv_exact(4), "little")
        return self._recv_exact(size)

    def _spin_swap(self, msg: bytes) -> bytes:
        deadline = time.monotonic() + self.timeout_s
        view = memoryview(msg)
        while view:
            try:
                view = view[self.sock.send(view) :]
            except BlockingIOError:
                self._spin_wait(deadline)
        size = int.from_bytes(self._spin_recv(4, deadline), "little")
        return self._spin_recv(size, deadline)

    def _spin_recv(self, n: int, deadline: float) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except BlockingIOError:
                self._spin_wait(deadline)
                continue
            if not chunk:
                raise RuntimeError("EDGE leader link closed by the peer leader")
            buf += chunk
        return bytes(buf)

    def _spin_wait(self, deadline: float) -> None:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"EDGE leader link timed out after {self.timeout_s:.0f}s; "
                "the peer leader's scheduler is stuck or dead"
            )
        os.sched_yield()

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("EDGE leader link closed by the peer leader")
            buf += chunk
        return bytes(buf)


class _LeaderMesh:
    """Persistent TCP connections between every pair of host leaders (more than two
    hosts). A call sends this leader's rows to every peer, then spins on the peer
    sockets until each has delivered its rows: one network hop and no thread wakeups,
    where a gloo all_gather walks a ring through its worker threads."""

    def __init__(self, leaders: dist.ProcessGroup, timeout_s: float):
        from sglang.srt.utils.network import get_local_ip_auto

        n = dist.get_world_size(leaders)
        me = dist.get_rank(leaders)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", 0))
        server.listen(n)
        addrs: List[Optional[tuple]] = [None] * n
        dist.all_gather_object(
            addrs, (get_local_ip_auto(), server.getsockname()[1]), group=leaders
        )
        # Leader j connects to every leader i < j and says who it is.
        self.peers: Dict[int, socket.socket] = {}
        for i in range(me):
            sock = socket.create_connection(addrs[i], timeout=timeout_s)
            sock.sendall(me.to_bytes(4, "little"))
            self.peers[i] = sock
        server.settimeout(timeout_s)
        for _ in range(n - 1 - me):
            sock, _ = server.accept()
            sock.settimeout(timeout_s)
            j = int.from_bytes(_recv_exact(sock, 4), "little")
            self.peers[j] = sock
        server.close()
        for sock in self.peers.values():
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
        self.me = me
        self.n = n
        self.timeout_s = timeout_s

    def all_gather(self, payload: bytes) -> List[bytes]:
        msg = len(payload).to_bytes(4, "little") + payload
        deadline = time.monotonic() + self.timeout_s
        for sock in self.peers.values():
            view = memoryview(msg)
            while view:
                try:
                    view = view[sock.send(view) :]
                except BlockingIOError:
                    self._wait(deadline)
        out: List[bytes] = []
        for i in range(self.n):
            if i == self.me:
                out.append(payload)
            else:
                size = int.from_bytes(self._spin_recv(i, 4, deadline), "little")
                out.append(self._spin_recv(i, size, deadline))
        return out

    def _spin_recv(self, peer: int, n: int, deadline: float) -> bytes:
        sock = self.peers[peer]
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except BlockingIOError:
                self._wait(deadline, peer)
                continue
            if not chunk:
                raise RuntimeError(f"EDGE leader link to leader {peer} closed")
            buf += chunk
        return bytes(buf)

    def _wait(self, deadline: float, peer: int = -1) -> None:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"EDGE leader mesh timed out after {self.timeout_s:.0f}s waiting for "
                f"leader {peer}; a peer scheduler is stuck or dead"
            )
        os.sched_yield()


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("EDGE leader link closed by a peer leader")
        buf += chunk
    return bytes(buf)


class HierTagExchange:
    """ShmTagExchange.allgather for a group that spans hosts: ranks exchange through
    shared memory on their host, and only the host leaders run one gloo all_gather."""

    def __init__(self, group: dist.ProcessGroup, kind: str, width: int = 8):
        ranks = dist.get_process_group_ranks(group)
        me = dist.get_rank(group)
        hostnames: List[Optional[str]] = [None] * len(ranks)
        dist.all_gather_object(hostnames, socket.gethostname(), group=group)
        hosts = list(dict.fromkeys(hostnames))
        # Group ranks per host, in group-rank order (new_group keeps that order).
        self.members = [
            [i for i, h in enumerate(hostnames) if h == host] for host in hosts
        ]
        # new_group is collective over the default group: every rank creates every subgroup.
        local_groups = [
            dist.new_group([ranks[i] for i in m], backend="gloo") for m in self.members
        ]
        self.leaders = dist.new_group(
            [ranks[m[0]] for m in self.members], backend="gloo"
        )
        host = hosts.index(hostnames[me])
        self.is_leader = me == self.members[host][0]
        self.host = host
        self.world = len(ranks)
        self.width = width
        self.local_max = max(len(m) for m in self.members)
        self.local = ShmTagExchange(local_groups[host], kind + "l", width)
        self.full = ShmTagExchange(
            local_groups[host], kind + "f", width, rows=self.world
        )
        # Two hosts: the leaders swap rows over one TCP connection instead of a gloo all_gather.
        self.link = None
        self.mesh = None
        if (
            self.is_leader
            and len(self.members) == 2
            and envs.SGLANG_Q35MF_EDGE_MULTIHOST_TCP.get()
        ):
            self.link = _LeaderLink(self.leaders, host == 0, self.local.timeout_s)
        elif self.is_leader and len(self.members) > 2 and _leader_mesh_enabled():
            self.mesh = _LeaderMesh(self.leaders, self.local.timeout_s)

    def allgather(self, values: Sequence[int]) -> List[List[int]]:
        n = len(values)
        rows_here = self.local.allgather(values)
        if not self.is_leader:
            return self.full.read_rows(n)
        mine = [v for row in rows_here for v in row]
        if self.link is not None:
            peer = array("q")
            peer.frombytes(self.link.swap(array("q", mine).tobytes()))
            per_host = (
                [mine, peer.tolist()] if self.host == 0 else [peer.tolist(), mine]
            )
        elif self.mesh is not None:
            per_host = []
            for raw in self.mesh.all_gather(array("q", mine).tobytes()):
                rows = array("q")
                rows.frombytes(raw)
                per_host.append(rows.tolist())
        else:
            flat = torch.zeros(self.local_max * n, dtype=torch.int64)
            flat[: len(mine)] = torch.tensor(mine, dtype=torch.int64)
            outs = [torch.empty_like(flat) for _ in self.members]
            dist.all_gather(outs, flat, group=self.leaders)
            per_host = [out.tolist() for out in outs]
        table: List[List[int]] = [[] for _ in range(self.world)]
        for m, vals in zip(self.members, per_host):
            for j, g in enumerate(m):
                table[g] = vals[j * n : (j + 1) * n]
        self.full.publish_rows(table)
        return table


_EXCHANGES: Dict[tuple, object] = {}
# (values_fn, store_fn): extra values one consumer adds to the request-size exchange
# (SGLANG_PERFLL_HOST_HC_PIGGYBACK). values_fn returns the same-length list on every
# rank or None; store_fn gets every rank's extra values, or None when not available.
_RECV_PIGGYBACK: Optional[tuple] = None


def set_recv_piggyback(values_fn, store_fn) -> None:
    global _RECV_PIGGYBACK
    _RECV_PIGGYBACK = (values_fn, store_fn)


def recv_piggyback() -> Optional[tuple]:
    return _RECV_PIGGYBACK


def get_shm_exchange(
    group: Optional[dist.ProcessGroup], kind: str, multihost: bool = False
):
    """Collective on first call per (group, kind): every rank of the group must call it
    at the same point. Returns None (stock path) when disabled, under --sleep-on-idle,
    or when the group spans hosts and the caller did not opt in with multihost=True."""
    if group is None:
        return None
    key = (id(group), kind)
    if key not in _EXCHANGES:
        if (
            not envs.SGLANG_Q35MF_EDGE_SHM_SYNC.get()
            # an idle rank 0 sleeps in its poller; the waiting ranks would spin meanwhile
            or get_device().sleep_on_idle
            or dist.get_world_size(group) <= 1
        ):
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
            elif (
                multihost
                and envs.SGLANG_Q35MF_EDGE_SHM_SYNC_MULTIHOST.get()
                and dist.get_world_size() == dist.get_world_size(group)
            ):
                # Only when this group is the whole default group, so the subgroup
                # creation inside is collective over exactly these ranks.
                _EXCHANGES[key] = HierTagExchange(group, kind)
                logger.info(
                    "EDGE hierarchical exchange '%s' enabled for a %d-rank group on %d hosts",
                    kind,
                    dist.get_world_size(group),
                    len(_EXCHANGES[key].members),
                )
            else:
                _EXCHANGES[key] = None
    return _EXCHANGES[key]


def fold_u64(x: int) -> List[int]:
    """Split an arbitrary int into two 32-bit halves of its low 64 bits."""
    x &= (1 << 64) - 1
    return [x & 0xFFFFFFFF, x >> 32]
